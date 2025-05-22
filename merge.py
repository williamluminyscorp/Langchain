import json
import uuid
import hashlib
import time
import numpy as np
from fastapi import FastAPI, HTTPException
import uvicorn
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import logging
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
from typing import Optional,List
import os
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = None
        self.model_name = model_name
        self.load_time = 0
        self.inference_times = []
        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),  # 从环境变量读取，默认值localhost
            port=int(os.getenv("QDRANT_PORT", 6333))     # 从环境变量读取，默认值6333
        )
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")          # 从环境变量读取
        )
    def load(self):
        """加载并预热模型"""
        if self.model is None:
            start = time.time()
            logger.info(f"🔄 正在加载 {self.model_name} 模型...")
            
            self.model = SentenceTransformer(
                self.model_name,
                device="cpu",  # 使用CPU
                # quantize=True  # 如需量化取消注释(需安装onnxruntime)
            )
            
            # 预热模型
            self.model.encode(["warmup text"], batch_size=1)
            self.load_time = time.time() - start
            logger.info(f"✅ 模型加载完成，耗时: {self.load_time:.1f}秒")
    
    def embed(self, texts, normalize=True):
        """生成embedding并记录性能"""
        if not isinstance(texts, list):
            texts = [texts]
            
        start_time = time.time()
        embeddings = self.model.encode(texts, normalize_embeddings=normalize)
        elapsed = time.time() - start_time
        self.inference_times.append(elapsed)
        return embeddings
    
    def get_stats(self):
        """获取性能统计"""
        return {
            "model": self.model_name,
            "load_time": self.load_time,
            "avg_inference": np.mean(self.inference_times) if self.inference_times else 0,
            "total_requests": len(self.inference_times)
        }
    def configure_openai(self, api_key: str):
        """配置OpenAI客户端"""
        self.openai_client = OpenAI(api_key=api_key)
        logger.info("✅ OpenAI客户端已配置")
    
    def search_similar_texts(self, query_vec: List[float], collection_name: str, top_k: int = 3):
        try:
            results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vec,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload["text"],
                    "payload": hit.payload  # 完整payload
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

    def store_embeddings(self, texts, collection_name="pdf_chunks"):
        """存储embedding到Qdrant"""
        if not isinstance(texts, list):
            texts = [texts]
            
        embeddings = self.embed(texts)
        content_hashes = [hashlib.md5(text.encode('utf-8')).hexdigest() for text in texts]
        
        # Get or create collection
        try:
            collections = self.qdrant_client.get_collections()
            if collection_name not in [col.name for col in collections.collections]:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
                )
                # 创建集合后立即建索引
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="content_hash",
                    field_schema="keyword",
                    wait=True
                )
                logger.info(f"Created new collection '{collection_name}' with index")
            
            # Prepare points for upsert
            points = []
            for text, embedding, content_hash in zip(texts, embeddings, content_hashes):
                # Check for duplicates
                existing_points = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="content_hash", match=MatchValue(value=content_hash))]
                    ),
                    limit=1,
                    with_payload=["content_hash"],
                    with_vectors=False
                )
                
                if not existing_points[0]:  # Only insert if not exists
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding.tolist(),
                        payload={"text": text, "content_hash": content_hash}
                    ))
            
            if points:
                response = self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True
                )
                logger.info(f"Inserted {len(points)} embeddings successfully! Status: {response.status}")
                return True
            else:
                logger.info("All texts already exist in the collection")
                return False
                
        except Exception as e:
            logger.error(f"Storage operation failed: {e}")
            return False
        
    def generate_answer(self, query: str, context: str, model: str = "gpt-3.5-turbo", temperature: float = 0.3):
        """使用GPT生成回答"""
        prompt = f"""你是一个专业的公司知识助手，请严格根据以下公司手册内容回答问题：
        
        【相关手册内容】
        {context}

        【用户问题】
        {query}

        要求：
        1. 回答必须完全基于提供的手册内容
        2. 如果内容中没有明确答案，请回答"根据手册内容，没有找到相关信息"
        3. 保持回答专业、简洁
        """

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content
    

class CreateCollectionRequest(BaseModel):
    collection_name: str
    vector_size: Optional[int] = None  # 可选，不指定则使用模型默认维度
    distance: Optional[str] = "COSINE"  # 默认为COSINE相似度

# 添加这个类定义
class QueryRequest(BaseModel):
    question: str
    collection_name: str = "pdf_chunks"
    top_k: int = 3
    openai_model: str = "gpt-3.5-turbo"
    temperature: float = 0.3

# Initialize FastAPI application
app = FastAPI()
service = EmbeddingService()
service.load()

@app.post("/embed")
async def embed(texts: list[str]):
    """API endpoint to generate embeddings"""
    return {"embeddings": service.embed(texts).tolist()}

@app.post("/store")
async def store(texts: list[str], collection: str = "pdf_chunks"):
    """API endpoint to store embeddings"""
    success = service.store_embeddings(texts, collection_name=collection)
    return {"status": "success" if success else "failed"}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    完整问答流程:
    1. 问题向量化
    2. 搜索相似内容
    3. 调用OpenAI生成回答
    返回: 回答 + 搜索结果详情
    """
    try:
        # 1. 生成问题向量
        query_vec = service.embed(request.question).tolist()[0]
        
        # 2. 搜索相似内容（返回完整结果）
        search_results = service.search_similar_texts(
            query_vec=query_vec,
            collection_name=request.collection_name,
            top_k=request.top_k
        )
        
        # 提取文本用于生成回答
        context = "\n\n".join([hit["text"] for hit in search_results])
        
        # 3. 生成回答
        answer = service.generate_answer(
            query=request.question,
            context=context,
            model=request.openai_model,
            temperature=request.temperature
        )
        
        return {
            "status": "success",
            "answer": answer,
            "search_results": search_results,  # 包含ID、得分、文本等
            "used_model": request.openai_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_collection")
async def create_collection(request: CreateCollectionRequest):
    """创建新的向量集合"""
    try:
        # 检查集合是否已存在
        collections = service.qdrant_client.get_collections()
        if request.collection_name in [col.name for col in collections.collections]:
            return {"status": "failed", "reason": "Collection already exists"}
        
        # 确定向量维度
        vector_size = request.vector_size if request.vector_size else service.embed(["sample"]).shape[1]
        
        # 创建新集合
        service.qdrant_client.create_collection(
            collection_name=request.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance[request.distance] if request.distance else Distance.COSINE
            )
        )
        
        # 创建内容哈希索引
        service.qdrant_client.create_payload_index(
            collection_name=request.collection_name,
            field_name="content_hash",
            field_schema="keyword",
            wait=True
        )
        
        return {
            "status": "success",
            "collection_name": request.collection_name,
            "vector_size": vector_size,
            "distance": request.distance or "COSINE"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/stats")
async def stats():
    """API endpoint to get service statistics"""
    return service.get_stats()

if __name__ == "__main__":
    logger.info("正在启动服务...")
    
    # Example usage
    if False:  # Set to True to run test cases
        test_texts = [
            "Qdrant is a vector search engine",
            "BGE is a powerful embedding model",
            "This is a test sentence"
        ]
        
        # Test embedding generation
        logger.info("\n🔍 测试embedding生成...")
        start_inference = time.time()
        embeddings = service.embed(test_texts)
        
        logger.info(f"Embedding维度: {embeddings.shape[1]}")
        logger.info(f"本次推理耗时: {time.time()-start_inference:.3f}秒")
        logger.info(f"示例embedding值: {embeddings[0][:5]}...")
        
        # Test storage
        service.store_embeddings(test_texts)
        
        # Print stats
        stats = service.get_stats()
        logger.info(f"\n📊 统计信息: 平均耗时 {stats['avg_inference']:.3f}秒/次, 总请求数 {stats['total_requests']}")
    
    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)