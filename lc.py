from sentence_transformers import SentenceTransformer
import time
import numpy as np

class EmbeddingService:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = None
        self.model_name = model_name
        self.load_time = 0
        self.inference_times = []
    
    def load(self):
        """加载并预热模型"""
        if self.model is None:
            start = time.time()
            print(f"🔄 正在加载 {self.model_name} 模型...")
            
            self.model = SentenceTransformer(
                self.model_name,
                device="cpu",  # 使用CPU
                # quantize=True  # 如需量化取消注释(需安装onnxruntime)
            )
            
            # 预热模型
            self.model.encode(["warmup text"], batch_size=1)
            self.load_time = time.time() - start
            print(f"✅ 模型加载完成，耗时: {self.load_time:.1f}秒")
    
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

# ============ 使用示例 ============ 
if __name__ == "__main__":
    # 初始化服务
    service = EmbeddingService()
    service.load()  # 首次加载
    
    # 测试文本
    texts = [
        "Qdrant is a vector search engine",
        "BGE is a powerful embedding model",
        "This is a test sentence"
    ]
    
    # 生成embedding
    print("\n🔍 测试embedding生成...")
    start_inference = time.time()
    embeddings = service.embed(texts, normalize=True)
    
    # 打印结果
    print(f"Embedding维度: {embeddings.shape[1]}")  # 输出 768
    print(f"本次推理耗时: {time.time()-start_inference:.3f}秒")
    print(f"示例embedding值: {embeddings[0][:5]}...")  # 前5个值
    
    # 性能统计
    stats = service.get_stats()
    print(f"\n📊 统计信息: 平均耗时 {stats['avg_inference']:.3f}秒/次, 总请求数 {stats['total_requests']}")

