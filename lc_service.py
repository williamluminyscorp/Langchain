# lc_service.py
from fastapi import FastAPI
import uvicorn
from lc import EmbeddingService  # 导入之前写的类
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("正在启动服务...")  # 添加这行
app = FastAPI()
service = EmbeddingService()
service.load()

@app.post("/embed")
async def embed(texts: list[str]):
    return {"embeddings": service.embed(texts).tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)