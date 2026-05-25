from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.predict import router as predict_router

app = FastAPI(
    title="Brain Tumor AI API",
    description="Hệ thống AI phân tích và đánh giá khối u não từ ảnh MRI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "name": "Brain Tumor AI",
        "version": "1.0.0",
        "docs": "/docs",
    }
