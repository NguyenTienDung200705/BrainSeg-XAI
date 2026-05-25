from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.inference import predict
from app.utils.image_utils import load_image_from_bytes

router = APIRouter()


@router.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh.")

    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File ảnh quá lớn (tối đa 20MB).")

    try:
        img_rgb = load_image_from_bytes(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        result = predict(img_rgb)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý mô hình: {str(e)}")

    return result


@router.get("/health")
async def health():
    return {"status": "ok", "message": "Brain Tumor AI API đang hoạt động"}
