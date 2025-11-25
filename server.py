import os
print(f"Setting ORT_PROVIDERS to CPUExecutionProvider")
os.environ["ORT_PROVIDERS"] = "CPUExecutionProvider"
print(f"ORT_PROVIDERS is now set to: {os.environ.get('ORT_PROVIDERS', 'Not set')}")

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from app.config import settings
from app.infer_sign import infer_sign_image, infer_sign_video
from app.infer_lane import infer_lane_image, infer_lane_video
from app.models import warmup_models
import uvicorn
from fastapi.responses import RedirectResponse

app = FastAPI(title="YOLO Web Test (Sign + Lane)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Static
STATIC_DIR = os.path.abspath("./static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Warmup models
print("Warming up models...")
warmup_models()
print("Models warmed up.")

@app.get("/health")
def health():
    return {"ok": True, "device": settings.DEVICE, "base_models_dir": settings.BASE_MODELS_DIR}

# ---------- IMAGE ----------
@app.post("/infer/sign/image")
async def route_sign_image(
    image: UploadFile = File(...),
    variant: str = Query(settings.DEFAULT_VARIANT, pattern="^(best|last)$"),
    conf: float = Query(0.25, ge=0.0, le=1.0)
):
    out_bytes = await infer_sign_image(image=image, variant=variant, conf=conf)
    return StreamingResponse(out_bytes, media_type="image/jpeg")

@app.post("/infer/lane/image")
async def route_lane_image(
    image: UploadFile = File(...),
    variant: str = Query(settings.DEFAULT_VARIANT, pattern="^(best|last)$"),
    conf: float = Query(0.25, ge=0.0, le=1.0)
):
    out_bytes = await infer_lane_image(image=image, variant=variant, conf=conf)
    return StreamingResponse(out_bytes, media_type="image/jpeg")

# ---------- VIDEO ----------
@app.post("/infer/sign/video")
async def route_sign_video(
    video: UploadFile = File(...),
    variant: str = Query(settings.DEFAULT_VARIANT, pattern="^(best|last)$"),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    stride: int = Query(1, ge=1, le=10)  # lấy mỗi stride-frame để nhanh
):
    vid_bytes = await infer_sign_video(video=video, variant=variant, conf=conf, stride=stride)
    return StreamingResponse(vid_bytes, media_type=settings.VIDEO_MIMETYPE)

@app.post("/infer/lane/video")
async def route_lane_video(
    video: UploadFile = File(...),
    variant: str = Query(settings.DEFAULT_VARIANT, pattern="^(best|last)$"),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    stride: int = Query(1, ge=1, le=10)
):
    vid_bytes = await infer_lane_video(video=video, variant=variant, conf=conf, stride=stride)
    return StreamingResponse(vid_bytes, media_type=settings.VIDEO_MIMETYPE)
@app.get("/")
def root():
    # Mở thẳng giao diện test webcam
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    print("Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)