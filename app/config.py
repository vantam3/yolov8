import os

class Settings:
    BASE_MODELS_DIR: str = os.path.abspath(os.getenv("BASE_MODELS_DIR", "./exports_yolo"))
    DEFAULT_VARIANT: str = os.getenv("DEFAULT_VARIANT", "best")  # best|last
    DEVICE: str = os.getenv("DEVICE", "auto")  # auto|cpu|cuda:0

    # Image sizes
    SIGN_IMGSZ: int = 640
    LANE_IMGSZ: int = 960

    # ==== VIDEO ENCODE (xuất WebM cho browser) ====
    # VP8 trong WebM: được Chrome/Firefox/Edge support tốt
    VIDEO_FOURCC: str = os.getenv("VIDEO_FOURCC", "VP80")  # fourcc VP8
    VIDEO_FPS_FALLBACK: float = 25.0
    VIDEO_EXT: str = ".webm"
    VIDEO_MIMETYPE: str = "video/webm"



settings = Settings()
