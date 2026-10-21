from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    bundle_dir: Path = Path('models/selected-sibi-bisindo-v1')
    allowed_origins: tuple[str, ...] = ('http://localhost:5173', 'http://localhost:8000')
    max_frame_bytes: int = 2_000_000
    max_frame_pixels: int = 2_073_600
    queue_timeout_seconds: float = .1
    torch_threads: int = 2
    websocket_idle_seconds: float = 60.

    @classmethod
    def from_env(cls):
        origins = tuple(x.strip() for x in os.getenv('ALLOWED_ORIGINS','http://localhost:5173,http://localhost:8000').split(',') if x.strip())
        if not origins or any('*' in o or urlsplit(o).scheme not in {'http','https'} or not urlsplit(o).netloc
                              or urlsplit(o).path or urlsplit(o).query or urlsplit(o).fragment or urlsplit(o).username for o in origins):
            raise ValueError('ALLOWED_ORIGINS must contain explicit http(s) origins without a path or wildcard')
        result = cls(bundle_dir=Path(os.getenv('MODEL_BUNDLE_DIR',str(cls.bundle_dir))), allowed_origins=origins,
            max_frame_bytes=int(os.getenv('MAX_FRAME_BYTES','2000000')),
            max_frame_pixels=int(os.getenv('MAX_FRAME_PIXELS','2073600')),
            queue_timeout_seconds=float(os.getenv('QUEUE_TIMEOUT_SECONDS','0.1')),
            torch_threads=int(os.getenv('TORCH_THREADS','2')),
            websocket_idle_seconds=float(os.getenv('WEBSOCKET_IDLE_SECONDS','60')))
        if min(result.max_frame_bytes,result.max_frame_pixels,result.torch_threads,result.websocket_idle_seconds) <= 0 or result.queue_timeout_seconds <= 0:
            raise ValueError('Resource settings must be positive')
        return result
