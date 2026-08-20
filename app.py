import os
import sys
import uvicorn

# ZeroGPU compatibility for Hugging Face Spaces
try:
    import spaces
    @spaces.GPU
    def _zerogpu_init():
        return True
    _zerogpu_init()
except Exception:
    pass

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.main import app

# Expose app for ASGI servers (Uvicorn, Gunicorn, Hugging Face)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

