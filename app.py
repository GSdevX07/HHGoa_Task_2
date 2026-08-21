"""
Hugging Face Spaces Entrypoint — Serves Original HTML/CSS/JS Frontend
====================================================================
Serves the full Neo-Brutalist Voice RAG Console at '/' while satisfying ZeroGPU.
"""

import os
import sys
import uvicorn
from fastapi.staticfiles import StaticFiles

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ZeroGPU handler registration
try:
    import spaces
    @spaces.GPU(duration=60)
    def gpu_task_handler():
        return "ZeroGPU Active"
except Exception:
    spaces = None
    def gpu_task_handler():
        return "CPU Mode"

import gradio as gr
from backend.main import app as fastapi_app

# Minimal Gradio block to bind @spaces.GPU to route table for ZeroGPU supervisor
with gr.Blocks(title="Voice RAG Backend") as demo:
    gr.Markdown("### Voice RAG Engine")
    check_btn = gr.Button("Init GPU")
    out_lbl = gr.Label()
    check_btn.click(fn=gpu_task_handler, outputs=out_lbl)

# Mount Gradio on /gradio so root '/' remains dedicated to the original frontend
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

# Mount original custom frontend to root '/'
_FRONTEND_DIR = os.path.join(_REPO_ROOT, "frontend")
if os.path.exists(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
