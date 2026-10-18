import os
import sys
from pathlib import Path

import spaces

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent
DATA = Path(os.getenv("DATA_DIR", str(Path.home() / "data")))
for key, value in {
    "DATA_DIR": str(DATA),
    "APP_VECTORS_DIR": str(DATA / "vectors"),
    "APP_MODELS_DIR": str(DATA / "models"),
    "APP_FRONTEND_DIR": str(ROOT / "frontend" / "dist"),
    "APP_TRUST_PROXY": "1",
    "APP_PRELOAD": "1",
    "ENCODER_DEVICE": "cpu",
}.items():
    os.environ.setdefault(key, value)

import gradio as gr
import uvicorn

from arxiv_rag.api.app import create_app
from scripts.deploy.fetch_data import DEFAULT_REPO, fetch


@spaces.GPU
def unused_gpu_function():
    return None


if os.getenv("SKIP_DATA_DOWNLOAD") != "1":
    manifest = fetch(DATA, os.getenv("HF_DATA_REPO", DEFAULT_REPO), os.getenv("HF_TOKEN") or None)
    print(f"Veri hazır: {len(manifest['files'])} dosya, {manifest.get('papers')} makale", flush=True)

app = create_app()
with gr.Blocks(title="arXiv araştırma asistanı") as demo:
    gr.Markdown("Uygulama ana sayfada: [aç](/)")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, ssr_mode=False, prevent_thread_lock=True)
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
