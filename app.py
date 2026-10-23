"""Root Hugging Face Spaces & Local Gradio Entrypoint for MicroLLM Decision Studio."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure `src/` is on sys.path when launched directly on Hugging Face Spaces without `pip install -e .`
_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from microllm.gradio_app import create_gradio_demo, zero_gpu_hook  # noqa: E402

demo = create_gradio_demo()

if __name__ == "__main__":
    _ = zero_gpu_hook()
    port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", "7860")))
    demo.launch(server_name="0.0.0.0", server_port=port)  # nosec B104
