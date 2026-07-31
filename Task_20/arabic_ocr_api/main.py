"""
Entrypoint for the Arabic OCR microservice.

Run locally with:
    python main.py

Or directly with uvicorn (recommended for production):
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
"""

import os

import uvicorn

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    reload_flag = os.environ.get("RELOAD", "false").lower() == "true"

    uvicorn.run("app:app", host=host, port=port, reload=reload_flag)
