# Arabic OCR Microservice

FastAPI service wrapping the CRNN + CTC Arabic OCR model trained in the
companion notebook.

## Project structure

```
ocr-service/
├── main.py                  # entrypoint (runs uvicorn)
├── app.py                   # FastAPI app: GET / and POST /predict
├── requirements.txt
├── ocr/
│   ├── __init__.py
│   ├── config.py            # loads config.json / paths
│   ├── model_loader.py      # loads model + vocab once, at startup
│   ├── preprocessing.py     # mirrors the notebook's preprocess_image()
│   └── decoding.py          # beam-search CTC decoding
└── deployment_artifacts/    # put crnn_ocr_model.keras, vocab.json, config.json here
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy the `deployment_artifacts/` folder produced by the training
notebook's "Save the Best Model" cell into this project, so it contains:

```
deployment_artifacts/
├── crnn_ocr_model.keras
├── vocab.json
└── config.json
```

## Run

```bash
python main.py
# or, for production:
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

The model loads once at startup (see the `lifespan` handler in `app.py`) --
not per request.

> Note on `--workers`: each uvicorn worker is a separate process and will
> load its own copy of the model at startup, so scaling to N workers means
> N copies of the model in memory. Size your instance accordingly, or keep
> `--workers 1` behind a load balancer with multiple instances instead.

## Endpoints

### `GET /`
Health check.

```json
{
  "status": "ok",
  "model_loaded": true,
  "vocab_size": 65,
  "ctc_time_steps": 256
}
```

### `POST /predict`
Upload an image file (`multipart/form-data`, field name `file`).

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample_address.jpg"
```

Response:

```json
{
  "success": true,
  "filename": "sample_address.jpg",
  "recognized_text": "القاهرة - مدينة نصر - شارع..."
}
```

Interactive API docs are available at `http://localhost:8000/docs` once the
server is running.
