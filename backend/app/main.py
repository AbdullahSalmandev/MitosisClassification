from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .model_runtime import ModelRuntime, RuntimeConfig
from .schemas import HealthResponse, PredictionResponse

app = FastAPI(title="Recovered Model API", version="1.0.0")
runtime = ModelRuntime(RuntimeConfig())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    runtime.load()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", payload_type=runtime.payload_type)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), top_k: int = 5) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    data = await file.read()
    preds = runtime.predict(data, top_k=top_k)
    return PredictionResponse(predictions=preds)
