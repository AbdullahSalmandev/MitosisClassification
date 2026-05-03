# Project Documentation

## Overview

This project reconstructs a PyTorch model from an extracted checkpoint folder and exposes it as a deployable application with:

- a FastAPI backend for model loading and inference
- a React + Tailwind frontend for image upload and result display
- a local CLI utility for one-off inference

The key recovery target is the extracted checkpoint directory `inception_balanced_trained_weights_1`, which contains files such as `data.pkl`, `data/`, `version`, and `byteorder`.

---

## Backend Architecture

### Entry point

The backend is implemented under `backend/app/main.py`.

Key behavior:

- Initializes a `FastAPI` application named `Recovered Model API`
- Adds `CORSMiddleware` to allow requests from any origin
- Creates a global `ModelRuntime` instance with `RuntimeConfig`
- Loads the model on application startup using `runtime.load()`
- Exposes two endpoints:
  - `GET /health` - returns runtime status and detected payload type
  - `POST /predict` - accepts an image upload and returns predictions

### API contracts

The response models are defined in `backend/app/schemas.py` using Pydantic:

- `HealthResponse`: status and optional `payload_type`
- `PredictionItem`: single prediction item with `index`, `label`, and `score`
- `PredictionResponse`: list of `PredictionItem`s

### Request validation

The `POST /predict` endpoint validates that the uploaded file is an image via the `content_type` header and raises a 400 error otherwise.

---

## Model Runtime and Checkpoint Recovery

The core model runtime logic is in `backend/app/model_runtime.py`.

### Runtime configuration

`RuntimeConfig` provides default values for:

- `extracted_checkpoint_dir`: `inception_balanced_trained_weights_1`
- `repacked_checkpoint_file`: `artifacts/recovered_checkpoint.pth`
- `labels_file`: `artifacts/labels.json`
- `default_arch`: `inception_v3`
- `image_size`: `299`
- `num_classes`: `2`
- `class_names`: `("mitotic", "non_mitotic")`

### Loading flow

`ModelRuntime.load()` does the following:

1. Calls `repack_extracted_checkpoint(...)` from `scripts/checkpoint_tools.py` to rebuild a usable checkpoint file from the extracted directory.
2. Loads the repacked file with `torch.load(...)`.
3. Detects the payload type using `detect_payload_type(...)`.
4. Extracts a `state_dict` from the payload with `extract_state_dict(...)`, unless the payload is already a `torch.nn.Module`.
5. Builds a model architecture using `_build_model_from_state_dict(...)`.
6. Summarizes architecture hints with `summarize_architecture_signals(...)`.
7. Loads class labels from `artifacts/labels.json` if present, otherwise falls back to `config.class_names`.

### Model reconstruction

The runtime supports reconstruction of an `inception_v3` model by default. It compares parameter keys and tensor shapes from the loaded checkpoint against the architecture's state dictionary.

- Compatible keys are copied into the model
- Incompatible keys are dropped and logged
- `missing` and `unexpected` keys are printed for diagnosis

This makes the runtime robust against partially compatible recovered checkpoints.

### Inference pipeline

`ModelRuntime.predict(image_bytes, top_k)` performs:

1. Reads incoming image bytes and converts them to RGB
2. Applies preprocessing:
   - resize to `299x299`
   - convert to tensor
   - normalize with ImageNet-style mean/std
3. Runs the model in inference mode on CPU
4. Extracts logits, handles tuple outputs (e.g. Inception auxiliary logits)
5. Computes softmax probabilities
6. Returns the top-K predictions with label, score, and class index

---

## Model and Recovery Tools

The project also includes scripts for inspecting and recovering the checkpoint:

- `scripts/inspect_extracted_checkpoint.py`
- `scripts/infer_architecture.py`
- `scripts/checkpoint_tools.py`

### What they do

- `inspect_extracted_checkpoint.py` rebuilds `artifacts/recovered_checkpoint.pth`, loads the payload, and exports metadata such as `artifacts/checkpoint_report.json` and `artifacts/tensors.json`.
- `infer_architecture.py` examines the recovered checkpoint to infer likely architecture signals.

These tools are used to determine if the payload is:

- a full `torch.nn.Module`
- a plain `state_dict`
- a checkpoint dict containing a state dict or model state dict

The runtime uses this information to choose the correct recovery path.

---

## One-off Inference CLI

The repository includes `run_inference.py`, which lets you run a single prediction from the command line.

Usage:

```bash
python3 run_inference.py --image /path/to/image.jpg --top-k 5
```

This script:

- creates a `ModelRuntime`
- loads the model
- runs prediction on the specified image
- prints top-K results with label, score, and index

---

## Frontend Overview

The frontend is a React application located in `frontend/`.

### Core UI flow

`frontend/src/App.tsx` implements:

- file chooser for image upload
- top-K selection input
- submission handler that calls the backend
- preview of the selected image
- result panel showing prediction labels and confidence percentages

### Backend integration

`frontend/src/api.ts` defines `predictImage(...)`:

- sends a `POST` request to `http://localhost:8000/predict` by default
- supports `VITE_API_URL` environment override
- sends the image as multipart form data with `top_k`
- parses the JSON response and returns prediction items

### UI libraries

The frontend uses:

- `React 18`
- `Vite` for development and build
- `Tailwind CSS` for styling
- `framer-motion` for animation
- `lucide-react` for icons

### Package configuration

Dependencies are declared in `frontend/package.json`.

---

## Running the Project

### Backend

```bash
cd /home/abdullah/Downloads/AI/aiproj
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

To point the frontend at the backend use:

```bash
export VITE_API_URL=http://localhost:8000
```

### Notes

- The backend is CPU-only by default and uses `torch.device("cpu")`.
- `artifacts/recovered_checkpoint.pth` is rebuilt on every backend startup by `ModelRuntime.load()`.
- If your recovered checkpoint is not compatible with `inception_v3`, update `RuntimeConfig.default_arch` and the model reconstruction logic in `backend/app/model_runtime.py`.

---

## Important Files

- `backend/app/main.py` - FastAPI application and endpoints
- `backend/app/model_runtime.py` - checkpoint recovery, model loading, preprocessing, inference
- `backend/app/schemas.py` - response schemas
- `frontend/src/App.tsx` - UI form and result rendering
- `frontend/src/api.ts` - frontend API client
- `run_inference.py` - CLI inference runner
- `scripts/inspect_extracted_checkpoint.py` - checkpoint recovery inspection
- `scripts/infer_architecture.py` - architecture inference helper
- `artifacts/labels.json` - optional class labels used by the runtime
- `inception_balanced_trained_weights_1/` - extracted recovered checkpoint data
