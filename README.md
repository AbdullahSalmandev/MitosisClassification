# Recovered PyTorch Checkpoint -> Deployable App

This project reconstructs a model from an extracted PyTorch checkpoint folder (`data.pkl`, `data/`, `version`, `byteorder`) and deploys it with a FastAPI backend and React + Tailwind frontend.

## 1) Inspect and recover the checkpoint

Run:

```bash
python3 scripts/inspect_extracted_checkpoint.py \
  --extracted-dir inception_balanced_trained_weights_1 \
  --repacked-file artifacts/recovered_checkpoint.pth
```

This does three things:

1. Rebuilds `artifacts/recovered_checkpoint.pth` from the extracted files.
2. Loads the payload with `torch.load(...)`.
3. Exports:
   - `artifacts/checkpoint_report.json`
   - `artifacts/tensors.json`

Then infer architecture hints:

```bash
python3 scripts/infer_architecture.py
```

## 2) Understanding what you have

`inspect_extracted_checkpoint.py` classifies payload into:

- `full_model` (`torch.nn.Module` stored directly)
- `state_dict` (plain parameter dictionary)
- `checkpoint_dict_with_state_dict`
- `checkpoint_dict_with_model_state_dict`
- generic dictionary/object

If architecture class code is missing, the runtime uses `state_dict` keys + tensor dimensions to reconstruct with a likely backbone (`inception_v3` by default).

## 3) Backend (FastAPI)

Install and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

Endpoints:

- `GET /health`
- `POST /predict` (multipart image file, optional `top_k`)

## 4) Frontend (React + Tailwind)

```bash
cd frontend
npm install
npm run dev
```

Optional environment:

```bash
export VITE_API_URL=http://localhost:8000
```

## 5) One-off inference test

```bash
python3 run_inference.py --image /path/to/image.jpg --top-k 5
```

## Notes for architecture reconstruction

- Check `artifacts/tensors.json` for last classifier layer shape (often gives class count).
- Check `artifacts/checkpoint_report.json` for naming signals (`mixed`, `inception`, `aux`, `fc`).
- If strict loading fails, adapt `default_arch` in `backend/app/model_runtime.py`.
