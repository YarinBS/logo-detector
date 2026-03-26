# Logo Detector

Logo Detector is a computer-vision project that classifies whether an uploaded image contains one of the target logos.

The current target-logo group is:
- Cocacola
- McDonalds
- Starbucks
- Disney

The project includes:
- A training and evaluation pipeline built with PyTorch
- A FastAPI backend with prediction and health endpoints
- A simple Streamlit frontend that uploads images and displays predictions from the backend

## Setup

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd logo_detector
```

### 2. Install dependencies

Using uv (recommended):

```bash
uv sync
```

If needed, activate the virtual environment created by uv:

```bash
source .venv/bin/activate
```

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

If you do not want editable mode, use:

```bash
pip install .
```

## Usage

### Start the backend (FastAPI)

Run this in terminal 1:

```bash
uv run python src/main.py --mode serve
```

If you used pip/venv setup:

```bash
python src/main.py --mode serve
```

Backend defaults to:
- Host: `0.0.0.0`
- Port: `8000`

Available endpoints:
- `GET /health` returns service status
- `POST /predict` accepts an image file and returns a binary prediction

### Start the frontend (Streamlit)

Run this in terminal 2:

```bash
uv run streamlit run src/streamlit_app.py
```

If you used pip/venv setup:

```bash
streamlit run src/streamlit_app.py
```

Frontend behavior:
- Pings `GET /health` every 5 seconds
- Shows a clear offline message when backend is unavailable
- Uploads image files (`jpg`, `jpeg`, `png`, `bmp`, `webp`)
- Calls `POST /predict` and displays:
	- `Model returned {0|1}`
	- `No logo found` or `Logo found`

### Optional: Train the model

```bash
uv run python src/main.py --mode train
```

If you used pip/venv setup:

```bash
python src/main.py --mode train
```

Training and data paths are configured in `config/config.yaml`.

## Project Structure

```text
logo_detector/
├── README.md
├── pyproject.toml
├── uv.lock
├── config/
│   └── config.yaml
├── data/
│   ├── flickr_logos_27_dataset/
│   │   ├── flickr_logos_27_dataset_images/
│   │   ├── flickr_logos_27_dataset_query_set_annotation.txt
│   │   └── flickr_logos_27_dataset_training_set_annotation.txt
│   └── processed/
├── models/
│   ├── downloaded/
│   │   └── checkpoints/
│   └── trained/
│       └── model.pth
└── src/
    ├── app.py
    ├── config.py
    ├── dataset.py
    ├── evaluator.py
    ├── main.py
    ├── model.py
    ├── streamlit_app.py
    └── trainer.py
```
