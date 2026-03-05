# CSV Analyst Pro — Flask

## Project Structure

```
project/
├── app.py                    # Flask app — all routes + API endpoints
├── requirements.txt
├── .env                      # GEMINI_API_KEY=your_key_here
│
├── modules/
│   ├── data_cleaner.py       # Missing value imputation + structural repair
│   ├── eda_report.py         # ydata-profiling HTML generation
│   ├── automl_trainer.py     # FLAML AutoML training + model export
│   └── gemini_pipeline.py    # Gemini AI query pipeline
│
└── templates/
    ├── upload.html           # Landing page / file upload (/)
    └── workspace.html        # Main 5-tab workspace (/workspace)
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env
echo "GEMINI_API_KEY=your_key_here" > .env
echo "FLASK_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex())')" >> .env

# 3. Run
python app.py
# → http://localhost:5000
```

## API Reference

| Method | Endpoint               | Description                        |
|--------|------------------------|------------------------------------|
| GET    | /                      | Upload landing page                |
| GET    | /workspace             | Main workspace (requires upload)   |
| POST   | /api/upload            | Upload CSV → returns profile JSON  |
| GET    | /api/profile           | Dataset profile (raw + cleaned)    |
| GET    | /api/preview           | First N rows as JSON               |
| POST   | /api/query             | AI query via Gemini                |
| POST   | /api/clean             | Run cleaning pipeline              |
| GET    | /api/clean/download    | Download cleaned CSV               |
| POST   | /api/eda               | Generate ydata-profiling report    |
| GET    | /api/eda/report        | Serve EDA HTML (for iframe)        |
| POST   | /api/automl/detect-task| Detect classification/regression   |
| POST   | /api/automl/train      | Train FLAML AutoML                 |
| GET    | /api/automl/download   | Download best model (.pkl)         |

## Data Flow

Upload page → POST /api/upload → session stores df_raw on disk
     ↓
/workspace renders with profile injected via Jinja2
     ↓
Tabs make fetch() calls to API endpoints
     ↓
Cleaning: POST /api/clean → stores df_clean → used by Query/EDA/AutoML
EDA:      POST /api/eda   → stores eda_html → served via iframe
AutoML:   POST /api/automl/train → stores model_pkl → downloadable
