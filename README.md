# Data Analyst Agent

Flask app: upload a CSV, run a deterministic sklearn pipeline (cleaning, EDA, Random Forest, plots), optional OpenAI insights, and a tool-calling chat grounded in stored artifacts.

## Setup

```bash
cd data-analyst-agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `OPENAI_API_KEY` in `.env` for insights and chat. The analysis pipeline works without it (insights show a placeholder).

## Run

```bash
python run.py
```

Open http://127.0.0.1:5000 — upload CSV, pick target column, review dashboard, use chat.

## Tests

```bash
python -m pytest tests -v
```
