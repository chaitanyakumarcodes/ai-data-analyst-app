# Data Analyst Agent

A Flask-based application that transforms CSV datasets into comprehensive analysis reports through automated machine learning pipelines and AI-powered insights. Upload a dataset, select a target column, and receive model metrics, visualizations, risk assessments, and an interactive chat interface grounded in your analysis results.

## Screenshots

### Upload Interface
<img width="1876" height="863" alt="Screenshot 2026-04-05 104344" src="https://github.com/user-attachments/assets/399d51da-cdcc-4c51-bb9d-97544c82e331" />

*Drag-and-drop CSV upload with real-time validation*

### Target Selection
<img width="1883" height="875" alt="Screenshot 2026-04-05 104536" src="https://github.com/user-attachments/assets/94e1ffa5-747e-4e9c-92fe-738bda2e8649" />

*Interactive column picker with data preview and type detection*

### Interactive Chat
<img width="1877" height="872" alt="Screenshot 2026-04-05 104744" src="https://github.com/user-attachments/assets/dae7bafc-b4c5-4f62-8f2e-400a7a4729e4" />

*AI assistant with tool-calling capabilities grounded in your analysis results*

### Sample PDF Report
<img width="1146" height="614" alt="image" src="https://github.com/user-attachments/assets/cb656177-8e95-4c0b-bb40-397618a860c7" />

*Professional PDF export with metrics, visualizations, and insights*

## Overview

Data Analyst Agent bridges automated ML pipelines with interpretable AI insights. It separates deterministic computation (data cleaning, model training, metrics) from optional LLM-powered interpretation, ensuring reliable results while providing natural language explanations when needed.

**Core Workflow:**
1. Upload a CSV file (up to 16 MB)
2. Select the target column to predict
3. Automated pipeline runs: cleaning, EDA, feature engineering, model training
4. Review dashboard: metrics, feature importance, visualizations, risk flags
5. Query your analysis through an AI chat interface with tool-calling capabilities
6. Export a PDF report with key findings

## Key Features

### Automated ML Pipeline
- **Data Cleaning**: Handles missing values, mixed types, and encoding detection (UTF-8, Latin-1, CP1252)
- **EDA**: Summary statistics, correlation matrices, distribution analysis
- **Feature Engineering**: Automatic preprocessing for numeric and categorical features
- **Model Training**: Random Forest classifier/regressor with balanced class weights
- **Task Detection**: Automatically infers classification vs regression based on target column

### Insights and Risk Detection
- **Deterministic Risk Flags**: Identifies class imbalance, high missingness, potential target leakage
- **LLM-Generated Insights**: Executive highlights and detailed analysis (optional, requires OpenAI API key)
- **Feature Importance**: Top drivers ranked by model importance scores

### Interactive Chat
- **Tool-Backed Responses**: Chat agent calls functions on your session data instead of generating statistics
- **Available Tools**:
  - Summary statistics and column details
  - Row filtering with multiple operators
  - Value counts for categorical analysis
  - Model metrics retrieval
  - Feature importance queries
  - Correlation calculations
  - Single-row predictions
  - Risk flag summaries
  - Plot listings

### Visualizations
- Target distribution (histogram for regression, bar chart for classification)
- Correlation heatmap for numeric features
- Feature importance bar chart
- All plots rendered at 120 DPI for clarity

### Export and Reporting
- **PDF Report**: Single-page landscape A4 with metrics, charts, top features, and highlights
- **Session Persistence**: Results stored in-memory for the session lifecycle
- **Dashboard**: Comprehensive view with KPIs, metrics cards, plots, and insights

## Architecture

### Design Principles
- **Separation of Concerns**: Deterministic pipeline (cleaning, training, plotting) is isolated from LLM components (insights, chat)
- **Session-Scoped Memory**: Each upload creates a unique session with stored artifacts
- **Tool-Calling Pattern**: Chat uses function calling to query actual data rather than hallucinating metrics
- **Graceful Degradation**: Analysis runs without OpenAI API key; insights show placeholder message

### Project Structure
```
data-analyst-agent/
├── app/
│   ├── __init__.py                 # Application factory
│   ├── config.py                   # Configuration settings
│   ├── jinja_filters.py            # Template filters (numeric highlighting)
│   ├── routes/
│   │   ├── main.py                 # Upload, preview, dashboard routes
│   │   └── chat.py                 # Chat API endpoint
│   ├── services/
│   │   ├── analysis_service.py     # Main orchestrator for full pipeline
│   │   ├── data_layer.py           # CSV loading with encoding fallbacks
│   │   ├── dashboard_presenter.py  # Metric formatting and plot ordering
│   │   ├── pdf_report.py           # ReportLab PDF generation
│   │   ├── chat/
│   │   │   ├── orchestrator.py     # OpenAI chat loop with tool calls
│   │   │   └── tools.py            # Tool definitions and implementations
│   │   ├── insights/
│   │   │   ├── generator.py        # LLM insight generation
│   │   │   └── risk.py             # Deterministic risk detection
│   │   ├── memory/
│   │   │   └── analysis_store.py   # Session storage (in-memory registry)
│   │   └── pipeline/
│   │       ├── cleaning.py         # Target handling, X/y separation, task inference
│   │       ├── eda.py              # Summary stats, correlation matrices
│   │       ├── feature_engineering.py  # ColumnTransformer pipeline
│   │       ├── train.py            # RandomForest training and metrics
│   │       └── plots.py            # Matplotlib/Seaborn chart generation
│   ├── static/
│   │   └── css/                    # Page-specific stylesheets
│   └── templates/                  # Jinja2 HTML templates
├── tests/
│   └── test_pipeline_smoke.py      # Basic classification and regression tests
├── uploads/                        # User-uploaded CSVs (gitignored)
├── requirements.txt
├── run.py                          # Development server entry point
└── README.md
```

## Technology Stack

**Backend:**
- Flask 3.0+ (web framework)
- Pandas 2.0+ (data manipulation)
- NumPy 1.24+ (numerical operations)
- Scikit-learn 1.3+ (ML pipeline)
- OpenAI 1.40+ (insights and chat, optional)

**Visualization:**
- Matplotlib 3.7+ (plotting)
- Seaborn 0.13+ (statistical visualizations)
- ReportLab 4.0+ (PDF generation)

**Frontend:**
- Bootstrap 5.3.3 (UI framework)
- Bootstrap Icons 1.11.3
- Marked.js 12.0.2 (Markdown rendering in chat)
- DOMPurify 3.1.6 (HTML sanitization)

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/data-analyst-agent.git
cd data-analyst-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

5. Edit `.env` and add your OpenAI API key (optional):
```
OPENAI_API_KEY=sk-your-api-key-here
FLASK_SECRET_KEY=change-me-in-production
```

**Note:** The application runs without an OpenAI API key. Analysis pipeline, metrics, and plots work normally; insights will show a placeholder message.

## Usage

### Running the Application

Development server:
```bash
python run.py
```

The application will be available at `http://127.0.0.1:5000`

Production deployment (using Gunicorn):
```bash
gunicorn run:app
```

### Workflow

1. **Upload Dataset**
   - Navigate to `/upload`
   - Drag and drop a CSV file or click to browse
   - Maximum file size: 16 MB
   - Maximum rows: 100,000
   - Supported encodings: UTF-8, Latin-1, CP1252

2. **Select Target Column**
   - Preview the first 5 rows of your dataset
   - Choose the column you want to predict
   - The system automatically infers classification vs regression

3. **Review Dashboard**
   - Task type and target column
   - Model performance metrics (accuracy, F1, RMSE, R², etc.)
   - Feature importance rankings
   - Data quality and risk flags
   - Visualizations: target distribution, correlation heatmap, feature importance
   - AI-generated insights (if API key configured)

4. **Interactive Chat**
   - Ask questions about your analysis
   - Query specific columns, filter data, check correlations
   - Request predictions for new data points
   - Get explanations of metrics and drivers

5. **Export Report**
   - Download a PDF with metrics, charts, and highlights
   - Single-page landscape A4 format

### Example Chat Queries

```
What are the top 5 most important features?
Show me rows where age > 30
What's the correlation between income and credit_score?
Summarize model performance in plain language
What data quality risks were flagged?
Predict the outcome for: {"age": 35, "income": 75000}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for insights and chat | None (optional) |
| `FLASK_SECRET_KEY` | Flask session secret | `dev-secret-change-me` |
| `OPENAI_MODEL` | Model for chat | `gpt-4o-mini` |
| `INSIGHTS_MODEL` | Model for insight generation | `gpt-4o-mini` |
| `UPLOAD_FOLDER` | Upload directory path | `uploads` |

### Application Limits

- Maximum file size: 16 MB
- Maximum rows per dataset: 100,000
- Maximum rows returned in chat queries: 100
- Correlation heatmap: Limited to 20 columns for readability

## Testing

Run the test suite:
```bash
python -m pytest tests -v
```

Current test coverage includes:
- Classification pipeline smoke test
- Regression pipeline smoke test
- Data loading and validation
- Feature engineering
- Model training and metrics

## Deployment

### Render.com (Configured)

The repository includes `render.yaml` for one-click deployment to Render.com:

1. Connect your repository to Render
2. Set environment variables in Render dashboard:
   - `OPENAI_API_KEY` (optional)
   - `FLASK_SECRET_KEY` (required in production)
3. Deploy

### General Deployment Notes

- **Static Files**: Flask serves static assets; consider using a CDN for production
- **Session Storage**: Currently in-memory; use Redis or database for multi-instance deployments
- **File Uploads**: Stored on disk; consider object storage (S3, GCS) for scalability
- **Model Persistence**: Models stored in session memory; not persisted to disk
- **Background Jobs**: Long-running analyses may benefit from Celery or similar task queue

## API Reference

### Chat API

**Endpoint:** `POST /api/chat`

**Request:**
```json
{
  "session_id": "uuid-string",
  "message": "What are the top features?"
}
```

**Response:**
```json
{
  "reply": "Based on the analysis, the top 5 features are:\n\n1. **feature_A** — 0.2341\n2. **feature_B** — 0.1876\n..."
}
```

### Available Chat Tools

| Tool | Purpose | Parameters |
|------|---------|------------|
| `get_summary_stats` | Dataset shape, columns, missingness | `column` (optional) |
| `filter_rows` | Filter by condition | `column`, `op`, `value`, `limit` |
| `column_value_counts` | Value frequencies | `column`, `top_n` |
| `get_model_metrics` | Performance metrics | None |
| `get_feature_importance` | Top N features | `top_n` |
| `get_correlation` | Pearson correlation | `column_a`, `column_b` |
| `list_insights` | Precomputed insights | None |
| `get_risk_flags` | Data quality issues | None |
| `predict_row` | Single prediction | `values_json` |
| `list_plots` | Available plot URLs | None |

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Write tests for new functionality
4. Ensure tests pass (`pytest tests -v`)
5. Follow existing code style (Black formatting recommended)
6. Submit a pull request

## License

This project is provided as-is for educational and demonstration purposes. Please review the license file for detailed terms.

## Acknowledgments

- Scikit-learn for robust ML pipelines
- Flask for web framework simplicity
- OpenAI for chat and insight generation capabilities
- Bootstrap for responsive UI components
