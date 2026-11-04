# AI-Powered Broadband Network Intelligence Platform

An AI/ML portfolio project for broadband network monitoring, anomaly detection, network issue classification, and prediction history.

This project uses synthetic broadband telemetry data to demonstrate an end-to-end machine learning application. It is an independent portfolio project and is not affiliated

## Features

- Broadband telemetry data analysis
- Machine learning-based network anomaly detection
- Network issue classification
- FastAPI prediction backend
- Streamlit monitoring dashboard
- SQLite prediction history
- CSV export
- Model evaluation reports and confusion matrix
- Network health analytics

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- FastAPI and Uvicorn
- Streamlit
- SQLite
- pytest

## Project Structure

```text

Broadband-Network-Intelligence/

├── data/

├── docs/

├── models/

├── reports/

├── src/

│   ├── api/

│   └── dashboard/

├── tests/

├── .gitignore

└── README.md
```

## Run Locally

Use PowerShell from the project root.

### 1. Activate the virtual environment

```powershell

.\\.venv\\Scripts\\Activate.ps1
```

### 2. Start the API

Open the first PowerShell terminal and run:

```powershell

python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

API health check: http://127.0.0.1:8000/health

Interactive API documentation: http://127.0.0.1:8000/docs

### 3. Start the dashboard

Keep the API running. Open a second PowerShell terminal, navigate to the project root, activate the virtual environment, and run:

```powershell

python -m streamlit run .\\src\\dashboard\\app.py --server.port 8511
```

Dashboard: http://localhost:8511

## Model Evaluation

The candidate anomaly detection model was evaluated using an 80/20 stratified train/test split on synthetic telemetry data.

| Metric | Score |

|---|---:|

| Accuracy | 96.35% |

| Precision | 93.24% |

| Recall | 96.57% |

| F1-score | 94.88% |

| ROC-AUC | 99.73% |

### Confusion Matrix

| Actual class | Predicted Normal | Predicted Anomaly |

|---|---:|---:|

| Normal | 1251 | 49 |

| Anomaly | 24 | 676 |

These results describe performance on the generated dataset and should not be interpreted as real-world broadband network performance.

## Limitations

- The included telemetry is synthetic data, not live network data.
- Performance on real network traffic may differ.
- Further validation on independent, representative datasets is needed.

## Future Improvements

- Validate against real or publicly available network telemetry.
- Add model monitoring and drift detection.
- Improve API and dashboard deployment.
- Add CI tests and containerized deployment.

## Disclaimer

This is an independent educational portfolio project developed for learning and demonstration purposes.
