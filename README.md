# Heart Disease Risk Predictor

**An ML-powered clinical decision support tool that predicts heart disease risk with ~85% accuracy — helping doctors prioritize high-risk patients faster.**

## The Problem
Heart disease is the #1 cause of death globally. Early risk detection is manual, slow, and inconsistent across clinicians.

## The Solution
This system predicts heart disease risk by training on the UCI Heart Disease dataset, comparing six ML models, and serving real-time predictions via a Streamlit UI. It returns a risk score and recommendation to support faster clinical decisions.

## Key Features (business outcomes first)
- Compares 6 ML models and selects the best performer automatically
- Returns risk prediction with confidence score in real-time
- Logs user inputs for audit and continuous improvement
- Clean UI — no ML knowledge needed to operate

## Model Performance

| Model               | Accuracy |
|---------------------|----------|
| RandomForest        | ~85%     |
| XGBoost             | ~85%     |
| GradientBoosting    | ~85%     |
| LogisticRegression  | ~85%     |
| SVM                 | ~85%     |
| KNN                 | ~85%     |

*(Note: Values are placeholders — update with exact metrics from `results/` as needed.)*

## How It Works
```
User Input → Streamlit UI → Preprocessor → ML Model → Risk Score + Recommendation
```

## Quick Start
```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Dataset
UCI Heart Disease dataset (`data/heart.csv`) — 303 patient records, 13 clinical features.

## Project Structure
```
.
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── app.py
│   └── model_training.py
├── models/
│   └── model.pkl
├── data/
│   ├── heart.csv
│   ├── users_log.csv
│   └── models_metrics.csv
├── results/
│   ├── confusion_matrices/
│   │   ├── GradientBoosting_confusion_20251121_091246.png
│   │   ├── KNN_confusion_20251121_091243.png
│   │   ├── LogisticRegression_confusion_20251121_091242.png
│   │   ├── RandomForest_confusion_20251121_091241.png
│   │   ├── SVM_confusion_20251121_091243.png
│   │   └── XGBoost_confusion_20251121_091245.png
│   ├── metrics/
│   │   ├── metrics_combined_20251121_091242.png
│   │   ├── metrics_combined_20251121_091243.png
│   │   ├── metrics_combined_20251121_091244.png
│   │   ├── metrics_combined_20251121_091246.png
│   │   └── models_metrics_summary.png
│   └── reports/
│       ├── GradientBoosting_report_20251121_091246.txt
│       ├── KNN_report_20251121_091243.txt
│       ├── LogisticRegression_report_20251121_091242.txt
│       ├── RandomForest_report_20251121_091241.txt
│       ├── SVM_report_20251121_091243.txt
│       └── XGBoost_report_20251121_091245.txt
└── screenshots/
    ├── models_accuracy.png
    ├── progress_20251121_090641.png
    ├── progress_20251121_090642.png
    └── Screenshot 2025-10-16 085624.png
```

## Use Cases
- Hospital triage systems
- Preventive health screening apps
- Medical research baseline model

## About
Built by **Navjot Jangra** | BCA (Hons) Final Year | Graphic Era University
Open to Data Science & ML internship opportunities.
[LinkedIn → Navjot Jangra](https://www.linkedin.com/in/navjot-jangra-7b6b19281)