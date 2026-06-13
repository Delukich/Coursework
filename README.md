# Forecasting Logistics Operation Efficiency with Machine Learning

A coursework project that predicts whether a logistics operation is efficient or risky using machine learning.

The system processes supply chain order data, enriches it with logistics, geographic, weather, and economic features, and trains an XGBoost binary classification model.

## Overview

In logistics, raw order profit does not always reflect the real efficiency of a delivery operation. A delivery may look profitable in the original dataset, but after considering distance, shipping mode, fuel cost, weather risk, tariffs, and country-level logistics indicators, the final result may be different.

This project estimates logistics efficiency through adjusted profit:

```text
Realistic_Profit =
    Order Profit Per Order
    - distance penalty
    - weather penalty
    - tariff penalty
```

The machine learning target is:

```text
Is_Profitable = 1 if Realistic_Profit >= 0 else 0
```

In this project, `Is_Profitable` is used as a practical indicator of logistics operation efficiency.

## Features

* Data preprocessing and feature engineering
* Country name normalization
* Route distance calculation
* Weather feature generation
* Economic and logistics indicators
* XGBoost binary classification model
* Hyperparameter optimization with Optuna
* Model evaluation with metrics and plots
* SHAP-based model interpretation
* Manual console mode for custom delivery scenarios

## Tech Stack

* Python 3.11
* pandas
* numpy
* scikit-learn
* XGBoost
* Optuna
* matplotlib
* seaborn
* SHAP
* requests
* joblib

## Project Structure

```text
.
|-- main.py
|-- requirements.txt
|-- README.md
|-- src/
|   |-- config.py
|   |-- pipeline.py
|   |-- train_model.py
|   |-- geo_service.py
|   |-- weather_service.py
|   `-- economics_service.py
`-- plots/
    |-- metrics_report.txt
    |-- confusion_matrix.png
    |-- roc_curve.png
    |-- precision_recall.png
    |-- feature_importance.png
    |-- threshold_search.png
    |-- calibration_curve.png
    |-- probability_distribution.png
    |-- shap_summary.png
    `-- shap_bar.png
```

## Dataset

The dataset is not included in this repository because it is large and should be stored separately from the source code.

This project uses the **DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS** dataset.

Dataset source: [Mendeley Data](https://data.mendeley.com/datasets/8gx2fvg2k6/5)

Dataset license: CC BY 4.0

Expected file name:

```text
DataCoSupplyChainDataset.csv
```

For a full run, download the dataset, create the `data/` folder if it does not exist, and place the file here:

```text
data/DataCoSupplyChainDataset.csv
```

Generated local files are not committed:

```text
data/
models/
cache/
```

The `plots/` folder is included so that the model results can be reviewed without downloading the dataset or rerunning the full pipeline.

## Installation

Clone the repository:

```bash
git clone https://github.com/Delukich/Coursework.git
cd Coursework
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Check Without the Dataset

If the dataset is not available, the full pipeline cannot be rerun. However, the project can still be checked by reviewing the source code and saved model evaluation results.

Run a syntax check:

```bash
python -m compileall main.py src
```

Then review:

```text
src/
plots/metrics_report.txt
plots/*.png
```

## How to Run With the Dataset

1. Download the dataset from Mendeley Data.
2. Create the `data/` folder if it does not exist:

```bash
mkdir data
```

3. Place the dataset file at:

```text
data/DataCoSupplyChainDataset.csv
```

4. Run the application:

```bash
python main.py
```

5. Use the console menu:

```text
1. Run data pipeline
2. Train model
3. Manual logistics efficiency assessment
0. Exit
```

Recommended order:

```text
1 -> 2 -> 3
```

### Option 1: Run Data Pipeline

Creates the enriched dataset:

```text
data/enriched_dataset.csv
```

This step adds distance, weather, economic, time-based, and derived logistics features.

### Option 2: Train Model

Trains and evaluates the XGBoost model.

Creates:

```text
models/xgboost_profit_classifier.joblib
models/feature_meta.joblib
plots/metrics_report.txt
plots/*.png
```

### Option 3: Manual Assessment

Loads the trained model and allows the user to enter a custom delivery scenario:

* warehouse coordinates
* destination country and city
* shipping mode
* scheduled delivery time
* product category
* quantity
* price
* discount

The program returns the predicted logistics efficiency class and probability.

## Results

Saved metrics from `plots/metrics_report.txt`:

| Metric            |  Value |
| ----------------- | -----: |
| Accuracy          | 0.6755 |
| Balanced Accuracy | 0.6144 |
| ROC AUC           | 0.6730 |
| Average Precision | 0.7621 |
| MCC               | 0.2479 |
| Cohen's Kappa     | 0.2428 |
| Brier Score       | 0.2011 |
| Macro F1          | 0.6180 |
| Optimal Threshold |   0.63 |

Cross-validation results:

| Metric   |      Mean +/- Std |
| -------- | ----------------: |
| ROC AUC  | 0.6573 +/- 0.0025 |
| Accuracy | 0.6952 +/- 0.0016 |
| Macro F1 | 0.5850 +/- 0.0026 |

The results show moderate predictive performance. The model performs better than a naive baseline and demonstrates that logistics-oriented feature engineering provides useful analytical signal.

## Model Interpretation

SHAP plots are generated to explain how features influence model predictions.

The most important feature is `price_per_km`, which represents the relationship between order value and delivery distance. Other important factors include shipping mode, destination country, fuel-distance interaction, and other logistics-related engineered features.

Generated interpretation plots:

```text
plots/shap_summary.png
plots/shap_bar.png
plots/feature_importance.png
```

## Limitations

* The target variable is based on an engineered business rule, not on a real company-defined efficiency label.
* Weather, tariff, fuel, and economic indicators are simplified and may not fully reflect real-world logistics costs.
* The model is an analytical prototype, not a production-ready business decision system.
* Prediction quality depends on the quality and completeness of the input dataset.
* Manual assessment results are approximate and should not be used as guaranteed business recommendations.

## Project Info

This project was developed as a coursework project and demonstrates the use of machine learning, feature engineering, external data enrichment, model evaluation, and SHAP-based interpretation for logistics efficiency forecasting.

* **Author:** Denys Lukianchuk
* **Year:** 2026
