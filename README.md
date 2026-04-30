# MTA Subway Ridership Forecasting

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Model](https://img.shields.io/badge/Final%20Model-LightGBM-green)
![Task](https://img.shields.io/badge/Task-Time%20Series%20Forecasting-purple)

A station-by-station, hour-by-hour forecasting project that predicts MTA subway ridership using fare-tap data from 2022 through 2024.

The goal is simple: given a station and a specific hour, can we predict how many riders will tap in accurately enough to support transit planning decisions?

---

## 1. Project Overview

The MTA moves millions of riders every day, and operational planning depends heavily on knowing when and where demand will appear. This project builds a machine learning model to forecast hourly ridership at the station level using historical subway tap records.

The project focuses on:

- forecasting total ridership per station-hour
- preserving time order to avoid leakage
- comparing linear, count-based, and tree-based models
- validating the final model with time-aware backtesting
- explaining the major ridership patterns found in the data

The final model is a **LightGBM regressor** trained on station, calendar, lag, and rolling ridership features.

---

## 2. Business Problem

Subway ridership is not evenly distributed across time or location. A weekday rush hour at a Manhattan hub behaves very differently from a weekend afternoon at a lower-volume outer-borough station.

This project is framed for an MTA operations or planning team that needs forecasts to support decisions such as:

- train frequency planning
- station staffing
- peak-hour resource allocation
- holiday and weekend service planning
- identifying where demand is likely to spike

The main modeling question:

> Can recent and historical station-level ridership patterns predict the next hour of subway demand?

---

## 3. Data

The base dataset is the **MTA Subway Hourly Ridership 2022–2024** dataset from Kaggle, originally sourced from the MTA open data portal.

Dataset link: [MTA Subway Hourly Ridership 2022 to 2024](https://www.kaggle.com/datasets/yaminh/mta-subway-hourly-ridership-2022-to-2024)

Raw dataset scale:

| Item | Value |
|---|---:|
| Raw records | ~51.2 million |
| Time span | May 2022 to May 2024 |
| Unique station IDs | 431 |
| Raw columns | 12 |
| Final modeling rows | 7,542,884 station-hours |

Each raw row represents a station, hour, fare class, and payment method. Since the prediction target is total station-hour ridership, the raw data was aggregated to one row per station per hour.

---

## 4. Repository Structure

```text
Capstone---Subway-Ridership-Forecasting/
│
├── CSVs/
│   ├── processed_data.csv
│   ├── wide_data.csv
│   ├── X_train_processed.npz
│   ├── X_test_processed.npz
│   ├── y_train.parquet
│   ├── y_test.parquet
│   ├── preprocessor.joblib
│   ├── final_model.joblib
│   ├── final_model_test_predictions.parquet
│   ├── final_model_test_results.csv
│   ├── model_validation_results.csv
│   ├── model_backtest_fold_results.csv
│   ├── model_backtest_summary.csv
│   ├── year_feature_check_fold_results.csv
│   └── year_feature_check_summary.csv
│
├── Notebooks/
│   ├── DataWrangling.ipynb
│   ├── EDA.ipynb
│   ├── Preprocessing.ipynb
│   └── Modeling.ipynb
│
├── Reports/
│   └── Capstone Two - Project Proposal.pdf
|   └── MTA Hourly Ridership Forecasting Final Report.pdf
|   └── MTA_Ridership_Forecasting_Slide_Deck.pptx
│
└── README.md
└── model_metrics.md
```

---

## 5. Project Workflow

### Data Wrangling

The raw file was too large to treat casually, so the first step was memory-conscious loading and cleanup.

Main steps:

- parsed timestamps into datetime format
- renamed columns for readability
- downcasted numeric columns where appropriate
- dropped redundant geospatial string data
- aggregated fare-class/payment-method rows into station-hour rows
- preserved station and borough attributes
- created a wide payment-method table for EDA

The modeling target is:

```text
ridership_total
```

at the station-hour level.

---

### Exploratory Data Analysis

The EDA showed three major patterns.

#### 1. Weekdays have two clear commute peaks

Weekday ridership rises sharply around the morning commute, dips during midday, then peaks again around the evening commute. Weekend ridership is smoother and shifts later in the day.

This made hour-of-day and day-of-week essential model features.

#### 2. Manhattan carries the highest per-station volume

Manhattan stations had the highest average ridership per station-day, followed by Queens, Brooklyn, the Bronx, and Staten Island.

| Borough | Avg ridership per station-day |
|---|---:|
| Manhattan | 13,957 |
| Queens | 5,895 |
| Brooklyn | 4,484 |
| Bronx | 3,301 |
| Staten Island | 2,777 |

#### 3. OMNY usage rose over the dataset window

The data also shows the shift from MetroCard to OMNY. Payment method was useful for understanding system behavior, although it was not part of the final forecasting feature set.

---

## 6. Feature Engineering

The final model uses a combination of station identifiers, calendar variables, lag features, and rolling statistics.

| Feature group | Features |
|---|---|
| Station/location | `station_id`, `borough` |
| Calendar | `hour`, `day_of_week`, `day`, `month`, `year`, `is_weekend`, `is_holiday` |
| Lag features | `lag_1h`, `lag_24h`, `lag_168h` |
| Rolling features | `rolling_mean_24h`, `rolling_std_24h` |

Lag and rolling features were shifted so the model could not see future ridership values.

Categorical features were one-hot encoded using `OneHotEncoder(handle_unknown="ignore")`. The processed design matrix was kept sparse to avoid memory blowups from the large station-hour dataset.

---

## 7. Train/Test Strategy

This is a time-series forecasting problem, so I avoided random train/test splitting.

Instead, I used:

- chronological 80/20 train/test split
- last 20% of the timeline as the final holdout test set
- three expanding-window backtest folds inside the training period

This setup prevents future rows from leaking into the training data and gives a more realistic view of how the model would perform when deployed forward in time.

---

## 8. Models Tested

I compared a simple baseline, linear models, a count model, and tree-based models.

Models tested:

- Dummy mean baseline
- Ridge Regression
- SGD Regressor
- Poisson Regressor
- XGBoost Regressor
- LightGBM Regressor

Tree-based models performed best because subway ridership has strong nonlinear patterns by hour, day, station, and recent demand.

### Backtest Model Comparison

| Model | MAE | RMSE | R² | WMAPE |
|---|---:|---:|---:|---:|
| LightGBM, 127 leaves, lr 0.05, 300 trees | 44.14 | 125.64 | 0.9615 | 0.1441 |
| XGBoost, depth 8, lr 0.05, 300 trees | 45.35 | 127.68 | 0.9602 | 0.1480 |
| LightGBM, 63 leaves, lr 0.10, 200 trees | 46.20 | 129.24 | 0.9592 | 0.1508 |
| XGBoost, depth 6, lr 0.10, 200 trees | 48.77 | 136.22 | 0.9547 | 0.1592 |
| Ridge, alpha 100 | 82.52 | 198.86 | 0.9037 | 0.2694 |
| SGD, L2 | 89.12 | 202.40 | 0.9004 | 0.2900 |
| Poisson, alpha 0.1 | 116.36 | 506.93 | 0.3723 | 0.3795 |
| Dummy mean baseline | 307.67 | 642.11 | -0.0013 | 1.0043 |

The strongest backtest model was the LightGBM configuration with 127 leaves, a 0.05 learning rate, and 300 trees.

---

## 9. Final Model Performance

The final model was evaluated once on the untouched chronological holdout test set.

| Metric | Final test score |
|---|---:|
| MAE | 36.4 riders per station-hour |
| RMSE | 94.5 |
| R² | 0.981 |
| WMAPE | 11.5% |

The final model explains about 98% of the variance in the held-out test set. The average absolute error is about 36 riders per station-hour, which is small relative to the traffic at high-volume stations.

The largest misses happen during rare high-volume station-hours, where ridership spikes above normal patterns. The model is generally accurate across normal demand ranges but can underpredict some extreme peaks.

---

## 10. Key Takeaways

- The subway system has strong, repeatable time patterns.
- Hour-of-day, day-of-week, and station identity are critical.
- Lag features carry a large amount of predictive signal.
- Tree-based models substantially outperform linear models.
- Time-aware validation is necessary; random splitting would overstate performance.
- LightGBM provided the best balance of accuracy and scalability on this dataset.

---

## 11. How to Reproduce

Clone the repository:

```bash
git clone https://github.com/eshchhura/Capstone---Subway-Ridership-Forecasting.git
cd Capstone---Subway-Ridership-Forecasting
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

Install the main Python packages:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn holidays lightgbm xgboost joblib pyarrow jupyter
```

Download the Kaggle dataset and place the raw CSV in the expected local data location used by the notebooks, or update the file paths in the notebooks.

Recommended notebook order:

1. `Notebooks/DataWrangling.ipynb`
2. `Notebooks/EDA.ipynb`
3. `Notebooks/Preprocessing.ipynb`
4. `Notebooks/Modeling.ipynb`

---

## 12. Limitations and Future Improvements

This model performs well using historical tap patterns, but future versions could improve by adding external context that is not present in the base dataset.

Potential improvements:

- service disruption and delay data
- weather data
- major event calendars
- school calendar effects
- station-level line/service metadata
- separate peak-hour error analysis
- probabilistic prediction intervals instead of point forecasts only
- a lightweight dashboard for station-level forecast exploration

---

## Author

**Esh S. Chhura**  
GitHub: [@eshchhura](https://github.com/eshchhura)
