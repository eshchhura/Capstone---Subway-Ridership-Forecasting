# Model Metrics

## Project

**MTA Subway Hourly Ridership Forecasting**

## Modeling goal

For this part of the project, I treated subway ridership forecasting as a supervised regression problem. The target variable was `ridership_total`, and the goal was to predict hourly ridership using the cleaned, engineered, and preprocessed features from the earlier notebooks.

I wanted the model to do more than just fit the training data well. Since this is a time-based problem, the important question was whether the model could generalize to later periods of ridership data.

## Data used for modeling

| Item | Value |
|---|---:|
| Training feature matrix | (6,034,307, 485) |
| Test feature matrix | (1,508,577, 485) |
| Training target vector | (6,034,307,) |
| Test target vector | (1,508,577,) |
| Number of transformed features | 485 |

The final transformed dataset had 485 features after preprocessing. These features included the numeric variables, encoded categorical variables, and engineered time-based features used by the models.

## Validation approach

I used an expanding-window time-series validation setup instead of a random split. This was important because a random split would mix earlier and later ridership records together, which can make a forecasting model look better than it really is.

With the expanding-window setup, each validation fold came after the training period. That better matches the real forecasting situation: train on the past, then predict a later period.

| Fold | Train rows | Validation rows |
|---:|---:|---:|
| 1 | 0–3,017,153 | 3,017,153–3,620,583 |
| 2 | 0–3,620,583 | 3,620,583–4,224,013 |
| 3 | 0–4,224,013 | 4,224,013–4,827,443 |

## Metrics used

I compared the models with several regression metrics instead of relying on only one number.

| Metric | How I used it |
|---|---|
| MAE | The average absolute prediction error in ridership units. This is the easiest error metric to interpret directly. |
| RMSE | Similar to MAE, but it penalizes larger misses more heavily. I used this to check whether a model was making large errors. |
| R² | The share of variation in ridership explained by the model. Higher values mean the model explains more of the pattern in the target. |
| WMAPE | A percentage-style error weighted by total actual ridership. This helped make the error easier to compare at scale. |

## Model performance comparison

The table below shows the average validation performance across the expanding-window folds. Lower MAE, RMSE, and WMAPE are better. Higher R² is better.

|   Rank | Model                                                             | Model_Family       |   mean_MAE |   std_MAE |   mean_RMSE |   std_RMSE |   mean_R2 |   std_R2 |   mean_WMAPE |   std_WMAPE |
|-------:|:------------------------------------------------------------------|:-------------------|-----------:|----------:|------------:|-----------:|----------:|---------:|-------------:|------------:|
|      1 | LGBM_num_leaves=127_lr=0.05_n_estimators=300_min_child_samples=50 | Gradient boosting  |    44.1373 |   4.71791 |     125.635 |    11.7454 |  0.961472 | 0.006898 |     0.144081 |    0.015545 |
|      2 | XGB_max_depth=8_lr=0.05_n_estimators=300_subsample=0.8            | Gradient boosting  |    45.346  |   4.80807 |     127.68  |    12.452  |  0.960193 | 0.007434 |     0.148025 |    0.015824 |
|      3 | LGBM_num_leaves=63_lr=0.10_n_estimators=200_min_child_samples=50  | Gradient boosting  |    46.196  |   4.60252 |     129.242 |    11.6516 |  0.959238 | 0.007054 |     0.150807 |    0.015286 |
|      4 | LGBM_num_leaves=63_lr=0.05_n_estimators=200_min_child_samples=50  | Gradient boosting  |    48.4993 |   5.05423 |     136.127 |    12.5251 |  0.954774 | 0.007971 |     0.158317 |    0.016635 |
|      5 | XGB_max_depth=6_lr=0.10_n_estimators=200_subsample=0.8            | Gradient boosting  |    48.7715 |   4.88248 |     136.223 |    12.9011 |  0.954712 | 0.008141 |     0.159189 |    0.015869 |
|      6 | XGB_max_depth=6_lr=0.05_n_estimators=200_subsample=0.8            | Gradient boosting  |    51.2663 |   5.11052 |     144.581 |    13.2199 |  0.948988 | 0.008918 |     0.167345 |    0.016796 |
|      7 | Ridge_alpha=100.0                                                 | Linear regularized |    82.5186 |   5.53434 |     198.864 |    12.8993 |  0.903739 | 0.011542 |     0.269378 |    0.018934 |
|      8 | Ridge_alpha=10.0                                                  | Linear regularized |    82.538  |   5.53315 |     198.863 |    12.8975 |  0.90374  | 0.011541 |     0.269441 |    0.018931 |
|      9 | Ridge_alpha=1.0                                                   | Linear regularized |    82.54   |   5.53303 |     198.863 |    12.8973 |  0.90374  | 0.01154  |     0.269448 |    0.018931 |
|     10 | SGDRegressor_alpha=1e-3_penalty=l2                                | Linear regularized |    89.1167 |  10.7519  |     202.401 |    11.3918 |  0.900442 | 0.008708 |     0.29001  |    0.022403 |
|     11 | SGDRegressor_alpha=1e-3_penalty=elasticnet                        | Linear regularized |    89.5264 |  11.2879  |     202.596 |    11.3527 |  0.900258 | 0.008563 |     0.2913   |    0.024088 |
|     12 | SGDRegressor_alpha=1e-4_penalty=l2                                | Linear regularized |    91.9322 |  14.5387  |     203.793 |    11.3485 |  0.89911  | 0.007941 |     0.298881 |    0.034483 |
|     13 | SGDRegressor_alpha=1e-4_penalty=elasticnet                        | Linear regularized |    91.9845 |  14.6223  |     203.824 |    11.3569 |  0.89908  | 0.007935 |     0.299045 |    0.034752 |
|     14 | PoissonRegressor_alpha=0.1_max_iter=300                           | Generalized linear |   116.362  |   6.98829 |     506.927 |    66.852  |  0.37227  | 0.139865 |     0.379475 |    0.01331  |
|     15 | PoissonRegressor_alpha=1.0_max_iter=300                           | Generalized linear |   147.175  |   7.58713 |     764.382 |   108.951  | -0.429935 | 0.356329 |     0.480036 |    0.012935 |
|     16 | PoissonRegressor_alpha=10.0_max_iter=300                          | Generalized linear |   197      |   8.15272 |    1073.86  |   132.998  | -1.8189   | 0.639435 |     0.642696 |    0.013455 |
|     17 | DummyRegressor_mean                                               | Baseline           |   307.669  |   6.00766 |     642.108 |    13.8397 | -0.001308 | 0.001557 |     1.00428  |    0.025799 |

## Final model selected

The final model I selected was:

**`LGBM_num_leaves=127_lr=0.05_n_estimators=300_min_child_samples=50`**

I chose this model because it had the strongest overall validation performance. It had the lowest average MAE, the lowest WMAPE, and one of the strongest RMSE/R² results across the models tested.

The main takeaway from the comparison is that the gradient boosting models clearly performed better than the linear models and the baseline. Ridge and SGDRegressor were useful comparison points, but they did not capture the ridership patterns as well as LightGBM or XGBoost. The dummy baseline was included to show what performance looks like when the model is not learning meaningful feature relationships.

## Final held-out test performance

After selecting the best model from validation, I refit the final LightGBM model on the full training set and evaluated it on the held-out test set.

| Result Type | Model | Model Family | Split | MAE | RMSE | R² | WMAPE |
|:---|:---|:---|:---|---:|---:|---:|---:|
| Held-out test | LGBM_num_leaves=127_lr=0.05_n_estimators=300_min_child_samples=50 | Gradient boosting | test | 36.4419 | 94.5129 | 0.980531 | 0.115476 |

## Results review

On the held-out test set, the final LightGBM model reached:

| Metric | Value |
|---|---:|
| MAE | 36.441853 |
| RMSE | 94.512892 |
| R² | 0.980531 |
| WMAPE | 0.115476 |

The final model performed very well on the held-out test data. Its MAE was about **36.44 riders**, and its WMAPE was about **11.55%**, which means the typical weighted error was relatively small compared with total ridership volume.

The test performance was also stronger than the average validation performance. That does not automatically mean the model is perfect, but it does suggest that the final train/test split was not showing a major generalization problem. The model also improved substantially over the baseline: compared with the validation `DummyRegressor_mean`, the selected LightGBM model reduced average validation MAE by about **85.65%**.

## Notes

- I tested a baseline model, regularized linear models, generalized linear models, LightGBM models, and XGBoost-style models.
- The `PoissonRegressor` models were kept in the comparison, but the notebook showed convergence warnings, so I did not treat them as realistic final-model candidates.
- The final model was selected from the validation results first, then refit on the full training data before the held-out test evaluation.
- Because this project uses time-based ridership data, the expanding-window validation results were especially important for judging whether the model could handle later periods.
