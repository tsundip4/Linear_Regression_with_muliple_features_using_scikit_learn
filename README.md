# California Housing Price Prediction (Linear Regression with SGD)

This Colab notebook trains a linear regression model using `SGDRegressor` on
the California Housing dataset.  
The goal is to understand feature influence on house prices after Z-score normalization.

## Steps
1. Load and inspect the dataset (`fetch_california_housing`)
2. Normalize features with `StandardScaler`
3. Fit an `SGDRegressor` model
4. Visualize predictions vs. targets across 8 features

- **Dependencies:** `scikit-learn`, `numpy`, `matplotlib`

## Results
The model outputs predictions for all 20,640 samples.
The visualization shows how predictions align with actual values across each feature.
Features like MedInc and HouseAge show clearer trends compared to others like Latitude or Longitude.
