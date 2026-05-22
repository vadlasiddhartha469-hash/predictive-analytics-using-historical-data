# Predictive Analytics Using Historical Data

A streamlined machine learning implementation that cleans historical datasets, builds a predictive time-series model (`SARIMAX`) to forecast future trends, and evaluates model accuracy.

## Key Features
* **Data Preprocessing:** Automated cleaning and handling of missing historical data points.
* **Time-Series Modeling:** Implements a state-space `SARIMAX` model optimized for seasonal trends.
* **Evaluation & Visualization:** Calculates standard error metrics (`MAE`, `RMSE`) and generates visual comparison plots of actual vs. predicted trends.

## Setup & Installation

Ensure you have the required dependencies installed:

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels
