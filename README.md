# E-Commerce OLIST Analytics and ML Dashboard

This project is an end-to-end analytics solution built on the Brazilian OLIST e-commerce dataset.

It combines:

1. data loading and preparation
2. exploratory data analysis
3. feature engineering
4. machine learning with PyCaret
5. Power BI semantic modeling and dashboard design
6. project documentation and storytelling assets

## Project Goal

The goal of the project is to turn raw e-commerce data into a business-ready analytics product that helps answer questions such as:

1. how sales and orders are performing
2. which customer segments are more valuable
3. where delivery and review issues exist
4. how machine learning can support scoring, anomaly detection, and forecasting

## Project Structure

1. `notebooks/`
   1. notebook workflow for data loading, EDA, feature engineering, and PyCaret modeling
2. `pipelines/`
   1. reusable Python pipeline code
3. `dashboards/`
   1. Power BI assets, themes, backgrounds, and semantic model files
4. `reports/`
   1. exported metrics, predictions, quality reports, and project documentation
5. `PROJECT_CONTEXT.md`
   1. summary of project state and key implementation details
6. `SESSION_CHANGELOG.md`
   1. working notes and session updates

## Main Workflow

Recommended order:

1. run `notebooks/01_data_loading.ipynb`
2. run `notebooks/02_eda.ipynb`
3. run `notebooks/03_feature_engineering.ipynb`
4. run `notebooks/04_ml_pycaret_suite.ipynb`
5. use outputs in `reports/` and `dashboards/`

## Tech Stack

Python | Pandas | PyCaret | Scikit-learn | Power BI | DAX | SVG | Jupyter

## Notes

1. raw data, large artifacts, environments, and local models are intentionally not meant for GitHub upload
2. use the `github_upload_bundle/` folder if you want a cleaner upload-ready version of the repo

## Recommended GitHub Upload

If uploading to GitHub, include:

1. `notebooks/`
2. `pipelines/`
3. `dashboards/`
4. `reports/`
5. `PROJECT_CONTEXT.md`
6. `SESSION_CHANGELOG.md`
7. `requirements.txt`
8. `README.md`
9. `.gitignore`

Exclude:

1. `.venv/`
2. `data/`
3. `models/`
4. temp and log files
