# E-Commerce OLIST Analytics and ML Dashboard

## Live Dashboard Link:
https://app.powerbi.com/view?r=eyJrIjoiZjljZGJmZWQtMTRmMC00OWJiLWE4ZjgtZTQzZTVkZWU3OWUwIiwidCI6ImU1NDhjMjU2LTRkODMtNDRiMi1iZWM2LTcwZDhhOTFhYzIxZSJ9&pageName=76a7434066ccc0018243

## kaggle Datset:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
Built on the Olist Brazilian E-Commerce public dataset from Kaggle — 100K+ orders, 9 tables, real-world complexity.

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
4. `docs/`
   1. exported metrics, predictions, quality reports, and project documentation

## What I built:
1. 4-page interactive dashboard (Executive, Operations, Products, ML Intel)
2. 5 ML tracks: Classification, Regression, Clustering, Anomaly Detection, Time Series
3. Full data pipeline from raw CSV to Power BI semantic model
4. Automated BI exports with model performance tracking


## Main Workflow

Recommended order:

1. run `notebooks/01_data_loading.ipynb`
2. run `notebooks/02_eda.ipynb`
3. run `notebooks/03_feature_engineering.ipynb`
4. run `notebooks/04_ml_pycaret_suite.ipynb`

## Tech Stack

Python | Pandas | PyCaret | Scikit-learn | Power BI | DAX | Jupyter
