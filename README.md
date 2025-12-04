# customer-behaviour-analysis

## Project overview

Analyze synthetic e-commerce customer activity to extract actionable business insights and build predictive models that help improve sales, engagement, and personalization. This repo contains data generation, EDA, feature engineering, modeling, big-data processing examples, visualizations, and a final report.

## Objectives

- Perform EDA and data cleaning.
- Engineer features (time, recency/frequency/monetary, session patterns, product affinities).
- Build and validate predictive model(s) (e.g., churn classification for now).
- Use a big-data tool for ingestion/processing and justify the choice.
- Produce insights, dashboards, and a documented report.

## Synthetic dataset (generation)

For this section, I leveraged on an ecommerce dataset available on Kaggle. More information on the attributes in the documentation

## Recommended environment & tools

- Languages: Python 3.12.3 with a Virtual Environment
- Libraries: pandas, numpy, scikit-learn, xgboost/lightgbm, imbalanced-learn, statsmodels, matplotlib, seaborn, plotly
- Notebooks: Jupyter / JupyterLab

## Project structure (suggested)

- datasets/
  - E commerce Dataset
- notebooks/
  - analysis.ipynb
- scripts/
  - preprocess.py
  - train_model.py
  - evaluate.py
  - kafka_producer.py
  - es_ingest.py
- reports/
  - final_report.md
- dashboards/
  - kibana_saved_objects.json
- setup/
  - requirements.txt
- README.md

## Step-by-step workflow

1. Data generation

   - Run scripts/generate_synthetic.py to create events, sessions, orders.
   - Save as Parquet/CSV; partition by date for scalability.

2. Data ingestion (big-data simulation)

   - Start Kafka and create topic(s) for events.
   - Use kafka_producer.py to stream generated events.
   - Use a consumer (or Logstash/Beats) to push events into Elasticsearch for indexing and visualization in Kibana.
   - Alternatively, batch-ingest Parquet via Spark.

3. Exploratory Data Analysis (EDA)

   - Inspect schema, data types, value ranges, distributions.
   - Identify missing values, outliers, and duplicates.
   - Visualize:
     - Time series: daily visits, orders, revenue.
     - Cohort/retention analysis.
     - Top products, categories, funnels (view→add→purchase).
     - Customer demographic breakdown.

4. Data cleaning

   - Drop/repair duplicates and erroneous timestamps.
   - Impute or remove missing values based on column importance.
   - Normalize categorical levels; apply consistent timezone to timestamps.

5. Feature engineering

   - Time features: hour, day_of_week, recency, tenure.
   - RFM: recency, frequency, monetary.
   - Session features: avg_session_length, events_per_session, cart_abandon_rate.
   - Behavioral features: product_affinity vectors, category_counts, device_preference.
   - Encoding: one-hot for low-cardinality, target/embedding for high-cardinality.
   - Scaling: StandardScaler/MinMax for models that need it.

6. Modeling

   - Define problem(s): e.g., churn classification (binary), next-purchase prediction (classification), sales forecasting (regression).
   - Split: time-aware train/validation/test or K-fold / time series CV.
   - Models: baseline logistic regression; tree-based (RandomForest, XGBoost/LightGBM); neural nets for embeddings if needed.
   - Handle imbalance: class weighting, SMOTE, focal loss.

7. Validation & metrics

   - Classification: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix.
   - Regression/forecasting: RMSE, MAE, MAPE.
   - Use cross-validation and time-series validation when appropriate.
   - Calibrate probabilities and evaluate business KPIs (e.g., lift, conversion impact).

8. Insights & visualization

   - Produce dashboards in Kibana/Grafana showing:
     - Active users, revenue trends, churn risk segments.
     - Funnel drop-offs and recommended interventions.
     - Top products by revenue and by conversion rate.
   - Summarize actionable recommendations: targeted marketing for high-risk churn segments, catalog changes, A/B experiments.

9. Documentation & presentation
   - Provide a final report covering methodology, key findings, model performance, limitations, and business recommendations.
   - Include commentary and docstrings in notebooks and scripts.
   - Save reproducible artifacts: trained model, preprocessing pipeline, sample inference script.

## How to run (example)

- Generate data:
  - python scripts/generate_synthetic.py --num_customers 10000 --num_events 500000
- Start Kafka & Elasticsearch (docker-compose provided optional)
- Stream events:
  - python scripts/kafka_producer.py --input data/synthetic/events.parquet --topic events
- Run preprocessing & train:
  - python scripts/preprocess.py --input data/synthetic --output data/processed
  - python scripts/train_model.py --data data/processed --model_out models/churn.pkl

## Deliverables

- Synthetic datasets (CSV/Parquet)
- Notebooks (EDA, FE, modeling)
- Scripts for data generation, ingestion, preprocessing, and training
- Saved models and evaluation metrics
- Kibana/Grafana dashboards + final report with actionable recommendations

## Notes on technology choice (Kafka + Elasticsearch)

- Kafka: durable, high-throughput event streaming that simulates live user behavior and decouples producers/consumers.
- Elasticsearch + Kibana: fast indexing, full-text search and aggregations; Kibana provides immediate dashboarding for event analytics.
- This stack allows easy scaling from sample data to realistic big-data workflows and rapid exploratory visualization.

## Contribution & license

- PRs welcome. Include tests and update notebooks when adding features.
- Add preferred open-source license file.

For more detailed commands and examples, open the notebooks in /notebooks.
