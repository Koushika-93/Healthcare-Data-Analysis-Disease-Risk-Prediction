# Healthcare Data Analysis for Disease Risk Prediction

## Project overview
A beginner-friendly Python project demonstrating healthcare-style tabular data analysis, cleaning, visualization, and a baseline machine-learning classification workflow.

## Important limitation
The included CSV contains **synthetic data** generated for this educational project. The `Risk_Label` is rule-generated, not based on clinician-adjudicated outcomes. This project is **not a medical device**, does not diagnose disease, and must not be used for treatment or real patient decisions. Model scores only describe performance on this synthetic dataset.

## Folder structure
```text
Healthcare_Data_Analysis_for_Disease_Risk_Prediction/
├── dataset/healthcare_risk_synthetic.csv
├── notebook/healthcare_risk_analysis.ipynb
├── visualizations/
├── dashboard/app.py
├── report/
├── requirements.txt
└── README.md
```

## Setup
1. Install Python 3.10 or newer.
2. Open a terminal in the project folder.
3. Run `pip install -r requirements.txt`.
4. To run the notebook, start `jupyter notebook` and open `notebook/healthcare_risk_analysis.ipynb`.
5. To run the dashboard, execute `streamlit run dashboard/app.py`.

## Workflow
1. Load the synthetic CSV using Pandas.
2. Inspect columns, missing values, duplicates, and summary statistics.
3. Visualize label counts, age distribution, and numeric correlations.
4. Train/test split the data and train a baseline Random Forest classifier.
5. Review classification metrics and discuss the limitations of synthetic labels.

## Suggested charts
- Synthetic risk-label distribution
- Age distribution by label
- Correlation heatmap of age, BMI, blood pressure, glucose, and cholesterol
- Average health measures by label

## Ethical considerations
Real healthcare prediction requires representative, consented or otherwise lawfully accessed, de-identified data; clinical oversight; bias assessment; external validation; privacy protection; and transparent communication of uncertainty.
