# FairSight AI
### AI Hiring Fairness & Governance Dashboard

**FairSight AI** is an interactive AI governance dashboard built with Python and Streamlit to evaluate machine-learning hiring models for predictive performance, potential fairness disparities, explainability, and governance risks.

The project demonstrates how organizations can assess AI-assisted hiring systems through quantitative analysis and structured governance reviews.

**[Live Dashboard](https://fairsight-ai-rg06.onrender.com)**

## Key Features

- **Model Performance:** Evaluate predictive accuracy and ROC AUC using held-out test data.
- **Fairness Analysis:** Compare selection rates across applicant groups and identify potential disparities.
- **Explainable AI:** Explore global and individual model explanations using SHAP.
- **Governance Audit:** Review potential AI risks and recommended mitigation actions using a framework inspired by the NIST AI Risk Management Framework.
- **CSV Upload:** Upload compatible datasets for interactive analysis.
- **PDF Reporting:** Generate downloadable governance audit reports.

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- SHAP
- Matplotlib
- ReportLab

## How to Run Locally

1. Clone this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the application:

```bash
streamlit run app.py
```

## Project Purpose

AI-assisted hiring systems can introduce risks involving fairness, transparency, accountability, and model reliability.

FairSight AI explores how technical model evaluation and responsible AI governance practices can be combined into a practical decision-support dashboard.

## Limitations & Responsible Use

- The included demonstration dataset is synthetic.
- Fairness metrics are diagnostic indicators, not definitive proof of discrimination or fairness.
- SHAP explanations describe model behavior and do not establish causation.
- Governance findings are educational and do not constitute legal advice, regulatory certification, or a formal compliance audit.
- The dashboard is a portfolio demonstration and is not intended for real-world employment decisions without further validation, oversight, and safeguards.

## Developer

**Shreeya Aggarwal**

Business Analytics Undergraduate | University of California, Riverside

Interests: AI Governance, Responsible AI, Algorithmic Fairness, and AI Risk Management

[GitHub Profile](https://github.com/shreeyaggarwal)
