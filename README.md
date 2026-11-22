# FairSight AI — Risk Intelligence Dashboard

An interactive AI governance portfolio prototype built with **Python, Streamlit, scikit-learn, SHAP, Plotly, and ReportLab**. It extends the original FairSight AI project with a polished multipage interface and a more complete PDF audit.

> **Important:** This is an educational demonstration. The built-in data are synthetic. The model predicts simulated *historical screening decisions*, not applicant ability or job performance. Do not use it to make real hiring decisions. Group disparities are screening signals, not proof of discrimination, fairness, regulatory compliance, or NIST certification.

## Start on your Mac

1. **Back up your current `fairsight-ai` folder.** This ZIP is a complete replacement, not a patch. It incorporates the global SHAP chart, individual explanations, and PDF export you already built.
2. Unzip the downloaded project to a **new folder** (for example, `Downloads/FairSight-Redesign`) so your original stays safe.
3. In VS Code choose **File → Open Folder**, then open the extracted `fairsight-ai` folder.
4. Open **Terminal → New Terminal** and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The application normally opens at `http://localhost:8501`. If the old dashboard is still running, stop it with **Control+C** in its Terminal before starting this one. On Windows use `.venv\Scripts\activate` instead.

## What is included?

| Page | Features |
|---|---|
| **Overview** | Model agreement, AUC, fairness gap, risk summary, screening chart |
| **Fairness analysis** | Group selection rates, TPR, FPR, properly formatted **0–100%** chart axis |
| **Explainable AI** | SHAP global feature importance and selectable individual explanations |
| **Governance & reports** | Transparent risk register, NIST AI RMF-inspired actions, downloadable multi-page PDF |
| **Sidebar** | CSV upload, data-source selector, decision threshold |

CSV schema: `group`, `years_experience`, `assessment_score`, `skills_score`, `historical_selected` (binary 0/1). Optional: `applicant_id`. Supply only synthetic or properly authorized, de-identified information; never upload real applicant personal details to this learning app.

## Methodology and limitations

- Logistic regression is trained on a stratified **70% split**, then evaluated on a **30% held-out split** using a fixed random seed (42).
- SHAP uses the first 200 training rows as its bounded reference cohort; values are explained on the logistic model's log-odds scale.
- Audit group is *excluded from the training features*, but other inputs may still serve as proxies.
- Accuracy, AUC, TPR, and FPR use historical selections as labels. Those labels can be biased and do **not** establish who is qualified.
- Group selection rates are reported as percentages; gaps are percentage-point differences.
- The risk priorities use intentionally simple demonstration cutoffs: selection/TPR gaps of 8 or 15 percentage points, historical-label agreement of 70%/80%, and AUC of 0.70. They are **not** legal or NIST requirements.
- Small group sample sizes are flagged; this prototype does not compute statistical significance, confidence intervals, intersectional disparity, job-related validity, or counterfactual fairness.
- A selection-rate ratio is a descriptive comparison, not a legal determination.
- Do not claim that the dashboard is an independent employment compliance audit or an endorsed NIST product.

## Test the project

```bash
python -m unittest discover -s tests -v
```

## Project files

```text
fairsight-ai/
├── app.py                 # Streamlit navigation and charts
├── analysis.py            # Model, held-out evaluation, SHAP explanations
├── governance.py          # Explicit risk screening criteria
├── pdf_report.py          # Professional in-memory PDF generator
├── generate_data.py       # Deterministic fictional applicant data
├── data/                  # Included synthetic demo CSV
├── tests/                 # Core regression checks
├── requirements.txt
└── README.md
```

## Recruiter demo flow (90 seconds)

1. **Overview (15s):** Explain the fictional employer's screening problem and the prototype scope.
2. **Fairness (25s):** Change the threshold and discuss the group selection-rate gap versus TPR disparity.
3. **Explainability (25s):** Show global and individual SHAP factors; stress that SHAP describes association, not causality.
4. **Governance (25s):** Show risk priorities, map suggested controls to NIST's Govern / Map / Measure / Manage, and download the PDF memo.

**Suggested resume line:** Developed FairSight AI, an interactive Python/Streamlit risk-intelligence prototype that assesses group-level screening disparities, visualizes SHAP explanations, and generates reproducible governance reports using synthetic hiring data.

## References

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [NIST AI RMF Playbook](https://airc.nist.gov/airmf-resources/playbook/)
- [EEOC Employment Tests and Selection Procedures](https://www.eeoc.gov/laws/guidance/employment-tests-and-selection-procedures)

