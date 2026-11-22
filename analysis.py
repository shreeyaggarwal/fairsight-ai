"""Model training and group fairness diagnostics; not legal determinations."""
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['years_experience', 'assessment_score', 'skills_score']
TARGET = 'historical_selected'
REQUIRED = {'group', TARGET, *FEATURES}


def evaluate(df: pd.DataFrame, threshold: float = 0.5):
    df = df.copy().reset_index(drop=True)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {", ".join(sorted(missing))}')
    if df.empty or len(df) < 50:
        raise ValueError('Provide at least 50 applicant records.')
    if len(df) > 25000:
        raise ValueError('This demo accepts at most 25,000 records.')
    for feature in FEATURES + [TARGET]:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
    if df[list(FEATURES)].isna().any().any() or df[[TARGET, 'group']].isna().any().any():
        raise ValueError('Required columns cannot contain missing values.')
    if not np.isfinite(df[FEATURES + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError('Numeric fields cannot contain infinite values.')
    if not set(df[TARGET].unique()).issubset({0, 1}) or df[TARGET].nunique() < 2:
        raise ValueError('historical_selected must contain both 0 and 1 labels.')
    if df['group'].nunique() < 2:
        raise ValueError('At least two groups are required.')
    if df['group'].nunique() > 12:
        raise ValueError('The demo supports at most 12 audit groups.')
    if df[TARGET].value_counts().min() < 4:
        raise ValueError('At least four examples of each historical outcome are required.')
    if 'applicant_id' not in df.columns:
        df['applicant_id'] = [f'ROW-{i:04d}' for i in range(1, len(df) + 1)]
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        df[FEATURES], df[TARGET], df.index, test_size=0.3, random_state=42,
        stratify=df[TARGET])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(x_train, y_train)
    probs = model.predict_proba(x_test)[:, 1]
    predictions = (probs >= threshold).astype(int)
    test = df.loc[idx_test, ['applicant_id', 'group', TARGET, *FEATURES]].copy()
    test['predicted_selected'] = predictions
    test['predicted_probability'] = probs
    test = test.reset_index(drop=True)
    rows = []
    for group, subset in test.groupby('group'):
        actual = subset[TARGET].to_numpy()
        pred = subset['predicted_selected'].to_numpy()
        positive = actual == 1
        negative = actual == 0
        rows.append({'group': str(group), 'count': len(subset), 'selected': int(pred.sum()),
                     'selection_rate': float(pred.mean()),
                     'true_positive_rate': float(pred[positive].mean()) if positive.any() else np.nan,
                     'false_positive_rate': float(pred[negative].mean()) if negative.any() else np.nan})
    metrics = pd.DataFrame(rows).sort_values('group').reset_index(drop=True)
    selection_gap = float(metrics['selection_rate'].max() - metrics['selection_rate'].min())
    min_rate = float(metrics['selection_rate'].min())
    max_rate = float(metrics['selection_rate'].max())
    selection_ratio = min_rate / max_rate if max_rate > 0 else np.nan
    overall = {'accuracy': float(accuracy_score(y_test, predictions)),
               'auc': float(roc_auc_score(y_test, probs)),
               'selection_gap': selection_gap, 'selection_ratio': selection_ratio,
               'test_count': len(test), 'source_count': len(df), 'threshold': threshold}
    scaler = model.named_steps['standardscaler']
    regression = model.named_steps['logisticregression']
    # A fixed, bounded reference cohort keeps reruns fast and avoids subsampling warnings.
    background = scaler.transform(x_train)[:200]
    masker = shap.maskers.Independent(background, max_samples=200)
    explainer = shap.LinearExplainer(regression, masker, feature_names=FEATURES)
    shap_values = explainer(scaler.transform(x_test))
    return model, test, metrics, overall, shap_values
