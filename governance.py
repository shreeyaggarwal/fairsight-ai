"""Transparent educational risk-screening rules, not legal standards."""

import numpy as np
import pandas as pd


def calculate_findings(groups: pd.DataFrame, overall: dict):
    gap = overall['selection_gap']
    selection = ('Insufficient data' if groups['count'].min() < 30 else
                 'High' if gap >= .15 else 'Moderate' if gap >= .08 else 'Low')
    tpr = groups['true_positive_rate'].dropna()
    tpr_gap = float(tpr.max() - tpr.min()) if len(tpr) >= 2 else np.nan
    tpr_priority = ('Insufficient data' if np.isnan(tpr_gap) or groups['count'].min() < 30 else
                    'High' if tpr_gap >= .15 else 'Moderate' if tpr_gap >= .08 else 'Low')
    accuracy, auc = overall['accuracy'], overall['auc']
    performance = 'High' if accuracy < .70 or auc < .70 else 'Moderate' if accuracy < .80 else 'Low'
    findings = pd.DataFrame([
        {'Risk category': 'Selection-rate disparity', 'Review priority': selection,
         'Evidence': f'{gap:.1%} group gap', 'RMF function': 'MEASURE'},
        {'Risk category': 'True-positive-rate disparity', 'Review priority': tpr_priority,
         'Evidence': 'Not estimable' if np.isnan(tpr_gap) else f'{tpr_gap:.1%} group gap', 'RMF function': 'MEASURE'},
        {'Risk category': 'Historical-label agreement', 'Review priority': performance,
         'Evidence': f'{accuracy:.1%} agreement; AUC {auc:.3f}', 'RMF function': 'MEASURE'},
    ])
    counts = findings['Review priority'].value_counts()
    return findings, {'high': int(counts.get('High', 0)),
                      'moderate': int(counts.get('Moderate', 0)),
                      'low': int(counts.get('Low', 0)),
                      'unassessable': int(counts.get('Insufficient data', 0)),
                      'tpr_gap': tpr_gap}


GOVERNANCE_ACTIONS = [
    ('GOVERN', 'Assign model ownership, approval authority, and human review.'),
    ('MAP', 'Document intended use, affected people, data lineage, and label quality.'),
    ('MEASURE', 'Investigate group outcomes, small samples, proxy features, and uncertainty.'),
    ('MANAGE', 'Record mitigation decisions and monitor drift, complaints, and appeals.'),
]
