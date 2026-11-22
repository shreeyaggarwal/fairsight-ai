"""Generate synthetic hiring records for educational fairness analysis."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    group = rng.choice(['Group A', 'Group B'], n, p=[0.52, 0.48])
    experience = rng.integers(0, 13, n)
    assessment = np.clip(rng.normal(67, 16, n), 0, 100).round(1)
    skills = rng.integers(1, 11, n)
    # Synthetic historical labels intentionally contain a group-linked disparity.
    score = -4.8 + 0.21 * experience + 0.038 * assessment + 0.22 * skills - 0.75 * (group == 'Group B')
    historical_selected = rng.binomial(1, 1 / (1 + np.exp(-score)))
    return pd.DataFrame({'applicant_id': [f'APP-{i:04d}' for i in range(1, n + 1)],
                         'group': group, 'years_experience': experience,
                         'assessment_score': assessment, 'skills_score': skills,
                         'historical_selected': historical_selected})

if __name__ == '__main__':
    output = Path(__file__).parent / 'data'
    output.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(output / 'synthetic_applicants.csv', index=False)
    print(f'Wrote {len(df)} synthetic rows')
