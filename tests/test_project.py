"""Run with: python -m unittest discover -s tests -v"""

import io
import unittest

import numpy as np
import pandas as pd

from analysis import FEATURES, evaluate
from generate_data import generate
from governance import calculate_findings
from pdf_report import generate_audit_pdf


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate()
        cls.model, cls.audit, cls.groups, cls.overall, cls.shap_values = evaluate(cls.data)
        cls.findings, cls.risks = calculate_findings(cls.groups, cls.overall)

    def test_heldout_split(self):
        self.assertEqual(len(self.audit), 360)
        self.assertEqual(len(self.groups), 2)
        self.assertTrue(self.audit['predicted_probability'].between(0, 1).all())

    def test_shap_alignment(self):
        self.assertEqual(self.shap_values.values.shape,(360,len(FEATURES)))
        self.assertTrue(np.isfinite(self.shap_values.values).all())
        self.assertNotIn('group', FEATURES)

    def test_threshold_changes_predictions(self):
        _, low, _, _, _ = evaluate(self.data,0.35)
        _, high, _, _, _ = evaluate(self.data,0.65)
        self.assertGreaterEqual(low['predicted_selected'].sum(),high['predicted_selected'].sum())

    def test_group_metrics(self):
        self.assertEqual(self.groups['count'].sum(),len(self.audit))
        self.assertTrue(self.groups['selection_rate'].between(0,1).all())
        self.assertTrue(0 <= self.overall['selection_gap'] <= 1)

    def test_risk_register(self):
        self.assertEqual(len(self.findings),3)
        self.assertEqual(self.risks['high']+self.risks['moderate']+self.risks['low']+self.risks['unassessable'],3)

    def test_upload_validation(self):
        with self.assertRaisesRegex(ValueError,'Missing'):
            evaluate(self.data.drop(columns=['skills_score']))
        with self.assertRaisesRegex(ValueError,'infinite'):
            broken=self.data.copy().astype({'skills_score':'float64'})
            broken.loc[0,'skills_score']=float('inf')
            evaluate(broken)

    def test_pdf(self):
        result = generate_audit_pdf(self.overall,self.groups,self.findings,
                                    source_label='Synthetic demonstration')
        self.assertTrue(result.startswith(b'%PDF-'))
        self.assertGreater(len(result),3000)


if __name__ == '__main__':
    unittest.main()
