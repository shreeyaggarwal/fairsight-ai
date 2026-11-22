"""Recruiter-demo PDF audit export based on the on-screen assessment."""

from __future__ import annotations

from datetime import date
from html import escape
from io import BytesIO

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)

NAVY = colors.HexColor('#14263E')
INK = colors.HexColor('#273A51')
TEAL = colors.HexColor('#087C81')
PALE = colors.HexColor('#EDF6F6')
LIGHT = colors.HexColor('#F3F6FA')
LINE = colors.HexColor('#DBE3EB')
MUTED = colors.HexColor('#617287')


def _format_percent(value: float) -> str:
    return 'N/A' if pd.isna(value) else f'{value:.1%}'


def _footer(canvas, doc):
    canvas.saveState()
    width, _ = letter
    canvas.setStrokeColor(LINE)
    canvas.line(48, 47, width - 48, 47)
    canvas.setFillColor(MUTED)
    canvas.setFont('Helvetica', 8)
    canvas.drawString(48, 33, 'FAIRSIGHT AI  /  EDUCATIONAL RISK REVIEW')
    canvas.drawRightString(width - 48, 33, f'Page {doc.page}')
    canvas.restoreState()


def _table(rows, widths, *, header=True):
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign='LEFT')
    commands = [
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ('LEFTPADDING', (0, 0), (-1, -1), 11),
        ('RIGHTPADDING', (0, 0), (-1, -1), 11),
        ('LINEBELOW', (0, -1), (-1, -1), .5, LINE),
    ]
    if header:
        commands.extend([
            ('BACKGROUND', (0, 0), (-1, 0), NAVY),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ])
    else:
        commands.append(('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, LIGHT]))
    table.setStyle(TableStyle(commands))
    return table


def generate_audit_pdf(overall: dict, groups: pd.DataFrame,
                       findings: pd.DataFrame, *, source_label: str) -> bytes:
    """Return a fully in-memory PDF; no real applicant data is embedded."""
    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter, rightMargin=48, leftMargin=48,
                            topMargin=48, bottomMargin=68, title='FairSight AI Governance Audit',
                            author='FairSight AI Educational Prototype')
    normal = getSampleStyleSheet()
    normal.add(ParagraphStyle(name='HeroF', fontName='Helvetica-Bold', fontSize=23,
                              leading=28, textColor=NAVY, spaceAfter=7))
    normal.add(ParagraphStyle(name='SubtitleF', fontSize=9, leading=14,
                              textColor=MUTED, spaceAfter=18))
    normal.add(ParagraphStyle(name='SectionF', fontName='Helvetica-Bold', fontSize=12,
                              leading=16, textColor=NAVY, spaceBefore=19, spaceAfter=9))
    normal.add(ParagraphStyle(name='BodyF', fontSize=9, leading=14, textColor=INK,
                              spaceAfter=8))
    normal.add(ParagraphStyle(name='SmallF', fontSize=8, leading=12, textColor=MUTED))
    normal.add(ParagraphStyle(name='CellF', fontSize=8, leading=11, textColor=INK))
    normal.add(ParagraphStyle(name='WhiteF', fontName='Helvetica-Bold', fontSize=8,
                              leading=11, textColor=colors.white))
    p = lambda content: Paragraph(escape(str(content)), normal['CellF'])
    h = lambda content: Paragraph(escape(str(content)), normal['WhiteF'])
    flow = [
        Paragraph('FAIRSIGHT AI', normal['HeroF']),
        Paragraph('AI GOVERNANCE  /  MODEL RISK ASSESSMENT  /  EDUCATIONAL DEMO', normal['SubtitleF']),
        _table([[p('ASSESSMENT DATE'), p('DATA SOURCE'), p('HOLDOUT RECORDS')],
                [p(date.today().isoformat()), p(source_label), p(overall['test_count'])]],
               [150, 216, 150], header=False),
        Spacer(1, 12),
        Paragraph('Executive summary', normal['SectionF']),
        Paragraph(
            'This assessment examines whether a logistic-regression model reproducing '
            'historical screening decisions yields different outcomes across audit groups. '
            'The findings are preliminary review signals, not compliance conclusions.', normal['BodyF']),
        Paragraph('01  /  Model performance', normal['SectionF']),
    ]
    perf = [
        [h('METRIC'), h('RESULT')],
        [p('Agreement with historical labels'), p(_format_percent(overall['accuracy']))],
        [p('ROC AUC vs. historical labels'), p(f"{overall['auc']:.3f}")],
        [p('Group selection-rate gap'), p(_format_percent(overall['selection_gap']))],
        [p('Lowest / highest selection-rate ratio'), p('N/A' if pd.isna(overall['selection_ratio']) else f"{overall['selection_ratio']:.2f}")],
        [p('Decision threshold'), p(f"{overall['threshold']:.2f}")],
    ]
    flow.extend([_table(perf, [340, 176]),
                 Paragraph('02  /  Governance risk findings', normal['SectionF'])])
    findings_rows = [[h('REVIEW AREA'), h('PRIORITY'), h('EVIDENCE')]]
    findings_rows.extend([[p(row['Risk category']), p(row['Review priority']), p(row['Evidence'])]
                          for _, row in findings.iterrows()])
    flow.extend([
        _table(findings_rows, [214, 106, 196]),
        Paragraph('Ratings use transparent demonstration thresholds and are not statutory standards.',
                  normal['SmallF']),
        PageBreak(),
        Paragraph('03  /  Group-level measurements', normal['SectionF']),
    ])
    group_rows = [[h('GROUP'), h('COUNT'), h('SELECT RATE'), h('TPR'), h('FPR')]]
    for row in groups.itertuples(index=False):
        group_rows.append([p(str(row.group)[:40]), p(row.count), p(_format_percent(row.selection_rate)),
                           p(_format_percent(row.true_positive_rate)), p(_format_percent(row.false_positive_rate))])
    flow.append(_table(group_rows, [156, 70, 104, 93, 93]))
    flow.append(Paragraph('TPR and FPR compare model predictions with historical selections, '
                          'not verified applicant qualifications.', normal['SmallF']))
    flow.append(Paragraph('04  /  Recommended governance actions', normal['SectionF']))
    actions = [
        ('GOVERN', 'Assign a model owner, approval authority, and human review pathway.'),
        ('MAP', 'Document intended use, affected groups, data provenance, and label quality.'),
        ('MEASURE', 'Investigate disparities, proxy signals, subgroup sizes, and uncertainty.'),
        ('MANAGE', 'Record mitigation decisions, monitoring plans, and candidate appeals.'),
    ]
    for tag, action in actions:
        flow.append(Paragraph(f'<b>{tag}</b> - {escape(action)}', normal['BodyF']))
    flow.append(Paragraph('05  /  Methodology & limitations', normal['SectionF']))
    limitations = (
        f"Source: {escape(source_label)}. The model trains on 70% of records and is evaluated on a 30% "
        'stratified holdout. It excludes group membership from training, but other inputs may be proxies. '
        'Accuracy and AUC measure agreement with potentially biased historical decisions, not job performance. '
        'Sampling uncertainty, intersectional groups, and job-related validity are not established. '
        'The 8- and 15-point gaps and 70/80% agreement thresholds are illustrative project choices. '
        'This is not legal advice, a hiring recommendation, NIST certification, or proof of fairness.'
    )
    flow.append(Paragraph(limitations, normal['BodyF']))
    flow.append(Paragraph('Reference: NIST AI RMF - nist.gov/itl/ai-risk-management-framework', normal['SmallF']))
    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)
    return output.getvalue()

