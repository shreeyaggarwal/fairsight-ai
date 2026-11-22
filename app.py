"""FairSight AI | polished, educational AI governance dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import FEATURES, evaluate
from generate_data import generate
from governance import GOVERNANCE_ACTIONS, calculate_findings
from pdf_report import generate_audit_pdf


NAVY = '#12283F'
TEAL = '#087F82'
MUTED = '#6A7B8C'
PAGE = '#F6F8FB'
GREEN = '#16745E'
ORANGE = '#B66A22'
RED = '#B23B4A'

st.set_page_config(page_title='FairSight AI | Risk Intelligence', page_icon='🔎', layout='wide',
                   initial_sidebar_state='expanded')
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"] {font-family:'DM Sans',sans-serif}
[data-testid="stAppViewContainer"] {background:#F6F8FB}
[data-testid="stSidebar"] {background:linear-gradient(165deg,#142B45 0%,#0E233A 100%)}
[data-testid="stSidebar"] * {color:#E8F1F6!important}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {color:#CCDCE8!important}
[data-testid="stSidebar"] div[role="radiogroup"] label {padding:6px 0}
.block-container {max-width:1240px;padding-top:1.6rem;padding-bottom:4rem}
h1,h2,h3 {font-family:'Manrope','DM Sans',sans-serif;letter-spacing:-.035em;color:#152C44}
h1 {font-size:2.22rem!important;font-weight:800!important}
h2 {font-size:1.46rem!important;margin-top:0}
h3 {font-size:1.06rem!important}
[data-testid="stMetric"] {border:1px solid #DEE7EE;background:#FFFFFF;border-radius:15px;
 box-shadow:0 5px 20px rgba(18,40,63,.035);padding:17px 20px}
[data-testid="stMetricLabel"] p {color:#607588!important;font-weight:600}
[data-testid="stMetricValue"] {color:#172F48;font-family:'Manrope',sans-serif;font-weight:800}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:#DDE7EF!important}
.f-eyebrow {color:#087F82;font-weight:800;letter-spacing:.17em;font-size:.71rem}
.f-hero {border-radius:18px;padding:26px 30px;margin:2px 0 18px;
background:linear-gradient(112deg,#132C47 0%,#1A3F59 60%,#086F78 100%);color:#E8F4F6}
.f-hero h1 {color:white!important;font-size:2.1rem!important;margin:7px 0}
.f-hero p {color:#C7E0E7;margin:4px 0;max-width:720px}
.f-hero span {color:#A7F1E3;font-size:.76rem;font-weight:800;letter-spacing:.15em}
.f-panel {border:1px solid #E0E8F0;background:#FFFFFF;border-radius:15px;padding:19px 23px;margin:10px 0}
.f-panel h4 {margin:0 0 7px;color:#17314B;font-weight:800;font-size:1rem}
.f-panel p {margin:0;color:#607487;line-height:1.65;font-size:.91rem}
.f-tag {display:inline-block;background:#E2F4F2;color:#067770!important;border-radius:20px;
padding:5px 10px;font-size:.72rem;font-weight:700;letter-spacing:.035em}
.f-warning {background:#FFF8EA;border-left:3px solid #DCA44E;border-radius:9px;
padding:13px 17px;color:#685333;font-size:.87rem;line-height:1.55;margin:12px 0}
[data-testid="stFileUploader"] {border-radius:10px}
footer {visibility:hidden}
</style>''', unsafe_allow_html=True)


def section_heading(number: str, title: str, description: str):
    st.markdown(f'<div class="f-eyebrow">{number} / FAIRSIGHT INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def card(title: str, body: str):
    st.markdown(f'<div class="f-panel"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def chart_style(fig, *, height=330):
    fig.update_layout(
        height=height, margin=dict(l=10,r=12,t=15,b=20),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, Arial',color='#344A60',size=12),
        showlegend=False, hoverlabel=dict(bgcolor='#17314B',font_color='white'),
    )
    fig.update_xaxes(showgrid=False,zeroline=False,linecolor='#DCE5ED')
    fig.update_yaxes(gridcolor='#E8EEF3',zeroline=False)
    return fig


def pct(value, decimals=1):
    return 'N/A' if pd.isna(value) else f'{value:.{decimals}%}'


with st.sidebar:
    st.markdown('### ◈ FairSight AI')
    st.caption('RISK INTELLIGENCE PLATFORM')
    st.divider()
    page = st.radio('WORKSPACE', ['Overview', 'Fairness analysis', 'Explainable AI', 'Governance & reports'],
                    key='page_navigation')
    st.divider()
    st.markdown('**ASSESSMENT SETTINGS**')
    source = st.radio('Data source', ['Synthetic demo', 'Upload CSV'], key='dataset_source')
    uploaded = st.file_uploader('Applicant CSV', type='csv', key='applicant_csv') if source == 'Upload CSV' else None
    threshold = st.slider('Selection threshold', min_value=0.10, max_value=0.90,
                          value=0.50, step=0.05, key='decision_threshold',
                          help='Predicted probabilities at or above this level are screened in.')
    st.divider()
    st.caption('DEMO ONLY • NOT FOR REAL HIRING')
    st.caption('Group labels are excluded from training; proxy effects can still occur.')

if source == 'Upload CSV' and uploaded is None:
    section_heading('00', 'Upload your assessment data',
                    'The dashboard will run when a valid CSV is provided.')
    st.info('Required columns: group, years_experience, assessment_score, skills_score, historical_selected (0/1). Optional: applicant_id.')
    st.warning('Use fictional or appropriately approved, de-identified data. Do not upload real applicant personal information to this educational demo.')
    st.stop()

try:
    raw_data = generate() if source == 'Synthetic demo' else pd.read_csv(uploaded)
    model, test, group_metrics, overall, shap_values = evaluate(raw_data, threshold)
except (ValueError, KeyError, pd.errors.ParserError) as exc:
    st.error(f'Unable to analyze this dataset: {exc}')
    st.stop()

findings, risk_summary = calculate_findings(group_metrics, overall)
data_label = 'Synthetic demonstration' if source == 'Synthetic demo' else 'User-supplied CSV'
st.markdown(
    '<div class="f-warning"><b>Educational risk-screening prototype.</b> '
    'The model predicts historical selections, not verified qualifications or job performance. '
    'This is not a hiring recommendation, legal compliance finding, or NIST certification.</div>',
    unsafe_allow_html=True,
)

if page == 'Overview':
    st.markdown('''<div class="f-hero"><span>MODEL ASSURANCE / WORKSPACE 01</span>
        <h1>AI risk, made visible.</h1>
        <p>Understand screening outcomes, investigate model behavior, and turn evidence into governance actions.</p>
        </div>''', unsafe_allow_html=True)
    st.markdown('<span class="f-tag">● Assessment ready</span>',unsafe_allow_html=True)
    st.write('')
    a,b,c,d = st.columns(4)
    a.metric('Historical-label agreement', pct(overall['accuracy']),help='This is not accuracy against candidate merit.')
    b.metric('ROC AUC', f"{overall['auc']:.3f}",help='Against historical selection labels.')
    c.metric('Selection-rate gap', pct(overall['selection_gap']))
    d.metric('High-priority findings', risk_summary['high'])
    st.write('')
    l,r = st.columns([1.35,1],gap='large')
    with l:
        st.subheader('Screening outcomes')
        fig = px.bar(group_metrics,x='group',y='selection_rate',color='group',
                     color_discrete_sequence=[TEAL,'#4D8EB7','#A2A9E0','#D19B66'])
        fig.update_traces(hovertemplate='%{x}: %{y:.1%}<extra></extra>',marker_line_width=0)
        fig.update_yaxes(tickformat='.0%',range=[0,1],dtick=.2,title=None)
        fig.update_xaxes(title=None)
        st.plotly_chart(chart_style(fig),width='stretch')
    with r:
        st.subheader('Assessment snapshot')
        card('Review scope',f"{overall['source_count']:,} records • {overall['test_count']:,} held-out profiles • {len(group_metrics)} audit groups")
        card('Decision policy',f"Probability threshold: {threshold:.0%}. Adjust it in the sidebar to evaluate different screening outcomes.")
        card('Risk signal',f"{risk_summary['high']} high-priority and {risk_summary['moderate']} moderate-priority findings under illustrative rules.")
    st.subheader('Governance workflow')
    cols = st.columns(4)
    for col,(function,action) in zip(cols,GOVERNANCE_ACTIONS):
        with col:
            card(function,action)

elif page == 'Fairness analysis':
    section_heading('02','Fairness analysis',
                    'Inspect group outcomes in the held-out cohort. Percent axes are fixed to 0–100%.')
    a,b,c,d = st.columns(4)
    a.metric('Selection-rate gap',pct(overall['selection_gap']))
    b.metric('Selection-rate ratio','N/A' if pd.isna(overall['selection_ratio']) else f"{overall['selection_ratio']:.2f}")
    c.metric('Audit groups',str(len(group_metrics)))
    d.metric('Holdout records',f"{overall['test_count']:,}")
    st.subheader('Selection rates by group')
    fig = px.bar(group_metrics,x='group',y='selection_rate',color='group',
                 color_discrete_sequence=[TEAL,'#4D8EB7','#A2A9E0','#D19B66'])
    fig.update_traces(hovertemplate='%{x}: %{y:.1%}<extra></extra>')
    fig.update_yaxes(tickformat='.0%',range=[0,1],dtick=.10,title='Selection rate')
    fig.update_xaxes(title=None)
    st.plotly_chart(chart_style(fig,height=360),width='stretch')
    st.subheader('Additional group diagnostics')
    group_formatted = group_metrics.rename(columns={
        'group':'Group','count':'Records','selected':'Selected',
        'selection_rate':'Selection rate','true_positive_rate':'TPR','false_positive_rate':'FPR'})
    for name in ['Selection rate','TPR','FPR']:
        group_formatted[name] = group_formatted[name].map(pct)
    st.dataframe(group_formatted,hide_index=True,width='stretch')
    st.caption('All rate columns are displayed as percentages; the plot axis is explicitly fixed from 0% to 100%.')
    st.info('TPR and FPR compare predictions with historically selected/not-selected records. Historical labels may themselves be unfair.')
    st.markdown('**How to interpret a gap:** Differences flag a need for investigation; they do not identify causation, statistical significance, business necessity, or legal discrimination.')

elif page == 'Explainable AI':
    section_heading('03','Explainable AI',
                    'SHAP illustrates model associations, not causal effects or applicant suitability.')
    st.subheader('Global feature importance')
    importance = np.abs(shap_values.values).mean(axis=0)
    features = pd.DataFrame({'Feature':FEATURES,'Mean absolute SHAP':importance}).sort_values('Mean absolute SHAP')
    fig = px.bar(features,x='Mean absolute SHAP',y='Feature',orientation='h',
                 color_discrete_sequence=[TEAL])
    fig.update_traces(hovertemplate='%{y}: %{x:.3f} log-odds<extra></extra>')
    fig.update_yaxes(title=None)
    st.plotly_chart(chart_style(fig,height=270),width='stretch')
    st.caption('Mean absolute SHAP contributions are measured in log-odds, not percentages. Higher bars mean greater average influence on modeled historical selections.')
    st.divider()
    st.subheader('Individual prediction explanation')
    choice = st.selectbox('Choose a fictional audit profile',options=list(range(len(test))),
                          format_func=lambda i:f"{test.iloc[i]['applicant_id']} · Profile {i+1}",key='chosen_profile')
    selected = test.iloc[choice]
    k1,k2,k3 = st.columns(3)
    k1.metric('Predicted selection probability',pct(selected['predicted_probability']))
    k2.metric('Model outcome','Screened in' if selected['predicted_selected'] else 'Not screened in')
    k3.metric('Probability threshold',pct(threshold,0))
    st.write('**Recorded inputs**')
    st.dataframe(pd.DataFrame([selected[FEATURES].to_dict()]),hide_index=True,width='stretch')
    effects = pd.DataFrame({'Feature':FEATURES,'Contribution':shap_values.values[choice]})
    effects['Direction'] = np.where(effects['Contribution']>=0,'Raises prediction','Lowers prediction')
    effects = effects.sort_values('Contribution')
    fig = px.bar(effects,x='Contribution',y='Feature',orientation='h',color='Direction',
                 color_discrete_map={'Raises prediction':TEAL,'Lowers prediction':RED})
    fig.update_xaxes(zeroline=True,zerolinecolor='#7C8FA2',title='SHAP contribution (log-odds)')
    fig.update_yaxes(title=None)
    fig.update_layout(showlegend=True,legend=dict(orientation='h',y=1.15,x=0))
    st.plotly_chart(chart_style(fig,height=290).update_layout(showlegend=True),width='stretch')
    st.caption('These are feature contributions to the model log-odds. They are not percentage-point changes or evidence of a legitimate hiring reason.')

elif page == 'Governance & reports':
    section_heading('04','Governance & reports',
                    'A transparent audit summary using NIST AI RMF vocabulary, not a NIST certification.')
    a,b,c,d = st.columns(4)
    a.metric('High priority',risk_summary['high'])
    b.metric('Moderate priority',risk_summary['moderate'])
    c.metric('Low priority',risk_summary['low'])
    d.metric('Insufficient data',risk_summary['unassessable'])
    st.subheader('Review register')
    st.dataframe(findings,hide_index=True,width='stretch')
    st.caption('Illustrative review cutoffs: selection/TPR gaps ≥8pp = Moderate, ≥15pp = High; historical agreement below 80% = Moderate, below 70% or AUC below 0.70 = High. Not legal thresholds.')
    st.subheader('Recommended actions')
    for function,action in GOVERNANCE_ACTIONS:
        with st.expander(f'{function}  •  {action.split(",")[0]}',expanded=function=='MEASURE'):
            st.write(action)
    st.divider()
    st.subheader('Export assessment')
    st.write('Generate a formatted report containing model metrics, all group rates, the governance register, recommended actions, and assessment limitations.')
    pdf = generate_audit_pdf(overall,group_metrics,findings,source_label=data_label)
    st.download_button('↓ Download governance audit (PDF)',data=pdf,
                       file_name='FairSight_AI_Governance_Audit.pdf',mime='application/pdf',
                       type='primary',width='stretch')
    with st.expander('Governance methodology and references'):
        st.markdown('''
        **GOVERN, MAP, MEASURE, MANAGE** are the four functions of the
        [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework).
        These review priorities are project-defined heuristics, not requirements of NIST or the EEOC.

        Employment decisions also require domain-specific review. See
        [EEOC guidance on selection procedures](https://www.eeoc.gov/laws/guidance/employment-tests-and-selection-procedures).
        ''')
    with st.expander('Preview input records'):
        st.dataframe(raw_data.head(20),hide_index=True,width='stretch')

st.markdown('<div style="color:#8193A4;font-size:.76rem;padding-top:26px">'
            'FAIRSIGHT AI  •  EDUCATIONAL MODEL GOVERNANCE PROTOTYPE  •  SYNTHETIC DEMO AVAILABLE'
            '</div>',unsafe_allow_html=True)
