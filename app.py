"""
ComplaintIQ — AI-Powered Customer Complaint Intelligence System
Final production version. Requires:
  - model__1_.pkl        (LogisticRegression, class_weight='balanced')
  - vectorizer__1_.pkl   (TfidfVectorizer, max_features=5000, ngram_range=(1,2))
  - ecommerce_complaints_lstm_ready.csv
  - GEMINI_API_KEY in .env
"""

import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai
import os
import io
import json
import warnings
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ComplaintIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:      #0d0f14;
    --surface: #141720;
    --border:  #252a36;
    --accent:  #e8ff47;
    --accent2: #ff6b4a;
    --text:    #e8ecf4;
    --muted:   #6b7280;
    --danger:  #ff4a6b;
    --success: #47ffa3;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

h1,h2,h3,h4 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
}
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 12px 20px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}
.stTextArea textarea, .stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(232,255,71,0.15) !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    border: none !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 10px 24px !important;
    border-radius: 4px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #d4eb3a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(232,255,71,0.3) !important;
}
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
}
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-accent  { border-left: 3px solid var(--accent);  }
.card-danger  { border-left: 3px solid var(--danger);  }
.card-success { border-left: 3px solid var(--success); }

.tag { display:inline-block; padding:3px 10px; border-radius:4px;
       font-size:0.72rem; font-weight:500; letter-spacing:0.05em; text-transform:uppercase; }
.tag-critical { background:rgba(255,74,107,0.15);  color:#ff4a6b; border:1px solid rgba(255,74,107,0.3);  }
.tag-high     { background:rgba(255,107,74,0.15);  color:#ff6b4a; border:1px solid rgba(255,107,74,0.3);  }
.tag-medium   { background:rgba(232,255,71,0.15);  color:#e8ff47; border:1px solid rgba(232,255,71,0.3);  }
.tag-low      { background:rgba(71,255,163,0.15);  color:#47ffa3; border:1px solid rgba(71,255,163,0.3);  }

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}
div[data-testid="stSelectbox"] > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# ENV / API
# ─────────────────────────────────────────────────────────────────
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

# ─────────────────────────────────────────────────────────────────
# DATA & MODEL  (exact filenames from your project)
# ─────────────────────────────────────────────────────────────────
CSV_FILE        = "final_data.csv"
MODEL_FILE      = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

# Real categories from model.classes_ / label encoder
CATEGORIES = [
    "Account Issue",
    "Customer Service",
    "Delivery Issue",
    "Payment Problem",
    "Product Quality",
    "Refund Issue",
]

# Risk & insight mapped to exact category strings (no snake_case needed)
INSIGHT_MAP = {
    "Account Issue":    ("Account access failures reducing platform stickiness",  "Deploy self-serve recovery; reduce OTP failure rate",               "Medium"),
    "Customer Service": ("Support experience below customer expectation",          "Reduce first-response time with AI triage; train agents on CSAT",   "Medium"),
    "Delivery Issue":   ("Logistics failures degrading last-mile experience",      "Audit carrier SLAs; trigger auto-escalation on delay > 3 days",     "High"),
    "Payment Problem":  ("Payment failures blocking purchase completion",          "Stabilise gateway integrations; add fallback PSP routing",          "Critical"),
    "Product Quality":  ("Quality defects entering fulfilment pipeline",           "Enforce pre-shipment QA gates; flag repeat-SKU complaints",         "Critical"),
    "Refund Issue":     ("Refund friction eroding customer trust and LTV",         "Automate refund approvals under threshold; alert agents on high-val","High"),
}
RISK_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}

def get_insight(category: str):
    """Returns (meaning, action, risk) for a category string."""
    return INSIGHT_MAP.get(
        category,
        ("Unclassified complaint requiring manual review",
         "Escalate to relevant department",
         "Low")
    )

# ─────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

@st.cache_resource
def load_model():
    try:
        m = joblib.load(MODEL_FILE)
        v = joblib.load(VECTORIZER_FILE)
        return m, v, True
    except Exception:
        return None, None, False

df           = load_data()
model, vectorizer, model_ready = load_model()

# ─────────────────────────────────────────────────────────────────
# PLOTLY SHARED THEME
# ─────────────────────────────────────────────────────────────────
_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#6b7280", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
)
_AX = dict(gridcolor="#1e2330", linecolor="#252a36", tickcolor="#252a36")

# ─────────────────────────────────────────────────────────────────
# SENTIMENT (TextBlob)
# ─────────────────────────────────────────────────────────────────
def analyze_sentiment(text: str) -> dict:
    blob        = TextBlob(text)
    polarity    = blob.sentiment.polarity
    subjectivity= blob.sentiment.subjectivity
    if   polarity >=  0.2: label, col = "Positive", "#47ffa3"
    elif polarity <= -0.2: label, col = "Negative", "#ff4a6b"
    else:                  label, col = "Neutral",  "#e8ff47"
    intensity = "High" if abs(polarity) > 0.5 else ("Medium" if abs(polarity) > 0.2 else "Low")
    return {
        "polarity":     round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
        "label":        label,
        "color":        col,
        "intensity":    intensity,
        "bar_width":    int((polarity + 1) / 2 * 100),
    }

# ─────────────────────────────────────────────────────────────────
# GEMINI HELPERS
# ─────────────────────────────────────────────────────────────────
def _llm():
    return genai.GenerativeModel("gemini-3-flash-preview")

def generate_complaint_insight(complaint: str, category: str, sentiment: dict) -> str:
    prompt = f"""
You are a senior Customer Experience Intelligence Analyst.

COMPLAINT TEXT:
{complaint}

PREDICTED CATEGORY: {category}
SENTIMENT: {sentiment['label']} (polarity={sentiment['polarity']}, intensity={sentiment['intensity']})

Produce a structured intelligence report with EXACTLY these sections:

## Business Impact
One concise paragraph on operational and revenue impact.

## Root Cause Hypothesis
2-3 likely root causes as bullet points.

## Risk Assessment
Risk Level: [Low / Medium / High / Critical]
Justification: one sentence.

## Recommended Actions
Numbered list of 3 specific, actionable steps for the business.

## Customer Emotion Summary
One sentence describing the customer's emotional state.

Keep language sharp, professional, and executive-ready.
"""
    return _llm().generate_content(prompt).text


def generate_executive_summary(data: pd.DataFrame) -> str:
    sample = data.sample(min(60, len(data)), random_state=42).to_csv(index=False)
    prompt = f"""
You are the Chief Customer Officer preparing a board-level briefing.

DATASET SAMPLE (CSV):
{sample}

Produce a structured executive summary with EXACTLY these sections:

## Executive Overview
2-3 sentence summary of the current complaint landscape.

## Top Issues & Trends
Bullet list of the 3-5 most significant patterns.

## Risk Hotspots
Identify the highest-risk categories with brief justification.

## Operational Metrics Insight
Commentary on resolution times and high-value customer exposure.

## Strategic Recommendations
3 concrete recommendations for leadership action in the next 30 days.

## Outlook
One forward-looking paragraph.

Write in precise, data-informed executive prose. No filler.
"""
    return _llm().generate_content(prompt).text


def generate_batch_results(texts: list) -> list:
    joined = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    prompt = f"""
You are an AI complaint triage specialist.

Analyse the following {len(texts)} complaints and return a JSON array.
Each object must have keys:
  "index"          : 1-based integer
  "category"       : one of {CATEGORIES}
  "risk"           : one of Low / Medium / High / Critical
  "one_line_summary": short string

COMPLAINTS:
{joined}

Return ONLY valid JSON array. No markdown, no explanation, no code fences.
"""
    raw = _llm().generate_content(prompt).text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ─────────────────────────────────────────────────────────────────
# PDF REPORT BUILDER
# ─────────────────────────────────────────────────────────────────
def build_pdf_report(data: pd.DataFrame, summary_text: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)

    title_s = ParagraphStyle("T",  fontName="Helvetica-Bold",   fontSize=20, leading=26,
                              textColor=colors.HexColor("#0d0f14"), spaceAfter=6)
    h2_s    = ParagraphStyle("H2", fontName="Helvetica-Bold",   fontSize=13, leading=18,
                              textColor=colors.HexColor("#0d0f14"), spaceBefore=14, spaceAfter=6)
    body_s  = ParagraphStyle("B",  fontName="Helvetica",        fontSize=9,  leading=14,
                              textColor=colors.HexColor("#1a1a2e"))
    meta_s  = ParagraphStyle("M",  fontName="Helvetica-Oblique",fontSize=8,
                              textColor=colors.HexColor("#555555"), spaceAfter=10)
    kpi_lbl = ParagraphStyle("KL", fontName="Helvetica",        fontSize=7,
                              textColor=colors.HexColor("#555555"), alignment=TA_CENTER)
    kpi_val = ParagraphStyle("KV", fontName="Helvetica-Bold",   fontSize=15,
                              textColor=colors.HexColor("#0d0f14"), alignment=TA_CENTER)

    total    = len(data)
    avg_res  = round(data["resolution_time"].mean(), 1)
    top_cat  = data["category"].value_counts().idxmax()
    high_val = int((data["customer_value"] > 25000).sum())
    neg_pct  = round(data["sentiment"].eq("Negative").mean() * 100, 1)
    repeat   = int(data["is_repeat_customer"].sum())

    content = []
    content.append(Paragraph("ComplaintIQ Intelligence Report", title_s))
    content.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}  |  Records analysed: {total:,}",
        meta_s))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0d0f14")))
    content.append(Spacer(1, 14))

    # KPI row
    kpi_data = [
        [Paragraph("TOTAL", kpi_lbl), Paragraph("AVG RESOLUTION", kpi_lbl),
         Paragraph("TOP CATEGORY", kpi_lbl), Paragraph("HIGH-VALUE", kpi_lbl),
         Paragraph("NEG SENTIMENT", kpi_lbl), Paragraph("REPEAT CUST.", kpi_lbl)],
        [Paragraph(f"{total:,}", kpi_val), Paragraph(f"{avg_res}d", kpi_val),
         Paragraph(top_cat, kpi_val), Paragraph(f"{high_val:,}", kpi_val),
         Paragraph(f"{neg_pct}%", kpi_val), Paragraph(f"{repeat:,}", kpi_val)],
    ]
    kpi_tbl = Table(kpi_data, colWidths=[1.1*inch]*6)
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#f5f5f5")),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [colors.HexColor("#e8ff47"), colors.white]),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#dddddd")),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
    ]))
    content.append(kpi_tbl)
    content.append(Spacer(1, 18))

    # Category table
    content.append(Paragraph("Category Breakdown", h2_s))
    content.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    content.append(Spacer(1, 6))
    cat_df = data["category"].value_counts().reset_index()
    cat_df.columns = ["Category", "Count"]
    cat_df["% Share"] = (cat_df["Count"] / total * 100).round(1).astype(str) + "%"
    cat_df["Avg Res (days)"] = cat_df["Category"].map(
        lambda c: round(data[data["category"] == c]["resolution_time"].mean(), 1))
    cat_df["Risk"] = cat_df["Category"].map(lambda c: get_insight(c)[2])

    tbl_data = [["Category", "Count", "% Share", "Avg Res", "Risk"]] + \
               cat_df[["Category","Count","% Share","Avg Res (days)","Risk"]].values.tolist()
    tbl = Table(tbl_data, colWidths=[2.0*inch, 0.7*inch, 0.7*inch, 0.8*inch, 0.8*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#0d0f14")),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#dddddd")),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    content.append(tbl)
    content.append(Spacer(1, 18))

    # AI Summary
    content.append(Paragraph("AI Executive Summary", h2_s))
    content.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    content.append(Spacer(1, 6))
    for line in summary_text.split("\n"):
        line = line.strip()
        if not line:
            content.append(Spacer(1, 4))
        elif line.startswith("## "):
            content.append(Paragraph(line[3:], h2_s))
        else:
            safe = line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            content.append(Paragraph(safe, body_s))

    content.append(Spacer(1, 14))
    content.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    content.append(Paragraph("Confidential — ComplaintIQ Auto-Generated Report", meta_s))
    doc.build(content)
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────────
# SIDEBAR  (filters applied to entire app)
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">ComplaintIQ</p>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Customer  \nComplaint Intelligence**")
    st.markdown("---")

    st.markdown('<p class="section-header">Filters</p>', unsafe_allow_html=True)

    cat_options = ["All"] + sorted(df["category"].dropna().unique().tolist())
    sel_cat = st.selectbox("Category", cat_options)

    channel_options = ["All"] + sorted(df["channel"].dropna().unique().tolist())
    sel_channel = st.selectbox("Channel", channel_options)

    priority_options = ["All"] + sorted(df["priority"].dropna().unique().tolist())
    sel_priority = st.selectbox("Priority", priority_options)

    if df["date"].notna().any():
        min_d = df["date"].min().date()
        max_d = df["date"].max().date()
        date_range = st.date_input("Date Range", value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d)
    else:
        date_range = None

    st.markdown("---")
    st.markdown('<p class="section-header">System Status</p>', unsafe_allow_html=True)
    st.markdown("🟢 **ML Model** — Active" if model_ready else "🟡 **ML Model** — Demo Mode")
    st.markdown("🟢 **Gemini API** — Connected")
    st.markdown(f"📦 **Records** — {len(df):,}")
    st.markdown(f"🏷️ **Categories** — {df['category'].nunique()}")

# ── Apply filters
fdf = df.copy()
if sel_cat      != "All": fdf = fdf[fdf["category"] == sel_cat]
if sel_channel  != "All": fdf = fdf[fdf["channel"]  == sel_channel]
if sel_priority != "All": fdf = fdf[fdf["priority"] == sel_priority]
if date_range and len(date_range) == 2:
    fdf = fdf[(fdf["date"].dt.date >= date_range[0]) &
              (fdf["date"].dt.date <= date_range[1])]

# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:28px 0 20px 0;">
  <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
              letter-spacing:-0.03em;color:#e8ecf4;">
    ComplaintIQ <span style="color:#e8ff47;">Intelligence</span>
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:0.8rem;color:#6b7280;margin-top:4px;">
    AI-Powered Customer Complaint Analysis Platform
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧠  AI Analyzer",
    "📊  Dashboard",
    "📋  Batch Analysis",
    "📝  Executive Summary",
    "📄  Export Report",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — AI ANALYZER
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Single Complaint Analysis</p>',
                unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        user_input = st.text_area(
            "Complaint Text",
            placeholder="Paste or type the customer complaint here...",
            height=160,
            label_visibility="collapsed",
        )
        analyze_btn = st.button("⚡  Analyse Complaint", use_container_width=True)

    if analyze_btn:
        if not user_input.strip():
            st.warning("Please enter a complaint.")
        else:
            # Sentiment
            sent = analyze_sentiment(user_input)

            # Classification
            if model_ready:
                vec_input = vectorizer.transform([user_input])
                prediction = model.predict(vec_input)[0]   # e.g. "Refund Issue"
            else:
                prediction = "Refund Issue"

            meaning, action, risk = get_insight(prediction)
            risk_cls = f"tag-{risk.lower()}"

            with col_in:
                st.markdown(f"""
                <div class="card card-accent" style="margin-top:16px;">
                  <div class="section-header">Classification</div>
                  <div style="font-family:'Syne',sans-serif;font-size:1.3rem;
                              font-weight:700;margin-bottom:8px;">{prediction}</div>
                  <span class="tag {risk_cls}">{risk} Risk</span>
                  <div style="margin-top:12px;font-size:0.8rem;color:#6b7280;">
                    <b style="color:#e8ecf4;">Impact:</b> {meaning}
                  </div>
                  <div style="margin-top:6px;font-size:0.8rem;color:#6b7280;">
                    <b style="color:#e8ecf4;">Action:</b> {action}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Sentiment card
                pol_pct = sent["bar_width"]
                st.markdown(f"""
                <div class="card" style="margin-top:0;">
                  <div class="section-header">Sentiment Analysis</div>
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-family:'Syne',sans-serif;font-size:1.1rem;
                                 font-weight:700;color:{sent['color']};">{sent['label']}</span>
                    <span style="font-size:0.75rem;color:#6b7280;">
                      Polarity {sent['polarity']} &nbsp;|&nbsp; Subjectivity {sent['subjectivity']}
                    </span>
                  </div>
                  <div style="background:#252a36;border-radius:3px;height:6px;
                              margin-top:10px;overflow:hidden;">
                    <div style="width:{pol_pct}%;background:{sent['color']};height:6px;
                                border-radius:3px;"></div>
                  </div>
                  <div style="display:flex;justify-content:space-between;margin-top:4px;">
                    <span style="font-size:0.65rem;color:#6b7280;">Negative</span>
                    <span style="font-size:0.65rem;color:#6b7280;">Positive</span>
                  </div>
                  <div style="margin-top:8px;font-size:0.75rem;color:#6b7280;">
                    Intensity: <span style="color:{sent['color']};font-weight:600;">
                    {sent['intensity']}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with col_out:
                with st.spinner("Generating AI intelligence report..."):
                    try:
                        insight = generate_complaint_insight(user_input, prediction, sent)
                        st.markdown('<p class="section-header">AI Intelligence Report</p>',
                                    unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="card card-accent" style="font-size:0.85rem;line-height:1.8;">
                          {insight.replace(chr(10), "<br>")}
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Gemini Error: {e}")

# ══════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Executive Dashboard</p>',
                unsafe_allow_html=True)

    if len(fdf) == 0:
        st.warning("No data matches the current filters.")
    else:
        total    = len(fdf)
        avg_res  = round(fdf["resolution_time"].mean(), 1)
        top_cat  = fdf["category"].value_counts().idxmax()
        high_val = int((fdf["customer_value"] > 25000).sum())
        delayed  = int((fdf["resolution_time"] > fdf["resolution_time"].mean()).sum())
        repeat   = int(fdf["is_repeat_customer"].sum())
        neg_pct  = round(fdf["sentiment"].eq("Negative").mean() * 100, 1)
        high_pri = int(fdf["priority"].eq("High").sum())

        # Row 1 — 8 KPIs
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Complaints",  f"{total:,}")
        c2.metric("Avg Resolution",    f"{avg_res} days")
        c3.metric("High Priority",     f"{high_pri:,}")
        c4.metric("Negative Sentiment",f"{neg_pct}%")

        c5,c6,c7,c8 = st.columns(4)
        c5.metric("Top Category",      top_cat)
        c6.metric("High-Value Cases",  f"{high_val:,}")
        c7.metric("Delayed Cases",     f"{delayed:,}")
        c8.metric("Repeat Customers",  f"{repeat:,}")

        st.markdown("")

        # ── Row 2: Weekly trend | Category distribution
        r1a, r1b = st.columns(2, gap="large")

        with r1a:
            st.markdown('<p class="section-header">Weekly Complaint Volume</p>',
                        unsafe_allow_html=True)
            weekly = (fdf.resample("W", on="date").size()
                        .reset_index(name="count"))
            spikes = weekly[weekly["count"].pct_change() > 0.5]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=weekly["date"], y=weekly["count"],
                mode="lines+markers",
                line=dict(color="#e8ff47", width=2),
                marker=dict(size=5, color="#e8ff47"),
                fill="tozeroy", fillcolor="rgba(232,255,71,0.06)",
                name="Volume",
            ))
            if not spikes.empty:
                fig.add_trace(go.Scatter(
                    x=spikes["date"], y=spikes["count"],
                    mode="markers",
                    marker=dict(size=12, color="#ff4a6b", symbol="x"),
                    name="Spike",
                ))
            fig.update_layout(**_BASE, height=240, showlegend=True,
                              xaxis=_AX, yaxis=_AX,
                              legend=dict(bgcolor="rgba(0,0,0,0)",
                                          font=dict(color="#6b7280", size=10)))
            st.plotly_chart(fig, use_container_width=True)
            if not spikes.empty:
                spike_dates = ", ".join(spikes["date"].dt.strftime("%d %b %Y").tolist())
                st.markdown(
                    f'<div class="card card-danger" style="padding:10px 16px;'
                    f'font-size:0.8rem;">⚠️ Spike weeks: {spike_dates}</div>',
                    unsafe_allow_html=True)

        with r1b:
            st.markdown('<p class="section-header">Category Distribution</p>',
                        unsafe_allow_html=True)
            cat_cnt = fdf["category"].value_counts().reset_index()
            cat_cnt.columns = ["category", "count"]
            fig2 = px.bar(cat_cnt, x="count", y="category", orientation="h",
                          color="count",
                          color_continuous_scale=["#252a36", "#e8ff47"])
            fig2.update_layout(**_BASE, height=240, coloraxis_showscale=False,
                               xaxis=_AX,
                               yaxis=dict(**_AX, tickfont=dict(size=10)))
            st.plotly_chart(fig2, use_container_width=True)

        # ── Row 3: Sentiment by category | Priority by channel
        r2a, r2b = st.columns(2, gap="large")

        with r2a:
            st.markdown('<p class="section-header">Sentiment Breakdown by Category</p>',
                        unsafe_allow_html=True)
            sent_d = (fdf.groupby(["category", "sentiment"])
                        .size().reset_index(name="count"))
            fig3 = px.bar(sent_d, x="category", y="count", color="sentiment",
                          color_discrete_map={"Negative":"#ff4a6b",
                                              "Neutral":"#e8ff47",
                                              "Positive":"#47ffa3"},
                          barmode="stack")
            fig3.update_layout(**_BASE, height=240,
                               xaxis=dict(**_AX, tickangle=-20), yaxis=_AX,
                               legend=dict(bgcolor="rgba(0,0,0,0)",
                                           font=dict(color="#6b7280", size=10)))
            st.plotly_chart(fig3, use_container_width=True)

        with r2b:
            st.markdown('<p class="section-header">Priority Distribution by Channel</p>',
                        unsafe_allow_html=True)
            ch_d = (fdf.groupby(["channel", "priority"])
                      .size().reset_index(name="count"))
            fig4 = px.bar(ch_d, x="channel", y="count", color="priority",
                          color_discrete_map={"High":"#ff4a6b",
                                              "Medium":"#e8ff47",
                                              "Low":"#47ffa3"},
                          barmode="group")
            fig4.update_layout(**_BASE, height=240, xaxis=_AX, yaxis=_AX,
                               legend=dict(bgcolor="rgba(0,0,0,0)",
                                           font=dict(color="#6b7280", size=10)))
            st.plotly_chart(fig4, use_container_width=True)

        # ── Row 4: Avg resolution | Customer value histogram
        r3a, r3b = st.columns(2, gap="large")

        with r3a:
            st.markdown('<p class="section-header">Avg Resolution Time by Category</p>',
                        unsafe_allow_html=True)
            res_d = (fdf.groupby("category")["resolution_time"]
                       .mean().reset_index(name="avg_res"))
            global_avg = res_d["avg_res"].mean()
            res_d["bar_color"] = res_d["avg_res"].apply(
                lambda x: "#ff4a6b" if x > global_avg * 1.1 else "#e8ff47")
            fig5 = px.bar(res_d, x="category", y="avg_res",
                          color="bar_color", color_discrete_map="identity")
            fig5.add_hline(y=global_avg, line_dash="dash", line_color="#6b7280",
                           annotation_text="Avg", annotation_font_color="#6b7280")
            fig5.update_layout(**_BASE, height=240, showlegend=False,
                               xaxis=dict(**_AX, tickangle=-20), yaxis=_AX)
            st.plotly_chart(fig5, use_container_width=True)

        with r3b:
            st.markdown('<p class="section-header">Customer Value Distribution</p>',
                        unsafe_allow_html=True)
            fig6 = px.histogram(fdf, x="customer_value", nbins=40,
                                color_discrete_sequence=["#e8ff47"])
            fig6.add_vline(x=25000, line_dash="dash", line_color="#ff4a6b",
                           annotation_text="High-Value Threshold",
                           annotation_font_color="#ff4a6b")
            fig6.update_layout(**_BASE, height=240, showlegend=False,
                               xaxis=_AX, yaxis=_AX)
            st.plotly_chart(fig6, use_container_width=True)

        # ── Row 5: Repeat customer heatmap (category × channel)
        st.markdown('<p class="section-header">Repeat Customer Rate — Category × Channel</p>',
                    unsafe_allow_html=True)
        heat = (fdf.groupby(["category", "channel"])["is_repeat_customer"]
                  .mean().reset_index())
        heat_pivot = heat.pivot(index="category", columns="channel",
                                values="is_repeat_customer")
        fig7 = px.imshow(heat_pivot,
                         color_continuous_scale=["#0d0f14", "#252a36", "#e8ff47"],
                         text_auto=".0%", aspect="auto")
        fig7.update_layout(**_BASE, height=240, coloraxis_showscale=False,
                           xaxis=_AX, yaxis=_AX)
        st.plotly_chart(fig7, use_container_width=True)

        # ── Category risk overview table
        st.markdown('<p class="section-header">Category Risk Overview</p>',
                    unsafe_allow_html=True)
        rows = []
        for cat, grp in fdf.groupby("category"):
            _, _, risk = get_insight(cat)
            rows.append({
                "Category":         cat,
                "Count":            len(grp),
                "Avg Resolution":   round(grp["resolution_time"].mean(), 1),
                "High-Value Cases": int((grp["customer_value"] > 25000).sum()),
                "Neg Sentiment %":  f"{round(grp['sentiment'].eq('Negative').mean()*100,1)}%",
                "High Priority %":  f"{round(grp['priority'].eq('High').mean()*100,1)}%",
                "Repeat Cust %":    f"{round(grp['is_repeat_customer'].mean()*100,1)}%",
                "Risk":             risk,
            })
        risk_df = (pd.DataFrame(rows)
                     .sort_values("Risk",
                                  key=lambda s: s.map(RISK_ORDER),
                                  ascending=False))

        def _risk_color(v):
            c = {"Critical":"#ff4a6b","High":"#ff6b4a",
                 "Medium":"#e8ff47","Low":"#47ffa3"}.get(v, "#6b7280")
            return f"color:{c};font-weight:600"

        st.dataframe(
            risk_df.style.applymap(_risk_color, subset=["Risk"]),
            use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Batch Complaint Analysis</p>',
                unsafe_allow_html=True)

    mode = st.radio("Input Method", ["Paste Complaints", "Upload CSV"],
                    horizontal=True, label_visibility="collapsed")

    complaints_list = []

    if mode == "Paste Complaints":
        batch_text = st.text_area(
            "Complaints (one per line)",
            placeholder="Complaint 1\nComplaint 2\nComplaint 3...",
            height=180,
            label_visibility="collapsed",
        )
        if batch_text.strip():
            complaints_list = [c.strip() for c in batch_text.splitlines()
                               if c.strip()]
    else:
        uploaded = st.file_uploader(
            "Upload CSV", type=["csv"],
            help="Must contain a column named: complaint_text, complaint, text, or description")
        if uploaded:
            udf = pd.read_csv(uploaded)
            col_name = next(
                (c for c in ["complaint_text", "complaint", "text", "description"]
                 if c in udf.columns), None)
            if col_name:
                complaints_list = udf[col_name].dropna().astype(str).tolist()
                st.success(f"✅  Loaded {len(complaints_list):,} complaints "
                           f"from column `{col_name}`")
            else:
                st.error(f"No recognised column found. "
                         f"Got: {list(udf.columns)}")

    if complaints_list:
        st.markdown(f"**{len(complaints_list):,} complaints** ready for analysis.")
        if st.button("⚡  Run Batch Analysis", use_container_width=True):
            with st.spinner(f"Analysing {len(complaints_list)} complaints…"):
                try:
                    results = generate_batch_results(complaints_list)
                    res_df = pd.DataFrame(results)

                    # Add TextBlob sentiment per complaint
                    res_df["sentiment"] = [
                        analyze_sentiment(c)["label"]
                        for c in complaints_list[:len(res_df)]
                    ]

                    st.markdown('<p class="section-header">Results</p>',
                                unsafe_allow_html=True)

                    def _col_risk(v):
                        c = {"Critical":"#ff4a6b","High":"#ff6b4a",
                             "Medium":"#e8ff47","Low":"#47ffa3"}.get(v,"#6b7280")
                        return f"color:{c};font-weight:600"

                    def _col_sent(v):
                        c = {"Negative":"#ff4a6b","Neutral":"#e8ff47",
                             "Positive":"#47ffa3"}.get(v,"#6b7280")
                        return f"color:{c};font-weight:600"

                    st.dataframe(
                        res_df.style
                              .applymap(_col_risk, subset=["risk"])
                              .applymap(_col_sent, subset=["sentiment"]),
                        use_container_width=True, hide_index=True)

                    # Risk pie chart
                    if "risk" in res_df.columns:
                        rc = res_df["risk"].value_counts().reset_index()
                        rc.columns = ["risk","count"]
                        fig_pie = px.pie(
                            rc, names="risk", values="count",
                            color="risk",
                            color_discrete_map={"Critical":"#ff4a6b","High":"#ff6b4a",
                                                "Medium":"#e8ff47","Low":"#47ffa3"})
                        fig_pie.update_layout(**_BASE, height=260,
                                              xaxis=_AX, yaxis=_AX)
                        st.plotly_chart(fig_pie, use_container_width=True)

                    # CSV download
                    st.download_button(
                        "⬇️  Download Results CSV",
                        res_df.to_csv(index=False).encode(),
                        "batch_results.csv", "text/csv")

                except json.JSONDecodeError:
                    st.error("Could not parse Gemini response as JSON. "
                             "Try with fewer complaints (≤ 30).")
                except Exception as e:
                    st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════
# TAB 4 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">AI Executive Summary</p>',
                unsafe_allow_html=True)

    c_left, c_right = st.columns([1, 2], gap="large")

    with c_left:
        date_min = fdf["date"].min().strftime("%d %b %Y") if fdf["date"].notna().any() else "N/A"
        date_max = fdf["date"].max().strftime("%d %b %Y") if fdf["date"].notna().any() else "N/A"
        st.markdown(f"""
        <div class="card">
          <div class="section-header">Dataset Snapshot</div>
          <div style="font-size:0.85rem;line-height:2.1;">
            📦 <b>Records:</b> {len(fdf):,}<br>
            🗓️ <b>Date Range:</b> {date_min} – {date_max}<br>
            🏷️ <b>Categories:</b> {fdf["category"].nunique()}<br>
            ⏱️ <b>Avg Resolution:</b> {round(fdf["resolution_time"].mean(),1)} days<br>
            💰 <b>High-Value Cases:</b> {int((fdf["customer_value"]>25000).sum()):,}<br>
            😠 <b>Negative Sentiment:</b> {round(fdf["sentiment"].eq("Negative").mean()*100,1)}%<br>
            🔁 <b>Repeat Customers:</b> {int(fdf["is_repeat_customer"].sum()):,}<br>
            🔴 <b>High Priority:</b> {int(fdf["priority"].eq("High").sum()):,}
          </div>
        </div>
        """, unsafe_allow_html=True)
        gen_btn = st.button("🧠  Generate Summary", use_container_width=True)

    with c_right:
        if gen_btn:
            with st.spinner("Generating executive summary…"):
                try:
                    summary = generate_executive_summary(fdf)
                    st.session_state["exec_summary"] = summary
                except Exception as e:
                    st.error(f"Gemini Error: {e}")

        if "exec_summary" in st.session_state:
            st.markdown('<p class="section-header">Summary</p>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="card card-accent" style="font-size:0.85rem;line-height:1.8;">
              {st.session_state["exec_summary"].replace(chr(10), "<br>")}
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 5 — EXPORT REPORT
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-header">Export PDF Intelligence Report</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="font-size:0.85rem;line-height:2;">
      📄 The PDF report includes:<br>
      &nbsp;&nbsp;&nbsp;• Executive KPI summary (6 metrics)<br>
      &nbsp;&nbsp;&nbsp;• Category breakdown with risk ratings<br>
      &nbsp;&nbsp;&nbsp;• AI-generated executive summary<br>
      &nbsp;&nbsp;&nbsp;• Timestamps and metadata<br>
      <br>
      💡 Generate an <b>Executive Summary</b> in Tab 4 first for the richest report.
    </div>
    """, unsafe_allow_html=True)

    summary_for_pdf = st.session_state.get(
        "exec_summary",
        "No AI summary generated yet. Visit the Executive Summary tab first.")

    if st.button("📥  Generate & Download PDF Report", use_container_width=True):
        with st.spinner("Building PDF…"):
            try:
                pdf_bytes = build_pdf_report(fdf, summary_for_pdf)
                fname = f"ComplaintIQ_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.download_button(
                    label="⬇️  Download PDF",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.success(f"✅  Report ready: {fname}")
            except Exception as e:
                st.error(f"PDF generation error: {e}")