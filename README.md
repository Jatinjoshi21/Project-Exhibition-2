# 🧠 ComplaintIQ — AI-Powered Customer Complaint Intelligence System

> Transform raw customer complaints into structured, actionable business intelligence using Machine Learning + Google Gemini LLM.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-1.5_Flash-4285F4?style=flat&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📌 Overview

**ComplaintIQ** is an end-to-end complaint intelligence platform built for the **Project Exhibition 2024-25**. It goes beyond traditional complaint classification by combining a trained ML model with an LLM reasoning layer to deliver real business intelligence — not just category labels.

| What traditional tools do | What ComplaintIQ does |
|---|---|
| Classify complaints | Classify + explain + risk-rate |
| Show complaint counts | Generate executive intelligence reports |
| Treat all customers equally | Flag high-value customer exposure |
| Require manual analysis | Auto-generate board-ready summaries & PDFs |

---

## 🚀 Features

### 🧠 Tab 1 — AI Analyzer
- Classifies a single complaint into one of **6 categories** using the trained ML model
- Computes **sentiment** (polarity, subjectivity, intensity) via TextBlob
- Generates a **5-section AI intelligence report** via Gemini 1.5 Flash:
  - Business Impact · Root Cause Hypothesis · Risk Assessment · Recommended Actions · Customer Emotion Summary

### 📊 Tab 2 — Executive Dashboard
- **8 live KPI metrics** — Total Complaints, Avg Resolution, High Priority, Negative Sentiment %, Top Category, High-Value Cases, Delayed Cases, Repeat Customers
- **Weekly trend chart** with automatic spike detection (>50% WoW change flagged in red)
- **Sentiment breakdown** by category (stacked bar)
- **Priority distribution** by channel (grouped bar)
- **Resolution time** by category with global average reference line
- **Customer value histogram** with ₹25,000 high-value threshold marker
- **Repeat customer heatmap** — category × channel
- **Risk overview table** with Neg Sentiment %, High Priority %, Repeat Customer % per category
- All charts respond to **sidebar filters** (category, channel, priority, date range)

### 📋 Tab 3 — Batch Analysis
- Paste complaints (one per line) **or** upload a CSV file
- Gemini analyses all complaints in a single call → returns category + risk + one-line summary per complaint
- Independent TextBlob sentiment per complaint
- Colour-coded results table + risk distribution pie chart
- **One-click CSV download** of all results

### 📝 Tab 4 — Executive Summary
- Gemini generates a **6-section board-level briefing** from the full filtered dataset:
  - Executive Overview · Top Issues & Trends · Risk Hotspots · Operational Metrics · Strategic Recommendations · Outlook
- Fully responsive to sidebar filters

### 📄 Tab 5 — Export Report
- One-click **professional PDF report** generation (ReportLab)
- Includes: KPI table, category breakdown, risk ratings, full AI executive summary
- Auto-timestamped filename

---

## 📁 Project Structure

```
ComplaintIQ/
│
├── app.py                                    # Main Streamlit application (~1,000 lines)
├── model__1_.pkl                             # Trained LogisticRegression model
├── vectorizer__1_.pkl                        # Fitted TfidfVectorizer
├── ecommerce_complaints_lstm_ready.csv       # 25,000-record complaint dataset
├── .env                                      # API key config (create this yourself)
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

---

## 🗃️ Dataset

**File:** `ecommerce_complaints_lstm_ready.csv`  
**Records:** 25,000 | **Columns:** 10

| Column | Type | Description |
|---|---|---|
| `complaint_id` | Integer | Unique record identifier |
| `date` | Date | Complaint filing date |
| `complaint_text` | String | Raw complaint text — ML input feature |
| `category` | String | Target label (6 classes) |
| `sentiment` | String | Negative / Neutral / Positive |
| `priority` | String | High / Medium / Low |
| `channel` | String | App / Call / Website / Email |
| `customer_value` | Integer | Customer lifetime value (₹502 – ₹49,999) |
| `resolution_time` | Integer | Days to resolve (12 – 119, mean: 65.5) |
| `is_repeat_customer` | Integer | 1 if repeat complainant, else 0 |

**Category distribution (near-balanced):**

| Category | Count | Risk |
|---|---|---|
| Payment Problem | 4,215 | 🔴 Critical |
| Refund Issue | 4,201 | 🟠 High |
| Customer Service | 4,180 | 🔵 Medium |
| Product Quality | 4,161 | 🔴 Critical |
| Account Issue | 4,147 | 🔵 Medium |
| Delivery Issue | 4,096 | 🟠 High |

---

## 🤖 ML Model

**Algorithm:** Logistic Regression (`class_weight='balanced'`, `max_iter=1000`)  
**Vectorizer:** TF-IDF (`max_features=5000`, `ngram_range=(1,2)`, `stop_words='english'`)  
**Train/Test Split:** 80/20 stratified (20,000 train / 5,000 test)

**Test Set Performance:**

| Category | Precision | Recall | F1-Score |
|---|---|---|---|
| Account Issue | 0.78 | 0.88 | 0.83 |
| Customer Service | 0.80 | 0.84 | 0.82 |
| Delivery Issue | 0.78 | 0.84 | 0.81 |
| Payment Problem | 0.83 | 0.84 | 0.83 |
| Product Quality | 0.90 | 0.82 | 0.86 |
| Refund Issue | 0.96 | 0.80 | 0.88 |
| **Overall** | — | — | **0.84** |

> A deep learning model (Embedding + GlobalAveragePooling1D) was also prototyped (83.5% accuracy) but Logistic Regression was selected for production due to faster inference, simpler deployment, and equivalent accuracy.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit ≥ 1.35 | Web app, tabs, widgets, metrics |
| ML | scikit-learn ≥ 1.4 | TF-IDF vectorisation + classification |
| LLM | Google Gemini 1.5 Flash | Intelligence reports & executive summaries |
| Sentiment | TextBlob ≥ 0.18 | Polarity & subjectivity scoring |
| Data | Pandas ≥ 2.0 | DataFrame ops, filtering, aggregations |
| Charts | Plotly ≥ 5.20 | Interactive visualisations |
| PDF | ReportLab ≥ 4.1 | PDF report generation |
| Model I/O | Joblib ≥ 1.3 | Model & vectorizer serialisation |
| Config | python-dotenv ≥ 1.0 | Environment variable loading |

---

## ⚙️ Setup & Installation

### 1. Clone / Download the project
```bash
git clone https://github.com/your-username/complaintiq.git
cd complaintiq
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download TextBlob corpora
```bash
python -m textblob.download_corpora
```

### 4. Get a Gemini API key
Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) and create a free API key.

### 5. Create your `.env` file
```
GEMINI_API_KEY=your_api_key_here
```

### 6. Run the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📦 requirements.txt

```
streamlit>=1.35.0
pandas>=2.0.0
joblib>=1.3.0
google-generativeai>=0.5.0
python-dotenv>=1.0.0
plotly>=5.20.0
textblob>=0.18.0
reportlab>=4.1.0
scikit-learn>=1.4.0
```

---

## 🏗️ Architecture

```
Customer Complaint (text input)
        │
        ▼
┌───────────────────────┐
│   TF-IDF Vectorizer   │  ← fitted on training data only
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Logistic Regression  │  → Category (e.g. "Refund Issue")
│       Classifier      │  → Risk Level (Critical/High/Medium/Low)
└───────────┬───────────┘
            │
            ├─────────────────────────────────┐
            ▼                                 ▼
┌───────────────────────┐       ┌─────────────────────────┐
│   TextBlob Sentiment  │       │   Google Gemini 1.5 Flash│
│  polarity/subjectivity│  ───► │   (prompt = complaint +  │
└───────────────────────┘       │    category + sentiment) │
                                └────────────┬────────────┘
                                             │
                                             ▼
                                ┌─────────────────────────┐
                                │   Intelligence Report    │
                                │  • Business Impact       │
                                │  • Root Cause Hypothesis │
                                │  • Risk Assessment       │
                                │  • Recommended Actions   │
                                │  • Customer Emotion      │
                                └─────────────────────────┘
```

---

## 📊 Key Business Insights (from dataset analysis)

- **70% of complaints are Negative** — indicating systemic issues, not isolated incidents
- **Average resolution time: 65.5 days** — critically high; no SLA enforcement evident
- **~50% of complaints come from high-value customers (>₹25,000)** who face the most Critical-risk issues (Payment Problem, Product Quality)
- **Delivery Issue and Account Issue** show the highest repeat complaint rates — root causes are not being permanently resolved
- **All 4 channels receive equal complaint volume** — no single channel is disproportionately stressed

---

## 🔮 Future Enhancements

- [ ] Replace TextBlob with a fine-tuned BERT sentiment model for complaint-domain text
- [ ] CRM integration (Salesforce / Freshdesk) to push insights directly into ticket systems
- [ ] Real-time complaint ingestion via WebSocket or Kafka consumer
- [ ] Email / Slack alerts for Critical-risk complaint spikes
- [ ] Forecasting module to predict complaint volume surges using time-series data
- [ ] Multi-language support via translation preprocessing
- [ ] User authentication for team-based access control

---

> *ComplaintIQ bridges the gap between raw data and business intelligence — turning every complaint into a decision.*
