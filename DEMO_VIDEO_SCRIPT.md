# Holmes AI v2.0 - Demo Video Script

**Duration:** 8-10 minutes
**Date:** November 23, 2025
**Purpose:** Showcase production-ready transaction categorization system

---

## Pre-Recording Checklist

### 1. Environment Setup
- [ ] Clean desktop (close unnecessary windows)
- [ ] Open required applications:
  - [ ] Browser (for dashboards)
  - [ ] VS Code (for code walkthrough)
  - [ ] Terminal/PowerShell (for demos)
  - [ ] File Explorer (for file navigation)
- [ ] Test microphone and screen recording
- [ ] Prepare sample transactions for live demo
- [ ] Ensure models are loaded in `models/` directory

### 2. Files to Have Open
- [ ] `README.md` (project overview)
- [ ] `FINAL_SUBMISSION.md` (comprehensive documentation)
- [ ] `architecture_dashboard.html` (workflow visualization)
- [ ] `results_dashboard.html` (results showcase)
- [ ] `frontend/index.html` (web UI)
- [ ] `src/config/taxonomy.json` (for editing demo)

### 3. Terminal Commands Ready
```bash
# Navigate to project
cd "c:\Users\Pranav Mv\Documents\Holmes_Cloe"

# Activate environment (if needed)
# python -m venv venv
# .\venv\Scripts\activate

# Test demo script
python demo.py

# Run sample predictions
python -c "from src.models.lightgbm_classifier import LightGBMClassifier; print('Models loaded successfully!')"
```

---

## Video Structure

### **Section 1: Introduction (1 minute)**
**What to Show:** Title slide or project README

**Script:**
> "Hello! Today I'm excited to demonstrate **Holmes AI v2.0**, a production-ready financial transaction categorization engine that eliminates the need for expensive third-party APIs.
>
> Traditional transaction categorization APIs cost between 1 to 5 cents per transaction, resulting in monthly costs of $200,000 to $1 million for high-volume applications. They also introduce 100-500 millisecond latency and vendor lock-in.
>
> Holmes AI solves these problems by providing:
> - **97-99% accuracy** across all category levels
> - **10 millisecond latency** - that's 19 times faster than the 200ms target
> - **Less than $100** deployment cost with zero per-transaction fees
> - **Admin-configurable taxonomy** without code changes
> - **Complete transparency** with SHAP-based explainability
>
> Let me show you how it works."

**Actions:**
1. Show README.md with project overview
2. Highlight key statistics (99.6% accuracy, 10.22ms latency, <$100 cost)

---

### **Section 2: Architecture Visualization (1.5 minutes)**
**What to Show:** `architecture_dashboard.html`

**Script:**
> "First, let's visualize the system architecture. Holmes AI uses a 6-stage inference pipeline combining semantic understanding with structured machine learning.
>
> [Open architecture dashboard]
>
> Watch as a transaction flows through the system:"

**Actions:**
1. Open `architecture_dashboard.html` in browser
2. Click **"Inference Flow"** tab
3. Click **"Run Transaction"** button
4. Point out each stage as data flows:
   - **Data Ingestion:** Schema validation with Pydantic
   - **Token Enrichment:** Text cleaning + 5 engineered features
   - **Semantic Vector:** 768-dimensional BERT embeddings
   - **Classification:** LightGBM hierarchical prediction
   - **Taxonomy Mapping:** JSON-based category lookup
   - **Explainability:** SHAP feature importance

**Script (continued):**
> "Notice the Live Inspector on the right showing real-time data transformations, and the system logs tracking each operation.
>
> The training pipeline [switch to Training Pipeline tab] shows how we train models on 100,000 synthetic transactions using GPU acceleration, achieving these exceptional results in just 79 minutes."

**Actions:**
1. Switch to **"Training Pipeline"** tab
2. Click **"Run Training"** button
3. Show progress bar and training logs

---

### **Section 3: Live Prediction Demo (2 minutes)**
**What to Show:** `frontend/index.html` or `demo.py`

**Script:**
> "Now let's see the system in action with real predictions.
>
> [Open web UI or run demo script]
>
> I'll categorize some sample transactions:"

**Sample Transactions to Demo:**

1. **Coffee Shop (High Confidence)**
   ```
   Merchant: STARBUCKS #4532
   Amount: $5.25
   Date: 2025-01-15
   ```
   **Expected:** Dining → Coffee Shops → Starbucks (95%+ confidence)

2. **Gas Station (Medium Amount)**
   ```
   Merchant: SHELL GAS STATION
   Amount: $45.00
   Date: 2025-01-14
   ```
   **Expected:** Transportation → Gas Stations → Shell (92%+ confidence)

3. **Ambiguous Transaction (Lower Confidence)**
   ```
   Merchant: AMAZON MARKETPLACE
   Amount: $23.45
   Date: 2025-01-13
   ```
   **Expected:** Shopping → Online Retail → Amazon (85%+ confidence)

**Actions:**
1. Enter each transaction in web UI or watch demo.py output
2. Show predictions with confidence scores
3. Highlight hierarchical categorization (L1 → L2 → L3)
4. Point out latency (should be ~10-15ms per transaction)

**Script (continued):**
> "Notice how fast these predictions are - we're seeing 10-15 millisecond response times. The system also provides hierarchical categories from broad (Dining) to specific (Starbucks), along with confidence scores so users know when to review predictions."

---

### **Section 4: Admin Taxonomy Editing (1.5 minutes)**
**What to Show:** `src/config/taxonomy.json` in VS Code

**Script:**
> "One of Holmes AI's key innovations is the admin-configurable taxonomy. Business users can add categories without touching code.
>
> [Open taxonomy.json]
>
> Let me show you how easy it is to add a new coffee chain - let's say **Peet's Coffee**."

**Actions:**
1. Open `src/config/taxonomy.json` in VS Code
2. Navigate to `Dining → Coffee Shops → L3 categories`
3. Add new entry:
   ```json
   {
     "id": "peets_coffee",
     "name": "Peet's Coffee",
     "aliases": ["PEETS", "PEET'S COFFEE", "PEETS COFFEE & TEA"],
     "mcc_codes": [5812, 5814]
   }
   ```
4. Save the file

**Script (continued):**
> "I've just added Peet's Coffee with merchant name aliases and MCC codes. No retraining required - the system will immediately recognize transactions from Peet's Coffee using semantic similarity to other coffee shops.
>
> For optimal accuracy, we'd collect 500-1000 labeled transactions and retrain quarterly. But this instant approach works great for adding subcategories within existing domains.
>
> The system supports unlimited categories - currently 15 top-level, 42 mid-level, and 59 leaf categories, but it can scale to thousands."

---

### **Section 5: Explainability (1.5 minutes)**
**What to Show:** SHAP analysis (demo.py or explainability.py output)

**Script:**
> "Transparency is critical for AI systems. Holmes AI uses SHAP analysis to explain every prediction.
>
> [Run explainability demo]
>
> Let me explain why that Starbucks transaction was categorized with 95% confidence."

**Actions:**
1. Run explainability demo:
   ```bash
   python explainability.py --mode single --merchant "STARBUCKS #4532" --amount 5.25
   ```
   OR show pre-generated explanation from demo.py

2. Point out explanation components:
   - Natural language reasoning
   - Top contributing features
   - Confidence breakdown (Model 70%, MCC 20%, Hierarchy 10%)
   - SHAP feature importance

**Script (continued):**
> "The system tells us the prediction was high confidence because:
> - The merchant name has strong semantic similarity to known Starbucks patterns
> - The spending amount ($5.25) falls in the typical 'micro' range for coffee shops
> - The MCC code matches food service establishments
> - Daily transaction frequency is common for coffee purchases
>
> This explainability builds user trust and helps identify when the model needs more training data."

---

### **Section 6: Results & Performance (1.5 minutes)**
**What to Show:** `results_dashboard.html`

**Script:**
> "Let's look at the comprehensive evaluation results.
>
> [Open results dashboard]
>
> Holmes AI was trained on 100,000 synthetic transactions and tested on 10,000 held-out samples."

**Actions:**
1. Open `results_dashboard.html` in browser
2. Scroll through each section, highlighting:

   **Hero Stats:**
   - L1 Accuracy: 99.64% (+9.64% above target)
   - L2 Accuracy: 98.48% (+8.48% above target)
   - L3 Accuracy: 97.53% (+7.53% above target)
   - Latency: 10.2ms (19.5x faster than target)

   **Accuracy Breakdown:**
   - Macro F1 scores: L1: 0.996, L2: 0.979, L3: 0.973
   - All exceed the 0.90 target

   **Performance Benchmarks:**
   - Average latency: 10.22ms
   - P95 latency: 11.27ms
   - P99 latency: 12.06ms
   - Throughput: 486 transactions/second

   **Training Configuration:**
   - 100K training samples
   - 79.2 minutes on Tesla T4 GPU
   - 768D embeddings + 5 engineered features

   **Bias Analysis:**
   - L1: No categories below 0.90 F1 (perfect fairness)
   - L2: 1 category below 0.80 (Charitable - Donations)
   - L3: 11 categories below 0.90 (mostly low-frequency)

**Script (continued):**
> "These results demonstrate production-ready performance. The system handles transactions 19 times faster than the target, with accuracy exceeding industry standards, and comprehensive bias analysis ensures fairness across categories."

---

### **Section 7: Technology & Innovation (1 minute)**
**What to Show:** Architecture diagram or code walkthrough

**Script:**
> "Under the hood, Holmes AI uses several innovative techniques:
>
> [Show code or architecture]
>
> **Hybrid AI Architecture:**
> - Sentence-BERT generates 768-dimensional semantic embeddings capturing merchant name meaning
> - LightGBM gradient boosting provides fast, accurate classification
> - 5 engineered features add contextual signals like spending tier and transaction frequency
>
> **Unlimited Scalability:**
> - No hard limits on category count (tested: 116 classes, theoretical: 10,000+)
> - Two expansion modes: instant via JSON, or optimal with retraining
> - Supports any number of L1, L2, L3 categories
>
> **Privacy-First Design:**
> - 100% on-premise inference - financial data never leaves your infrastructure
> - Zero external API calls
> - GDPR, CCPA, PCI-DSS compliant by design
>
> **Cost Efficiency:**
> - Less than $100 deployment cost
> - Zero per-transaction fees
> - 99.4% cost savings vs external APIs at scale"

**Actions:**
1. Show `src/` directory structure
2. Briefly show key files:
   - `models/lightgbm_classifier.py`
   - `models/sentence_bert_encoder.py`
   - `config/taxonomy.json`

---

### **Section 8: Business Impact (30 seconds)**
**What to Show:** Cost comparison or ROI slide

**Script:**
> "The business impact is substantial:
>
> **Cost Savings:**
> - At 1 million transactions per month, Holmes AI saves $119,000 per year compared to external APIs
> - Break-even after just 1 month
>
> **Performance:**
> - 19.5 times faster latency improves user experience
> - 486 transactions per second throughput scales to 20 million per month with just 10 instances
>
> **Developer Empowerment:**
> - Non-technical admins can modify categories via JSON
> - Complete control over categorization logic
> - No vendor lock-in or API rate limits"

**Actions:**
1. Show cost comparison table from FINAL_SUBMISSION.md
2. Highlight ROI metrics

---

### **Section 9: Conclusion & Next Steps (30 seconds)**
**What to Show:** Project summary

**Script:**
> "Holmes AI v2.0 demonstrates that you can achieve enterprise-grade transaction categorization without expensive APIs.
>
> **Key Takeaways:**
> ✅ 97-99% accuracy across all category levels
> ✅ 10ms latency - 19x faster than target
> ✅ Less than $100 deployment cost
> ✅ Admin-configurable, unlimited categories
> ✅ Full transparency with SHAP explainability
> ✅ Privacy-first, on-premise design
>
> The system is production-ready with comprehensive evaluation, bias analysis, and documentation.
>
> **Thank you for watching!** All code, documentation, and dashboards are available in the repository."

**Actions:**
1. Show final slide with GitHub repository link
2. Show FINAL_SUBMISSION.md overview

---

## Recording Tips

### Audio Quality
- **Use a good microphone** (headset or USB mic)
- **Record in a quiet environment**
- **Speak clearly and at moderate pace**
- **Rehearse script 2-3 times before recording**

### Video Quality
- **Resolution:** 1920x1080 (1080p minimum)
- **Frame Rate:** 30 fps
- **Screen Recording Tools:**
  - Windows: OBS Studio (free), Camtasia, ShareX
  - Mac: QuickTime, ScreenFlow, OBS Studio
- **Zoom Level:** Ensure text is readable (125-150% zoom in browser/VS Code)

### Recording Best Practices
1. **Pause between sections** (easier to edit)
2. **If you make a mistake, pause, then repeat the sentence** (don't start over)
3. **Mouse movements should be deliberate and slow**
4. **Highlight key text** with cursor or annotations
5. **Hide desktop icons** and taskbar notifications
6. **Use full-screen mode** for dashboards and browser

### Editing (Optional)
- **Add title slides** between sections
- **Speed up boring parts** (file navigation, loading)
- **Add background music** (low volume, non-distracting)
- **Add text overlays** for key statistics
- **Export in MP4 format** (H.264 codec, AAC audio)

---

## Quick Start Commands (Copy-Paste Ready)

### Terminal Demo
```bash
# Navigate to project
cd "c:\Users\Pranav Mv\Documents\Holmes_Cloe"

# Run interactive demo
python demo.py

# Run explainability demo
python explainability.py --mode single --merchant "STARBUCKS #4532" --amount 5.25

# Show evaluation results
cat evaluation_results/EVALUATION_REPORT.md

# Show bias analysis
cat bias_analysis/BIAS_ANALYSIS_REPORT.md
```

### Open Dashboards
```bash
# Architecture dashboard
start architecture_dashboard.html

# Results dashboard
start results_dashboard.html

# Frontend UI
start frontend/index.html
```

### Show Files in VS Code
```bash
# Open project in VS Code
code .

# Key files to show:
# - src/config/taxonomy.json (editable taxonomy)
# - FINAL_SUBMISSION.md (comprehensive docs)
# - README.md (quick overview)
```

---

## Sample Transaction Data (For Live Demo)

```csv
merchant,amount,date,mcc_code
"STARBUCKS #4532",5.25,2025-01-15,5812
"SHELL GAS STATION",45.00,2025-01-14,5541
"AMAZON MARKETPLACE",23.45,2025-01-13,5968
"WALMART SUPERCENTER",87.32,2025-01-12,5411
"NETFLIX.COM",15.99,2025-01-11,4899
"CHIPOTLE #2345",12.50,2025-01-10,5814
"CVS PHARMACY",18.67,2025-01-09,5912
"UBER TRIP",23.45,2025-01-08,4121
"MCDONALD'S",7.89,2025-01-07,5814
"TARGET STORE",56.78,2025-01-06,5310
```

---

## Post-Recording Checklist

- [ ] Review entire video for errors
- [ ] Check audio levels (no clipping, consistent volume)
- [ ] Verify all dashboards/demos worked correctly
- [ ] Add intro/outro slides (optional)
- [ ] Export in 1080p MP4 format
- [ ] Upload to YouTube/drive with appropriate title:
  - **Title:** "Holmes AI v2.0 - Production-Ready Transaction Categorization (97-99% Accuracy, 10ms Latency)"
  - **Description:** Include GitHub link, key statistics, tech stack
  - **Tags:** machine learning, AI, fintech, transaction categorization, LightGBM, BERT, explainability

---

## Backup Plan (If Something Breaks)

### If demo.py crashes:
- Use pre-recorded screenshots from `results_dashboard.html`
- Show evaluation reports in markdown instead

### If web UI doesn't load:
- Use terminal-based predictions instead
- Show curl commands to API endpoints

### If models don't load:
- Show pre-generated predictions from `demo_output.txt`
- Focus on architecture and documentation

---

## Video Length Guide

| Section | Duration | Must-Have? |
|---------|----------|------------|
| Introduction | 1:00 | ✅ Yes |
| Architecture Visualization | 1:30 | ✅ Yes |
| Live Prediction Demo | 2:00 | ✅ Yes |
| Taxonomy Editing | 1:30 | ⚠️ Optional |
| Explainability | 1:30 | ✅ Yes |
| Results & Performance | 1:30 | ✅ Yes |
| Technology & Innovation | 1:00 | ⚠️ Optional |
| Business Impact | 0:30 | ✅ Yes |
| Conclusion | 0:30 | ✅ Yes |
| **Total** | **8-10 min** | |

**Minimum viable demo:** Sections 1, 2, 3, 5, 6, 8, 9 = **7 minutes**

---

**Good luck with your recording!** 🎥

Remember: **Show, don't tell.** Let the dashboards, live predictions, and results speak for themselves. Your enthusiasm and clarity will make the demo engaging!

---

**Generated:** November 23, 2025
**Status:** ✅ Ready for Recording
