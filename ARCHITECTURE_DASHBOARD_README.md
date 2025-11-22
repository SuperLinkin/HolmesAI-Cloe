# Holmes AI v2.0 - Interactive Architecture Dashboard

## Overview

An interactive, animated visualization of the Holmes AI architecture showcasing both **Inference Flow** and **Training Pipeline** workflows.

**File:** [architecture_dashboard.html](architecture_dashboard.html)

---

## Features

### 🎨 **Interactive Visualization**
- **Dual Tabs:** Switch between Inference Flow and Training Pipeline
- **Animated Workflows:** Watch data flow through each stage
- **Live Inspector:** Real-time state viewer showing data transformations
- **System Logs:** Console output tracking each operation

### 📊 **Production Metrics Display**
- **L1 Accuracy:** 99.6%
- **L1 Macro F1:** 0.996
- **Average Latency:** 10.2ms
- **Throughput:** 486 txns/sec

### 🔍 **Stage Details**
Click any stage card to view detailed technical specifications:
- System architecture
- Performance metrics
- Implementation details
- Technology stack

---

## How to Use

### **Method 1: Direct Browser Opening**
```bash
# Open in default browser
start architecture_dashboard.html

# Or double-click the file in Windows Explorer
```

### **Method 2: Local Web Server** (Recommended for demos)
```bash
# Python 3
python -m http.server 8000

# Then navigate to:
# http://localhost:8000/architecture_dashboard.html
```

### **Method 3: Live Server (VS Code)**
1. Install "Live Server" extension in VS Code
2. Right-click `architecture_dashboard.html`
3. Select "Open with Live Server"

---

## Workflow Tabs

### 🔵 **Tab 1: Inference Flow** (Real-time Categorization)

**6-Stage Pipeline:**

1. **Data Ingestion**
   - Schema validation (Pydantic)
   - Multi-source normalization (CSV, JSON, ERP)
   - Volume: 20M transactions/month

2. **Token Enrichment**
   - Text cleaning & normalization
   - Contextual feature injection
   - Impact: +8% F1 improvement

3. **Semantic Vector**
   - Model: all-mpnet-base-v2
   - Output: 768D dense embeddings
   - Device: Tesla T4 GPU

4. **Classification**
   - Algorithm: LightGBM
   - Latency: <50ms per transaction
   - Hierarchical output (L1/L2/L3)

5. **Taxonomy Mapping**
   - JSON-based configuration
   - No-code category management
   - Hierarchical validation

6. **Feedback Loop**
   - Human-in-the-loop review
   - Nightly retraining
   - Continuous improvement: 3-5% per quarter

**Interactive Features:**
- Click "Run Transaction" to see animated data flow
- Watch the data packet move through stages
- Live inspector shows transformations at each step
- System logs track operations in real-time

---

### 🟡 **Tab 2: Training Pipeline** (Model Development)

**6-Stage Pipeline:**

1. **Dataset Generation**
   - 100,000 synthetic transactions
   - 15 L1 categories
   - Realistic distributions

2. **GPU Embedding**
   - Device: Tesla T4 (Google Colab Pro)
   - Batch size: 64
   - Processing time: ~2 minutes

3. **Feature Engineering**
   - 768D semantic vectors
   - +5 engineered features
   - Total: 773D feature matrix

4. **Train L1 Model**
   - 500 boosting rounds
   - Metric: Multi-class logloss
   - Target accuracy: 90%+

5. **Train L2/L3 Models**
   - 42 L2 classes
   - 59 L3 classes
   - Hierarchical conditional training

6. **Artifact Registry**
   - Format: .txt (LightGBM) + .pkl (encoders)
   - Total size: ~150MB
   - Production-ready models

**Interactive Features:**
- Click "Run Training" to simulate training workflow
- Progress bar shows overall completion
- Training logs display in real-time
- Stage cards highlight during execution

---

## Visual Elements

### **Stage Cards**
- **Hover Effect:** Cards lift and border changes color
- **Active State:** Blue glow for inference, amber for training
- **Click Action:** Opens detailed specification drawer

### **Live Inspector** (Right Sidebar)
Shows real-time data transformations:

**Inference Mode:**
- Raw input JSON
- Enriched features
- 768D embedding visualization
- Final prediction with confidence

**Training Mode:**
- Overall progress percentage
- Progress bar animation
- Training logs (epoch, loss, metrics)

### **System Logs Console**
- Timestamped entries
- Color-coded by severity (INFO, SUCCESS, WARN)
- Auto-scroll to latest message
- Simulates real production logging

### **Detailed Drawer** (Slide-over Panel)
- Technical specifications
- System performance metrics
- Implementation details
- Close button or backdrop click to dismiss

---

## Technology Stack

### **Frontend**
- **Framework:** Vanilla JavaScript (no dependencies)
- **Styling:** Tailwind CSS (CDN)
- **Icons:** Font Awesome 6.4.0
- **Fonts:** Inter (UI) + JetBrains Mono (code)

### **Design Pattern**
- **Glassmorphism:** Frosted glass effect panels
- **Animations:** CSS transitions + JavaScript
- **Responsive:** Mobile-first grid layout
- **Accessibility:** Keyboard navigation, semantic HTML

---

## Architecture Highlights

### **Key Features Visualized:**

1. **Offline AI**
   - Full local inference
   - No API dependencies
   - Quantized models on T4 GPU

2. **JSON Taxonomy**
   - Business logic decoupled from code
   - Instant category updates
   - Admin-friendly configuration

3. **Hybrid Model**
   - Semantic Search (BERT)
   - Structured Classification (LightGBM)
   - Best of both worlds

---

## Customization Guide

### **Update Metrics:**
Edit lines 475-492 in `architecture_dashboard.html`:
```html
<div class="text-2xl font-bold text-slate-800">99.6%</div>
<div class="text-xs text-slate-500">L1 Accuracy</div>
```

### **Modify Stage Details:**
Edit the `inferenceData` or `trainingData` objects (lines 713-729):
```javascript
const inferenceData = {
    1: {
        title: "Your Stage Title",
        content: "Detailed description..."
    }
};
```

### **Change Animation Timing:**
Adjust `setTimeout` delays in `runInferenceSim()` or `runTrainingSim()` functions.

### **Add New Stages:**
1. Add HTML card in the grid (follow existing pattern)
2. Add data to `inferenceData` or `trainingData`
3. Update animation sequence in simulation functions

---

## Demo Workflow

### **For Presentations:**

1. **Open Dashboard**
   ```bash
   start architecture_dashboard.html
   ```

2. **Inference Tab Demo:**
   - Explain the 6-stage pipeline
   - Click "Run Transaction" to show animation
   - Point out the Live Inspector showing transformations
   - Highlight the 10.2ms latency in metrics

3. **Training Tab Demo:**
   - Switch to "Training Pipeline" tab
   - Explain dataset generation to artifact registry
   - Click "Run Training" to simulate workflow
   - Show progress bar and training logs

4. **Stage Details:**
   - Click any stage card to open detailed drawer
   - Show technical specifications
   - Explain system architecture

5. **Metrics Highlight:**
   - Point to Production Metrics panel:
     - 99.6% L1 Accuracy
     - 0.996 Macro F1
     - 10.2ms latency
     - 486 txns/sec throughput

---

## Use Cases

### **1. Stakeholder Presentations**
- Visual, non-technical explanation of system architecture
- Clear ROI metrics displayed prominently
- Interactive engagement through animations

### **2. Technical Documentation**
- Detailed stage specifications in drawer panels
- Technology stack clearly labeled
- System performance metrics

### **3. Demo Videos**
- Record screen with animations running
- Showcase both inference and training flows
- Professional, polished visual presentation

### **4. Team Onboarding**
- New developers understand architecture quickly
- Click-through exploration of each component
- Self-documenting with detailed descriptions

---

## Performance Notes

- **File Size:** ~40KB (single HTML file)
- **Load Time:** Instant (no external dependencies except CDNs)
- **Browser Support:** Modern browsers (Chrome, Firefox, Edge, Safari)
- **Responsive:** Works on desktop, tablet, mobile

---

## Future Enhancements (Optional)

### **Potential Additions:**
- [ ] Real API integration (connect to live backend)
- [ ] Export flow diagram as PNG/SVG
- [ ] Dark mode toggle
- [ ] Customizable color themes
- [ ] Additional metrics charts (accuracy over time, etc.)
- [ ] Zoom/pan controls for large pipelines

---

## Troubleshooting

### **Issue: Animations not playing**
- **Solution:** Ensure JavaScript is enabled in browser
- **Check:** Console for any errors (F12 → Console tab)

### **Issue: Fonts/icons not loading**
- **Solution:** Check internet connection (CDN dependencies)
- **Alternative:** Download and host Tailwind CSS + Font Awesome locally

### **Issue: Layout broken on mobile**
- **Solution:** Tailwind CSS responsive classes should handle this
- **Check:** Viewport meta tag is present in `<head>`

---

## Credits

**Generated By:** Gemini AI
**Project:** Holmes AI v2.0
**Architecture:** Production-ready transaction categorization system
**Date:** 2025-11-22

---

## Quick Reference

| Action | Result |
|--------|--------|
| Click "Run Transaction" | Animate inference flow |
| Click "Run Training" | Simulate training pipeline |
| Click any stage card | Open detailed specifications |
| Click tab buttons (top) | Switch between Inference/Training |
| Click backdrop/X button | Close detail drawer |

---

## Integration with Main Project

This dashboard complements the Holmes AI project documentation:

- **README.md** - Setup and usage instructions
- **SYSTEM_ARCHITECTURE.md** - Technical architecture details
- **FINAL_RESULTS_SUMMARY.md** - Evaluation results
- **architecture_dashboard.html** - ⭐ **Interactive visualization** ⭐

**Recommended:** Link this dashboard in your README for visual demonstrations.

---

**Status:** ✅ Production-ready interactive architecture visualization
**Purpose:** Showcase Holmes AI workflow for demos and documentation
**Maintenance:** Update metrics and content as system evolves

🤖 Enhanced by [Claude Code](https://claude.com/claude-code)
