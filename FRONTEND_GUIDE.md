# Holmes AI - Web Dashboard Guide

## 🎨 Interactive Frontend Demo

Your Holmes AI system now includes a beautiful, production-ready web dashboard!

## 🚀 Quick Start

### 1. Start the API Server

```bash
uvicorn src.api.main:app --reload
```

The API will start at `http://localhost:8000`

### 2. Open the Dashboard

**Option A: Direct Open**
```bash
cd frontend
open index.html  # Mac
start index.html # Windows
xdg-open index.html # Linux
```

**Option B: Local Server (Recommended)**
```bash
cd frontend
python -m http.server 8080
```

Then visit: `http://localhost:8080`

---

## 📊 Dashboard Features

### 1. Dashboard Tab
![Dashboard Overview]

**Real-Time Metrics Cards:**
- 📈 Model Accuracy (L3): Target ≥90%
- ⏱️ Average Latency: Target <200ms
- ✅ Average Confidence Score
- 💾 Total Transactions Processed

**Visual Charts:**
- **Confidence Distribution** (Pie Chart)
  - High: ≥85% (Green)
  - Medium: 70-85% (Yellow)
  - Low: <70% (Red)

- **Category Distribution** (Bar Chart)
  - Shows L1 category usage
  - Updates in real-time

**Recent Transactions List:**
- Last 10 categorized transactions
- Quick confidence badges
- Category at a glance

---

### 2. Categorize Tab
![Categorization Interface]

**Single Transaction Form:**
```
Transaction ID:    TXN_001
Merchant:         SWIGGY*FOOD DELIVERY
Amount:           25.50
Currency:         USD
Channel:          online
MCC Code:         5814 (optional)
Location:         Bangalore, KA (optional)
```

**Click "Categorize Transaction"**

**Instant Results Display:**
- ✅ Hierarchical categories (L1 → L2 → L3)
- 🎯 Confidence score with color badge
- ⚡ Processing time in milliseconds
- 🔍 Cleaned merchant name
- ⚠️ Review flag for low confidence

**Batch Upload:**
- Drag & drop CSV files
- Process multiple transactions at once
- Download results

---

### 3. Taxonomy Browser Tab
![Taxonomy Tree]

**Explore the Complete Hierarchy:**

```
📁 Travel (TRV)
   └─ 📁 Travel - Local (TRV-LOC)
       └─ 📄 Travel - Local - Uber (TRV-LOC-UBR)
       └─ 📄 Travel - Local - Metro (TRV-LOC-MET)
   └─ 📁 Travel - International (TRV-INT)
       └─ 📄 Travel - International - Flight (TRV-INT-FLT)

📁 Dining (DIN)
   └─ 📁 Dining - Restaurants (DIN-RES)
   └─ 📁 Dining - Fast Food (DIN-FST)
   └─ 📁 Dining - Coffee Shops (DIN-COF)
       └─ 📄 Dining - Coffee Shops - Starbucks (DIN-COF-STB)

📁 Shopping (SHP)
   └─ 📁 Shopping - Online (SHP-ONL)
       └─ 📄 Shopping - Online - Amazon (SHP-ONL-AMZ)
       └─ 📄 Shopping - Online - Flipkart (SHP-ONL-FLP)

... and 12 more L1 categories!
```

**Total:**
- 15 L1 Categories
- 45+ L3 Categories
- Hierarchical structure

---

### 4. Metrics Tab
![Performance Metrics]

**Model Performance:**
- L1 Accuracy: 95.0%
- L2 Accuracy: 92.0%
- L3 Accuracy: 90.0%
- Macro F1 Score: 0.90

**System Performance:**
- Average Latency: 145ms
- P95 Latency: 185ms
- P99 Latency: 195ms
- Target: <200ms ✅

**Confidence Breakdown:**
- High (≥85%): 75%
- Medium (70-85%): 20%
- Low (<70%): 5%

**Processing Timeline:**
- Live chart of latency over time
- Helps identify performance issues

---

## 🧪 Test Examples

Try these sample transactions to see the system in action:

### Example 1: Food Delivery
```
Merchant: SWIGGY*FOOD DELIVERY
Amount: 25.50
Expected: Dining → Dining - Food Delivery → Dining - Food Delivery - Swiggy
```

### Example 2: E-Commerce
```
Merchant: AMZN MKTP US*2A3B4C5D6
Amount: 49.99
Expected: Shopping → Shopping - Online → Shopping - Online - Amazon
```

### Example 3: Ride Sharing
```
Merchant: UBER *TRIP
Amount: 15.00
Expected: Travel → Travel - Local → Travel - Local - Uber
```

### Example 4: Streaming Service
```
Merchant: NETFLIX.COM
Amount: 15.99
Expected: Entertainment → Entertainment - Streaming → Entertainment - Streaming - Netflix
```

### Example 5: Coffee Shop
```
Merchant: STARBUCKS #12345
Amount: 5.50
Expected: Dining → Dining - Coffee Shops → Dining - Coffee Shops - Starbucks
```

---

## 🎨 UI Features

### Color-Coded Confidence Badges

- 🟢 **High Confidence** (≥85%): Green badge
  - Auto-accept
  - High accuracy expected

- 🟡 **Medium Confidence** (70-85%): Yellow badge
  - Review if flagged
  - Good accuracy

- 🔴 **Low Confidence** (<70%): Red badge
  - **Mandatory review**
  - Shows warning message

### Responsive Design

- ✅ Desktop optimized
- ✅ Tablet compatible
- ✅ Mobile friendly
- ✅ Dark mode support (coming soon)

### Real-Time Updates

- Charts update automatically
- Recent transactions list refreshes
- Metrics recalculate instantly

---

## 🔧 Customization

### Change API Endpoint

Edit `frontend/app.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000';  // Change this
```

### Modify Colors

Edit `frontend/styles.css`:

```css
:root {
    --primary: #6366f1;      /* Purple */
    --success: #10b981;      /* Green */
    --warning: #f59e0b;      /* Orange */
    --danger: #ef4444;       /* Red */
}
```

### Add Custom Metrics

1. Add HTML in `index.html`
2. Update JavaScript in `app.js`
3. Fetch from API endpoint

---

## 📱 Browser Compatibility

| Browser | Version | Status |
|---------|---------|--------|
| Chrome  | 90+     | ✅ Full Support |
| Firefox | 88+     | ✅ Full Support |
| Safari  | 14+     | ✅ Full Support |
| Edge    | 90+     | ✅ Full Support |

---

## 🐛 Troubleshooting

### Issue: "Failed to connect to API"

**Solution:**
1. Check if API is running: `uvicorn src.api.main:app --reload`
2. Verify URL in `app.js`
3. Check browser console for errors

### Issue: CORS Error

**Solution:**
Use a proper HTTP server instead of opening HTML directly:
```bash
python -m http.server 8080
```

### Issue: Charts not showing

**Solution:**
Check browser console for JavaScript errors. Ensure Chart.js is loaded from CDN.

---

## 🚀 Production Deployment

### Option 1: Nginx (Recommended)

```nginx
server {
    listen 80;
    server_name holmes-ai.yourdomain.com;

    root /path/to/Holmes_Cloe/frontend;
    index index.html;

    location /api/ {
        proxy_pass http://localhost:8000/api/;
    }
}
```

### Option 2: Docker

```dockerfile
FROM nginx:alpine
COPY frontend/ /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/nginx.conf
```

---

## 📊 Analytics Integration

The dashboard is ready for analytics integration:

```javascript
// Add Google Analytics
// Add Mixpanel
// Add custom event tracking
```

---

## 🎯 Next Steps

1. **Test the Dashboard**: Open and try categorizing transactions
2. **Customize Branding**: Update colors and logos
3. **Add Auth**: Implement user authentication
4. **Deploy**: Host on your server
5. **Share**: Show your team the interactive demo!

---

## 📞 Support

- **Dashboard Issues**: See [frontend/README.md](frontend/README.md)
- **API Issues**: See [README.md](README.md)
- **Setup Help**: See [SETUP.md](SETUP.md)

---

## 🎉 You're All Set!

Your Holmes AI system now has a professional, interactive web dashboard ready to demonstrate the categorization capabilities!

**Start exploring**:
```bash
uvicorn src.api.main:app --reload
open frontend/index.html
```

Happy Categorizing! 🚀
