# Holmes AI Frontend Dashboard

Interactive web dashboard for the Holmes AI Transaction Categorization Engine.

## Features

### 📊 **Dashboard**
- Real-time system metrics (accuracy, latency, confidence)
- Visual confidence distribution chart
- Category distribution analytics
- Recent transactions list

### 🎯 **Transaction Categorization**
- **Single Transaction Form**: Test individual transactions
- **Batch Upload**: Process CSV files (multiple transactions)
- Real-time categorization results with:
  - Hierarchical categories (L1 → L2 → L3)
  - Confidence scores with color-coded badges
  - Processing time metrics
  - Review flags for low-confidence predictions

### 🌳 **Taxonomy Browser**
- Complete 15 L1 category hierarchy
- 45+ L3 categories
- Interactive exploration of category structure
- Aliases and MCC codes

### 📈 **Performance Metrics**
- Model accuracy (L1, L2, L3)
- F1 Score tracking
- Latency percentiles (P95, P99)
- Confidence breakdown
- Processing timeline visualization

## Quick Start

### 1. Start the Holmes AI API

```bash
cd ..
uvicorn src.api.main:app --reload
```

The API should be running at `http://localhost:8000`

### 2. Open the Dashboard

Simply open `index.html` in your browser:

```bash
# Windows
start index.html

# Mac
open index.html

# Linux
xdg-open index.html
```

Or use a simple HTTP server:

```bash
# Python
python -m http.server 8080

# Node.js (if you have http-server installed)
npx http-server -p 8080
```

Then visit: `http://localhost:8080`

### 3. Configure API Endpoint

If your API is running on a different port, edit `app.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000';  // Change this if needed
```

## Usage Examples

### Categorize a Single Transaction

1. Go to the **Categorize** tab
2. Fill in the transaction details:
   - Transaction ID: `TXN_001`
   - Merchant: `SWIGGY*FOOD DELIVERY`
   - Amount: `25.50`
   - Currency: `USD`
   - Channel: `online`
   - MCC Code: `5814` (optional)
3. Click **Categorize Transaction**
4. View the result with:
   - Category hierarchy (L1 → L2 → L3)
   - Confidence score
   - Processing time

### Example Transactions to Test

```
Merchant: AMZN MKTP US*2A3B4C5D6
Amount: 49.99
Expected: Shopping → Shopping - Online → Shopping - Online - Amazon

Merchant: STARBUCKS #12345
Amount: 5.50
Expected: Dining → Dining - Coffee Shops → Dining - Coffee Shops - Starbucks

Merchant: UBER *TRIP
Amount: 15.00
Expected: Travel → Travel - Local → Travel - Local - Uber

Merchant: NETFLIX.COM
Amount: 15.99
Expected: Entertainment → Entertainment - Streaming → Entertainment - Streaming - Netflix
```

## Features Breakdown

### Dashboard Section
- **System Overview**: Key performance metrics at a glance
- **Confidence Distribution**: Pie chart showing high/medium/low confidence split
- **Category Distribution**: Bar chart of L1 category usage
- **Recent Transactions**: Last 10 categorized transactions

### Categorize Section
- **Single Transaction Form**: Interactive form for testing
- **Real-time Results**: Instant categorization with detailed breakdown
- **Batch Upload**: CSV file processing (coming soon)

### Taxonomy Section
- **Complete Hierarchy**: All 15 L1 categories
- **Expandable Categories**: Click to explore L2 and L3 levels
- **Category Metadata**: View aliases and MCC codes

### Metrics Section
- **Model Performance**: L1/L2/L3 accuracy and F1 scores
- **System Performance**: Latency metrics (avg, P95, P99)
- **Confidence Analysis**: Breakdown by confidence level
- **Timeline Chart**: Processing time over time

## API Integration

The dashboard communicates with the Holmes AI API:

### Endpoints Used

```javascript
GET  /api/v1/stats      // System statistics
GET  /api/v1/taxonomy   // Category hierarchy
POST /api/v1/categorize // Transaction categorization
```

### CORS Configuration

If you encounter CORS issues, the API has been configured to accept requests from all origins during development. For production, update the CORS settings in `src/api/main.py`.

## Customization

### Change Colors

Edit `styles.css`:

```css
:root {
    --primary: #6366f1;      /* Main color */
    --success: #10b981;      /* High confidence */
    --warning: #f59e0b;      /* Medium confidence */
    --danger: #ef4444;       /* Low confidence */
}
```

### Modify Charts

Charts use Chart.js. Edit `app.js` to customize:

```javascript
// Example: Change chart type
confidenceChart = new Chart(ctx, {
    type: 'doughnut',  // Change to 'bar', 'line', 'pie', etc.
    // ...
});
```

### Add New Metrics

1. Update the HTML in `index.html`
2. Add the metric display in `app.js`
3. Fetch data from the API

## Technologies Used

- **HTML5**: Semantic markup
- **CSS3**: Modern styling with CSS Grid and Flexbox
- **Vanilla JavaScript**: No framework dependencies
- **Chart.js**: Interactive charts and visualizations
- **Font Awesome**: Icons
- **Holmes AI API**: Backend categorization engine

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Troubleshooting

### API Connection Failed

**Error**: "Failed to connect to API"

**Solution**:
1. Ensure the API is running: `uvicorn src.api.main:app --reload`
2. Check the API URL in `app.js`
3. Verify no firewall is blocking localhost:8000

### No Transactions Showing

**Cause**: Models not trained yet

**Solution**: The API returns mock predictions until models are trained. See the main README for training instructions.

### CORS Errors

**Cause**: Browser security restrictions

**Solution**: Use a proper HTTP server (see Quick Start) instead of opening the HTML file directly.

## Development

### Adding New Features

1. **Add HTML**: Update `index.html` with new sections
2. **Style It**: Add CSS rules in `styles.css`
3. **Add Logic**: Implement functionality in `app.js`
4. **Test**: Open in browser and verify

### Connecting to Production API

Change the API base URL:

```javascript
const API_BASE_URL = 'https://your-production-api.com';
```

## Screenshots

### Dashboard View
![Dashboard with metrics and charts]

### Categorization Interface
![Single transaction categorization form and results]

### Taxonomy Browser
![Hierarchical category tree]

### Performance Metrics
![Detailed metrics and timeline]

## Support

For issues or questions:
- Check the main [README](../README.md)
- Review [SETUP.md](../SETUP.md)
- Contact: pranav@backbase.com, pratima@backbase.com

## License

Part of the Holmes AI project - Internal use only
