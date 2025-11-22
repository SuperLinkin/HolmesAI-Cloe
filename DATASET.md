# Holmes AI - Synthetic Transaction Dataset Documentation

## Overview

This document describes the synthetic transaction dataset generated for training the Holmes AI Transaction Categorization Engine. The dataset is designed to simulate realistic financial transactions across various categories, merchants, and payment channels.

## Dataset Generation

### Generation Script

- **Script**: [generate_dataset.py](generate_dataset.py)
- **Validation**: [validate_dataset.py](validate_dataset.py)
- **Taxonomy Source**: [src/config/taxonomy.json](src/config/taxonomy.json)

### Features

The dataset generator creates realistic synthetic transactions with the following characteristics:

1. **Hierarchical Categorization**: 15 L1 categories, 42 L2 subcategories, 59 L3 types
2. **Realistic Merchant Names**: Pattern-based generation with variations (location codes, transaction IDs, payment processor prefixes)
3. **MCC Code Integration**: Visa/Mastercard/Amex Merchant Category Codes
4. **Temporal Patterns**: Realistic transaction timing (higher frequency on weekdays, business hours)
5. **Amount Distribution**: Log-normal distribution based on category type
6. **Multiple Currencies**: USD (50%), EUR (20%), INR (15%), GBP (10%), AUD (5%)
7. **Payment Channels**: Online, POS, ATM, Mobile

## Dataset Files

### Generated Datasets

| File | Records | Size | Purpose |
|------|---------|------|---------|
| `data/synthetic_transactions_1k.csv` | 1,000 | ~0.72 MB | Quick testing |
| `data/synthetic_transactions_10k.csv` | 10,000 | ~7.25 MB | Development |
| `data/synthetic_transactions_50k.csv` | 50,000 | ~36.25 MB | Full training |

### Training Splits (from 50k dataset)

| Split | Records | Percentage | Purpose |
|-------|---------|------------|---------|
| `data/train.csv` | 35,000 | 70% | Model training |
| `data/val.csv` | 5,000 | 10% | Hyperparameter tuning |
| `data/test.csv` | 10,000 | 20% | Final evaluation |

**Note**: Splits are stratified by L3 category to ensure balanced representation.

## Schema

### Input Features

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `transaction_id` | string | Unique transaction identifier | `TXN_20241121_000001` |
| `merchant_raw` | string | Raw merchant description | `AMZN MKTP US*2A3B4C5D6` |
| `amount` | float | Transaction amount | `49.99` |
| `currency` | string | ISO 4217 currency code | `USD` |
| `timestamp` | datetime | Transaction timestamp | `2024-11-21T14:32:00Z` |
| `channel` | string | Payment channel | `online`, `pos`, `atm`, `mobile` |
| `location` | string | Merchant location (optional) | `Seattle, WA` |
| `mcc_code` | string | Merchant Category Code (optional) | `5942` |

### Ground Truth Labels

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `l1` | string | Level 1 category | `Shopping` |
| `l1_id` | string | L1 category ID | `SHP` |
| `l2` | string | Level 2 subcategory | `Shopping - Online` |
| `l2_id` | string | L2 category ID | `SHP-ONL` |
| `l3` | string | Level 3 specific type | `Shopping - Online - Amazon` |
| `l3_id` | string | L3 category ID | `SHP-ONL-AMZ` |

## Category Distribution

### Level 1 Categories (50k dataset)

| Category | Count | Percentage |
|----------|-------|------------|
| Dining | 11,026 | 22.1% |
| Shopping | 8,191 | 16.4% |
| Transportation | 5,426 | 10.9% |
| Entertainment | 4,534 | 9.1% |
| Travel | 4,132 | 8.3% |
| Utilities | 3,595 | 7.2% |
| Technology | 2,254 | 4.5% |
| Healthcare | 2,085 | 4.2% |
| Personal Care | 2,006 | 4.0% |
| Education | 1,715 | 3.4% |
| Financial Services | 1,378 | 2.8% |
| Housing | 1,342 | 2.7% |
| Business Expenses | 936 | 1.9% |
| Charitable | 891 | 1.8% |
| Miscellaneous | 489 | 1.0% |

## Data Quality Metrics

### Completeness

- **Transaction IDs**: 100% unique, no duplicates
- **Required Fields**: 100% populated
- **MCC Codes**: ~99% coverage
- **Locations**: ~79% populated (realistic for online transactions)

### Data Integrity

✓ No duplicate transaction IDs
✓ All amounts positive and non-zero
✓ All timestamps within expected range (365 days)
✓ All categories match taxonomy
✓ No data leakage between train/val/test splits
✓ Stratified splits maintain category distribution

### Statistical Properties

**Amount Statistics (50k dataset):**
- Mean: $434.32
- Median: $114.94
- Min: $5.00
- Max: $10,000.00
- Std Dev: $982.13

**Temporal Coverage:**
- Date Range: 365 days (1 year)
- Realistic weekday/weekend patterns
- Business hours emphasis for certain categories

## Usage Examples

### Loading the Dataset

```python
import pandas as pd

# Load training data
train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/val.csv')
test_df = pd.read_csv('data/test.csv')

# Parse timestamps
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

print(f"Train size: {len(train_df)}")
print(f"L3 categories: {train_df['l3'].nunique()}")
```

### Generating Custom Dataset

```python
from generate_dataset import SyntheticDatasetGenerator
from datetime import datetime, timedelta

# Initialize generator
generator = SyntheticDatasetGenerator()

# Generate custom dataset
df = generator.generate_dataset(
    num_samples=5000,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    output_path="data/custom_dataset.csv"
)

# Create custom splits
train, val, test = generator.generate_train_test_split(
    df,
    test_size=0.2,
    val_size=0.1,
    output_dir="data"
)
```

### Validating Dataset

```bash
# Validate all generated datasets
python validate_dataset.py
```

## Category Taxonomy

### Complete Hierarchy

The dataset includes 15 L1 categories with 42 L2 subcategories and 59 L3 specific types:

1. **Travel** (TRV)
   - Local: Uber, Metro, Auto/Rickshaw
   - International: Flight, Visa
   - Accommodation: Hotel

2. **Dining** (DIN)
   - Restaurants: Fine Dining, Casual
   - Fast Food: Burgers, Pizza
   - Coffee Shops: Starbucks, Local
   - Food Delivery: Swiggy, Zomato

3. **Shopping** (SHP)
   - Retail: Clothing, Electronics
   - Online: Amazon, Flipkart
   - Groceries: Supermarket, Online

4. **Entertainment** (ENT)
   - Movies: Theater
   - Streaming: Netflix, Prime, Other
   - Events: Concerts

5. **Utilities** (UTL)
   - Electric, Water, Internet (Broadband/Mobile)

6. **Healthcare** (HLT)
   - Pharmacy, Hospital, Insurance

7. **Education** (EDU)
   - Tuition (School/College), Books, Courses

8. **Transportation** (TRN)
   - Fuel (Petrol/Diesel), Parking, Maintenance

9. **Financial Services** (FIN)
   - Banking, Investment, Insurance

10. **Housing** (HSG)
    - Rent, Mortgage, Home Improvement

11. **Personal Care** (PER)
    - Salon, Gym, Wellness

12. **Technology** (TEC)
    - Electronics (Computer/Mobile), Software, Subscriptions

13. **Charitable** (CHR)
    - Donations, Nonprofits

14. **Business Expenses** (BUS)
    - Office Supplies, Professional Services

15. **Miscellaneous** (MSC)
    - Uncategorized

## MCC Code Mapping

The dataset uses real Merchant Category Codes (MCC) from Visa/Mastercard/Amex:

| MCC Code | Category | Example |
|----------|----------|---------|
| 4111 | Local Transportation | Metro, Rail |
| 4121 | Taxi & Rideshare | Uber, Auto |
| 4511 | Airlines | Indigo, Emirates |
| 5411 | Grocery Stores | Walmart, BigBasket |
| 5651 | Clothing Stores | Zara, H&M |
| 5732 | Electronics | Best Buy, Croma |
| 5812 | Restaurants | Fine Dining |
| 5814 | Fast Food | McDonald's, KFC |
| 5942 | Book/Online Stores | Amazon |
| 5912 | Drug Stores | CVS, Apollo |
| 6010 | Financial Institutions | Banks |
| 6211 | Securities Brokers | Zerodha |
| 6300 | Insurance | Premium Payments |
| 7011 | Hotels | OYO, Airbnb |
| 7230 | Beauty/Barber Shops | Salons |
| 7523 | Parking | Parking Lots |
| 7538 | Automotive Service | Car Repair |
| 7832 | Motion Pictures | PVR, AMC |
| 7922 | Theatrical Producers | Concerts |
| 7997 | Membership Clubs | Gyms |
| 8011 | Doctors/Physicians | Hospitals |
| 8211 | Elementary Schools | School Fees |
| 8220 | Colleges | University Fees |
| 8299 | Educational Services | Online Courses |
| 8398 | Charitable Organizations | NGOs |
| 8999 | Professional Services | Consulting |

## Dataset Characteristics

### Realistic Patterns

1. **Merchant Name Variations**
   - Payment processor prefixes: `SQ *`, `PAYPAL *`, `STRIPE *`
   - Location codes: `NY`, `CA`, `MUMBAI`
   - Transaction IDs: alphanumeric codes
   - Store numbers: `#1234`

2. **Temporal Patterns**
   - Higher transaction frequency on weekdays
   - Business hours emphasis (9 AM - 8 PM)
   - Category-specific timing (e.g., dining during meal times)

3. **Amount Distributions**
   - Log-normal distribution per category
   - Realistic ranges by category type
   - Occasional high-value transactions

4. **Channel Selection**
   - Online categories → primarily `online` channel
   - Retail categories → primarily `pos` channel
   - Mixed for other categories

## Data Augmentation Ideas

For enhanced training, consider:

1. **Typos and Misspellings**: Add realistic merchant name errors
2. **Currency Conversions**: Include foreign exchange rates
3. **Recurring Transactions**: Add subscription patterns
4. **Batch Transactions**: Multiple purchases from same merchant
5. **Regional Variations**: Location-specific merchant patterns
6. **Time Series**: Seasonal spending patterns
7. **User Profiles**: Generate coherent spending patterns per user

## Validation Checklist

When generating or modifying datasets:

- [ ] All required columns present
- [ ] No duplicate transaction IDs
- [ ] All amounts positive
- [ ] Timestamps within valid range
- [ ] Categories match taxonomy
- [ ] MCC codes valid
- [ ] No data leakage between splits
- [ ] Stratified splits maintain distribution
- [ ] Realistic merchant name patterns
- [ ] Appropriate currency distribution

## Known Limitations

1. **Synthetic Nature**: Not real transaction data; patterns may differ from production
2. **Simplified Patterns**: Real-world transactions have more complex behavioral patterns
3. **Limited Noise**: Minimal typos, errors, or edge cases
4. **Category Balance**: Some categories underrepresented by design
5. **No User Context**: No user-level spending patterns or preferences
6. **Static Taxonomy**: Fixed category structure; no new categories over time

## References

- **MCC Codes**: [Visa Merchant Category Codes](https://www.visa.com/supplierlocator-app/app/#/home/supplier-locator)
- **ISO 4217**: [Currency Codes](https://www.iso.org/iso-4217-currency-codes.html)
- **Taxonomy**: Based on common personal finance categorization systems

## License

This synthetic dataset is generated for training purposes only. The dataset generation code is part of the Holmes AI project.

## Contact

For questions or issues with the dataset:
- Report issues via project issue tracker
- Review taxonomy configuration in `src/config/taxonomy.json`
- Modify generation parameters in `generate_dataset.py`

---

**Generated**: 2025-11-21
**Version**: 1.0.0
**Dataset Size**: 50,000 transactions (train/val/test splits)
