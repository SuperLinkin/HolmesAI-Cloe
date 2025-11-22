# Dataset Quick Start Guide

## TL;DR

```bash
# Generate datasets (already done)
python generate_dataset.py

# Validate datasets
python validate_dataset.py

# Use the data
import pandas as pd
train = pd.read_csv('data/train.csv')
```

## What You Have

✓ **50,000** synthetic transactions with realistic patterns
✓ **15** L1 categories, **42** L2 subcategories, **59** L3 types
✓ **Train/Val/Test** splits (70%/10%/20%)
✓ **MCC codes** from Visa/Mastercard/Amex
✓ **Multi-currency** support (USD, EUR, INR, GBP, AUD)

## Files Generated

```
data/
├── synthetic_transactions_1k.csv    # 1K records - quick testing
├── synthetic_transactions_10k.csv   # 10K records - development
├── synthetic_transactions_50k.csv   # 50K records - full dataset
├── train.csv                        # 35K records (70%)
├── val.csv                          # 5K records (10%)
└── test.csv                         # 10K records (20%)
```

## Sample Data

| transaction_id | merchant_raw | amount | currency | l1 | l2 | l3 |
|----------------|-------------|--------|----------|----|----|-----|
| TXN_20241121_001 | AMZN MKTP US*ABC123 | 49.99 | USD | Shopping | Shopping - Online | Shopping - Online - Amazon |
| TXN_20241121_002 | STARBUCKS #1234 | 6.50 | USD | Dining | Dining - Coffee Shops | Dining - Coffee Shops - Starbucks |
| TXN_20241121_003 | UBER TRIP NY | 25.00 | USD | Travel | Travel - Local | Travel - Local - Uber |

## Top Categories

1. **Dining** (22.1%) - Restaurants, fast food, coffee, delivery
2. **Shopping** (16.4%) - Retail, online, groceries
3. **Transportation** (10.9%) - Fuel, parking, maintenance
4. **Entertainment** (9.1%) - Movies, streaming, events
5. **Travel** (8.3%) - Local, international, accommodation

## Quick Stats

- **Date Range**: 365 days (1 year)
- **Amount Range**: $5 - $10,000
- **Average Amount**: $434
- **MCC Coverage**: 99%
- **Unique IDs**: 100%
- **Data Leakage**: None

## Loading Data

```python
import pandas as pd

# Load splits
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/val.csv')
test = pd.read_csv('data/test.csv')

# Parse timestamps
train['timestamp'] = pd.to_datetime(train['timestamp'])

# Check shape
print(f"Train: {train.shape}")  # (35000, 14)
print(f"Val: {val.shape}")      # (5000, 14)
print(f"Test: {test.shape}")    # (10000, 14)

# View categories
print(f"L1 categories: {train['l1'].nunique()}")  # 15
print(f"L2 categories: {train['l2'].nunique()}")  # 42
print(f"L3 categories: {train['l3'].nunique()}")  # 59
```

## Columns

**Input Features:**
- `transaction_id` - Unique ID
- `merchant_raw` - Raw merchant name
- `amount` - Transaction amount
- `currency` - ISO currency code
- `timestamp` - Transaction time
- `channel` - Payment channel (online/pos/atm/mobile)
- `location` - Merchant location (optional)
- `mcc_code` - Merchant Category Code (optional)

**Ground Truth Labels:**
- `l1`, `l1_id` - Level 1 category
- `l2`, `l2_id` - Level 2 subcategory
- `l3`, `l3_id` - Level 3 specific type

## Using with Your Model

```python
# Prepare features
X_train = train[['merchant_raw', 'amount', 'currency',
                 'channel', 'mcc_code', 'timestamp']]

# Prepare labels (for L3 classification)
y_train = train['l3_id']

# Or get all levels
y_train_l1 = train['l1_id']
y_train_l2 = train['l2_id']
y_train_l3 = train['l3_id']
```

## Regenerating Data

```python
from generate_dataset import SyntheticDatasetGenerator

# Create generator
gen = SyntheticDatasetGenerator()

# Generate custom size
df = gen.generate_dataset(
    num_samples=25000,
    output_path="data/custom.csv"
)

# Create splits
train, val, test = gen.generate_train_test_split(df)
```

## Validation

```bash
# Check data quality
python validate_dataset.py
```

Expected output:
- ✓ No duplicate IDs
- ✓ No missing required fields
- ✓ All amounts positive
- ✓ No data leakage
- ✓ Stratified splits

## Need Help?

- **Full docs**: See [DATASET.md](DATASET.md)
- **Taxonomy**: [src/config/taxonomy.json](src/config/taxonomy.json)
- **Generator**: [generate_dataset.py](generate_dataset.py)
- **Validator**: [validate_dataset.py](validate_dataset.py)

## Key Features

✓ Realistic merchant name patterns
✓ MCC code integration
✓ Multi-currency support
✓ Temporal patterns
✓ Category-appropriate amounts
✓ No data leakage
✓ Stratified splits
✓ Fully reproducible

---

Ready to train your model! 🚀
