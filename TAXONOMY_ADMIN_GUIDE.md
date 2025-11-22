# Taxonomy Configuration Guide for Admins

**For Non-Technical Users:** This guide shows how to modify transaction categories without writing code.

---

## 📁 Where is the Taxonomy File?

**Location:** `src/config/taxonomy.json`

This file controls all transaction categories the system recognizes.

---

## 🛡️ Safety First: Validation Tool

**Before** modifying the taxonomy, always validate after changes:

```bash
python validate_taxonomy.py
```

This checks for:
- ✅ Valid JSON syntax
- ✅ No duplicate categories
- ✅ All required fields present
- ✅ Proper structure

**Run this EVERY TIME before using the updated taxonomy!**

---

## 📝 How to Add a New Category

### Example: Adding "Streaming Services" under "Entertainment"

**Before Making Changes:**
1. Make a backup copy of `taxonomy.json`
2. Open `taxonomy.json` in a text editor (VS Code, Notepad++)

### Step 1: Find the L1 Category

Find the "Entertainment" section:

```json
{
  "l1": "Entertainment",
  "l1_id": "ENT",
  "l2_subcategories": [
    ...
  ]
}
```

### Step 2: Add L2 Subcategory

Add a new L2 entry in the `l2_subcategories` array:

```json
{
  "l2": "Entertainment - Streaming",
  "l2_id": "ENT-STR",
  "l3_types": [
    ...
  ]
}
```

⚠️ **Important:** Add a comma `,` after the previous L2 entry!

### Step 3: Add L3 Types

Add specific streaming services:

```json
{
  "l3": "Entertainment - Streaming - Netflix",
  "l3_id": "ENT-STR-NFX",
  "aliases": ["netflix", "netflix.com", "netflix subscription"],
  "mcc_codes": ["5968", "7996"],
  "keywords": ["streaming", "video", "movies", "shows"]
}
```

### Complete Example:

```json
{
  "l2": "Entertainment - Streaming",
  "l2_id": "ENT-STR",
  "l3_types": [
    {
      "l3": "Entertainment - Streaming - Netflix",
      "l3_id": "ENT-STR-NFX",
      "aliases": ["netflix", "netflix.com"],
      "mcc_codes": ["5968"],
      "keywords": ["streaming", "video"]
    },
    {
      "l3": "Entertainment - Streaming - Spotify",
      "l3_id": "ENT-STR-SPT",
      "aliases": ["spotify", "spotify premium"],
      "mcc_codes": ["5968"],
      "keywords": ["music", "audio", "streaming"]
    }
  ]
}
```

---

## 📋 Required Fields Checklist

### For L1 Category:
- ✅ `l1` - Category name (e.g., "Travel")
- ✅ `l1_id` - Unique 3-letter code (e.g., "TRV")
- ✅ `l2_subcategories` - Array of L2 categories

### For L2 Category:
- ✅ `l2` - Subcategory name (e.g., "Travel - Flight")
- ✅ `l2_id` - Unique code (e.g., "TRV-FLT")
- ✅ `l3_types` - Array of L3 types

### For L3 Category:
- ✅ `l3` - Detailed type (e.g., "Travel - Flight - International")
- ✅ `l3_id` - Unique code (e.g., "TRV-FLT-INT")
- ⚠️ `aliases` - Optional merchant name variations
- ⚠️ `mcc_codes` - Optional MCC codes (as strings)
- ⚠️ `keywords` - Optional search keywords

---

## ⚠️ Common Mistakes

### 1. **Missing Commas**

❌ **Wrong:**
```json
{
  "l3": "Category 1"
  "l3_id": "CAT1"
}
{
  "l3": "Category 2"
  "l3_id": "CAT2"
}
```

✅ **Correct:**
```json
{
  "l3": "Category 1",
  "l3_id": "CAT1"
},
{
  "l3": "Category 2",
  "l3_id": "CAT2"
}
```

### 2. **Trailing Comma**

❌ **Wrong:**
```json
{
  "l3": "Last Category",
  "l3_id": "LAST",
}
```

✅ **Correct:**
```json
{
  "l3": "Last Category",
  "l3_id": "LAST"
}
```

### 3. **Duplicate IDs**

❌ **Wrong:**
```json
{"l1_id": "TRV"},
...
{"l1_id": "TRV"}  // Duplicate!
```

✅ **Correct:**
```json
{"l1_id": "TRV"},
...
{"l1_id": "ENT"}  // Unique
```

---

## 🔍 How Aliases Work

**Aliases** are alternative merchant name patterns that map to a category.

**Example:** For "Starbucks", add aliases:

```json
{
  "l3": "Dining - Coffee - Starbucks",
  "aliases": [
    "starbucks",
    "starbucks coffee",
    "starbucks #",
    "sbux"
  ]
}
```

When a transaction has merchant "STARBUCKS #12345", it will match "starbucks #" and categorize as Coffee.

**Tips:**
- Use lowercase
- Include common variations
- Add partial matches (e.g., "starbucks #" matches "STARBUCKS #12345")

---

## 🏷️ How MCC Codes Work

**MCC (Merchant Category Code)** is a 4-digit code assigned by credit card companies.

**Example:**
```json
{
  "l3": "Dining - Restaurants - Fine Dining",
  "mcc_codes": ["5812", "5813"]
}
```

**Common MCC Codes:**
- `5812` - Restaurants
- `5814` - Fast Food
- `5411` - Grocery Stores
- `5541` - Gas Stations
- `5999` - Miscellaneous Retail

**Find MCC codes:** https://www.citibank.com/tts/solutions/commercial-cards/assets/docs/govt/Merchant-Category-Codes.pdf

---

## ✅ Validation Workflow

### Step-by-Step Process:

1. **Backup** the current taxonomy:
   ```bash
   cp src/config/taxonomy.json src/config/taxonomy.backup.json
   ```

2. **Edit** `taxonomy.json` with your changes

3. **Validate** the changes:
   ```bash
   python validate_taxonomy.py
   ```

4. **Fix errors** if validation fails (check the error message)

5. **Re-validate** until it passes

6. **Test** with a sample transaction:
   ```bash
   python demo.py
   ```

7. **Deploy** - The updated taxonomy is ready to use!

---

## 🧪 Testing Your Changes

After updating taxonomy, test with a sample transaction:

```python
from src.models import LightGBMClassifier

# Reload with new taxonomy
classifier = LightGBMClassifier(taxonomy_path="src/config/taxonomy.json")

# Test prediction
prediction = classifier.predict(sample_transaction)
print(prediction)
```

---

## 📊 Viewing Current Taxonomy

To see all current categories:

```bash
python -c "import json; data = json.load(open('src/config/taxonomy.json')); print(f'L1: {len(data[\"categories\"])} categories')"
```

Or use the validation tool:

```bash
python validate_taxonomy.py
```

This shows:
- Total L1 categories
- Total L2 categories
- Total L3 categories

---

## 🆘 Getting Help

**If validation fails:**
1. Check the error message - it tells you exactly what's wrong
2. Look at the line number mentioned
3. Compare with examples in this guide
4. Restore from backup if needed:
   ```bash
   cp src/config/taxonomy.backup.json src/config/taxonomy.json
   ```

**Common Error Messages:**

| Error | Meaning | Fix |
|-------|---------|-----|
| `Invalid JSON syntax` | Missing bracket or comma | Check JSON structure |
| `Duplicate L1 category` | Same category name twice | Use unique names |
| `Missing 'l1' field` | Required field missing | Add the missing field |
| `'categories' must be a list` | Wrong data type | Use `[]` for arrays |

---

## 📝 Quick Reference Template

**Copy this to add a new L3 category:**

```json
{
  "l3": "CategoryLevel1 - CategoryLevel2 - CategoryLevel3",
  "l3_id": "XXX-YYY-ZZZ",
  "aliases": ["alias1", "alias2", "alias3"],
  "mcc_codes": ["1234", "5678"],
  "keywords": ["keyword1", "keyword2"]
}
```

**Remember:**
- Add comma `,` before if not the first item
- NO comma after if it's the last item
- Always validate after changes!

---

## ✨ Best Practices

1. **Always backup before editing**
2. **Make small changes** - easier to debug
3. **Validate after EVERY change**
4. **Use descriptive category names**
5. **Include comprehensive aliases**
6. **Test with real transactions** before deploying
7. **Document your changes** (add comments in a separate notes file)
8. **Keep IDs short and meaningful** (3-6 characters)

---

## 🚀 Summary

**To update taxonomy:**

1. ✅ Backup `taxonomy.json`
2. ✅ Edit the file (add/modify categories)
3. ✅ Run `python validate_taxonomy.py`
4. ✅ Fix any errors
5. ✅ Test with sample data
6. ✅ Deploy!

**No coding required!** Just edit the JSON file and validate.

---

**Need Help?** Contact the development team with:
- The validation error message
- What you're trying to add
- Your taxonomy backup file
