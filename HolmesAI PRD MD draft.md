# **Product Requirements Document: AI-Based Financial Transaction Categorization Engine**

## **1\. Executive Summary**

### **1.1 Overview**

An AI-native transaction categorization engine that converts unstructured bank transaction descriptions into structured, three-level hierarchical categories with high confidence scores. The system uses Sentence-BERT for semantic encoding and LightGBM for classification, operating entirely without third-party APIs.

### **1.2 Core Value Proposition**

* **Privacy-first**: On-premise processing with zero data exfiltration  
* **Hierarchical Intelligence**: Three-level category taxonomy (L1 → L2 → L3)  
* **Cost-effective**: Build and deploy for under $100 using open-source tools  
* **Self-improving**: Continuous learning from user feedback

  ### **1.3 Success Metrics**

* Macro F1 score ≥ 0.90 at L3 categorization  
* Inference latency \< 200ms per transaction  
* Misclassification rate \< 2%  
* Cost reduction of 65-75% vs. API-based solutions  
  ---

  ## **2\. Problem Statement**

  ### **2.1 Current Challenges**

* **Merchant Name Variability**: "AMZN MKTP US", "Amazon.com", "AMZ\*Retail" refer to the same entity  
* **Lack of Structure**: Raw bank feeds lack standardized categorization  
* **API Dependencies**: Third-party services introduce cost, latency, and privacy risks  
* **Context Loss**: Traditional keyword matching fails on ambiguous merchants (e.g., "Shell" could be fuel or convenience store)

  ### **2.2 Target Users**

* **Primary**: Fintech platforms, expense management tools, personal finance apps  
* **Secondary**: Enterprise finance teams, accounting software providers  
* **Tertiary**: Individual developers building financial applications  
  ---

  ## **3\. System Architecture**

  ### **3.1 Pipeline Overview**

```
Raw Transaction → Data Ingestion → Pre-processing → Semantic Encoding → Classification → Confidence Scoring → Hierarchical Output
```

  ### **3.2 Component Breakdown**

  #### **Stage 1: Data Ingestion & Standardization**

**Input Formats**:

* CSV/JSON exports from banks and ERPs  
* Expected volume: 10-20M records/month

**Schema Normalization**:

| Field | Type | Description | Example |
| ----- | ----- | ----- | ----- |
| `transaction_id` | String | Unique identifier | "TXN\_20240115\_001" |
| `merchant_raw` | String | Raw merchant description | "AMZN MKTP US\*2A3B4C5D6" |
| `amount` | Float | Transaction amount | 49.99 |
| `currency` | String | ISO 4217 code | "USD" |
| `timestamp` | DateTime | Transaction timestamp | "2024-01-15T14:32:00Z" |
| `channel` | String | Payment method | "online", "pos", "atm" |
| `location` | String | Merchant location | "Seattle, WA" |
| `mcc_code` | String | Merchant Category Code | "5411" |

#### **Stage 2: Pre-processing & Token Enrichment**

**Text Cleaning Operations**:

1. Remove special characters: `[*#@$%^&()]`  
2. Normalize whitespace and casing  
3. Extract semantic tokens (brand, location, product type)  
4. Remove payment processor IDs (e.g., "SQ\*", "PAYPAL\*")

**Enrichment Features**:

* **Frequency**: Transaction count for merchant in last 90 days  
* **Spend Band**: Categorize amount (micro: \<$10, small: $10-50, medium: $50-200, large: \>$200)  
* **Temporal**: Day of week, time of day, month  
* **Geographic**: City, state/region derived from location field  
* **MCC Alignment**: Map MCC code to expected category clusters

**Example Transformation**:

```
Raw: "AMZN MKTP US*2A3B4C5D6 SEATTLE WA"
↓
Cleaned: "amazon marketplace seattle"
Tokens: ["amazon", "marketplace", "online_retail", "seattle", "washington"]
Enriched: {frequency: 12, spend_band: "small", temporal: "weekday_evening", region: "US-West"}
```

#### **Stage 3: Semantic Encoding (Sentence-BERT)**

**Model Specification**:

* **Base Model**: `sentence-transformers/all-MiniLM-L6-v2`  
* **Fine-tuning Dataset**: Financial transaction corpus from HuggingFace  
  * **Recommended**: `nlptown/financial-transactions` or `cais/mmlu` (economics subset)  
  * Augment with 50K labeled transactions across banking domains  
* **Output Dimension**: 384-dimensional dense vectors  
* **Similarity Metric**: Cosine similarity

**Training Strategy**:

1. Load pre-trained Sentence-BERT model  
2. Fine-tune on domain-specific transaction data using triplet loss:  
   * Anchor: Target transaction  
   * Positive: Same L3 category transaction  
   * Negative: Different L1 category transaction  
3. Training epochs: 3-5 with early stopping  
4. Validation: Ensure similar merchants cluster with cosine similarity ≥ 0.80

**Vector Storage**:

* **Database**: Supabase with pgvector extension  
* **Index Type**: HNSW (Hierarchical Navigable Small World) for fast similarity search  
* **Use Cases**:  
  * Alias resolution (find similar known merchants)  
  * Anomaly detection (flag unusual category patterns)  
  * Cold-start handling (assign new merchants to nearest neighbors)

  #### **Stage 4: Classification (LightGBM)**

**Model Configuration**:

```py
params = {
    'objective': 'multiclass',
    'num_class': 45,  # 15 L1 × 3 avg L2 per L1
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_data_in_leaf': 20
}
```

**Input Features** (Total: 368,726D):

* **Sentence-BERT Embeddings**: 384D (semantic representation)  
* **Enrichment Features**: 342D (one-hot encoded categorical \+ numerical)  
  * MCC codes: 300D (top 300 MCCs one-hot encoded)  
  * Temporal: 24D (hour one-hot) \+ 7D (day of week) \+ 12D (month)  
  * Geographic: 50D (top 50 cities/regions)  
  * Frequency/Spend: 2D (normalized numerical)  
* **Alias Similarity Scores**: 368,000D (similarity to 368K known merchants in vector DB)  
  * **Implementation**: For each transaction, retrieve top-100 similar merchants from pgvector, create sparse vector of similarities

**Note on Dimensionality**: The 368,726D space is primarily sparse (similarity scores), with dense representation limited to 726D (embeddings \+ enrichment). LightGBM efficiently handles sparse features through gradient-based tree splits.

**Training Dataset**:

* **Size**: 100,000 labeled transactions  
* **Split**: 70% train (70K), 15% validation (15K), 15% test (15K)  
* **Class Balancing**: Stratified sampling to ensure ≥ 500 samples per L3 category

**Output**:

* **L1 Category**: Primary category (e.g., "Travel")  
* **L2 Category**: Sub-category (e.g., "Travel \- Local")  
* **L3 Category**: Specific type (e.g., "Travel \- Local \- Uber")  
* **Confidence Scores**: Probability distribution across all L3 categories  
* **Feature Importance**: Top 5 tokens/features influencing prediction

  #### **Stage 5: Hierarchical Taxonomy & Confidence Scoring**

**Taxonomy Structure** (`taxonomy.json`):

```json
{
  "categories": [
    {
      "l1": "Travel",
      "l1_id": "TRV",
      "l2_subcategories": [
        {
          "l2": "Travel - Local",
          "l2_id": "TRV-LOC",
          "l3_types": [
            {
              "l3": "Travel - Local - Uber",
              "l3_id": "TRV-LOC-UBR",
              "aliases": ["uber", "uber eats", "uber trip"],
              "mcc_codes": ["4121"],
              "keywords": ["rideshare", "taxi", "cab"]
            },
            {
              "l3": "Travel - Local - Metro",
              "l3_id": "TRV-LOC-MET",
              "aliases": ["metro card", "subway", "transit"],
              "mcc_codes": ["4111"],
              "keywords": ["public transport", "rail"]
            }
          ]
        },
        {
          "l2": "Travel - International",
          "l2_id": "TRV-INT",
          "l3_types": []
        }
      ]
    },
    {
      "l1": "Dining",
      "l1_id": "DIN",
      "l2_subcategories": []
    }
  ]
}
```

**Full Taxonomy Coverage** (15 L1 Categories):

1. **Travel**: Local, International, Accommodation  
2. **Dining**: Restaurants, Fast Food, Coffee Shops  
3. **Shopping**: Retail, Online, Groceries  
4. **Entertainment**: Movies, Streaming, Events  
5. **Utilities**: Electric, Water, Internet  
6. **Healthcare**: Pharmacy, Hospital, Insurance  
7. **Education**: Tuition, Books, Courses  
8. **Transportation**: Fuel, Parking, Maintenance  
9. **Financial Services**: Banking, Investment, Insurance  
10. **Housing**: Rent, Mortgage, Home Improvement  
11. **Personal Care**: Salon, Gym, Wellness  
12. **Technology**: Electronics, Software, Subscriptions  
13. **Charitable**: Donations, Nonprofits  
14. **Business Expenses**: Office, Supplies, Services  
15. **Miscellaneous**: Uncategorized, Other

**Confidence Scoring Methodology**:

**Global L3 Confidence Calculation**:

```
Confidence_L3 = (LightGBM_Probability × 0.60) + 
                (Alias_Match_Score × 0.25) + 
                (MCC_Alignment_Score × 0.15)
```

**Component Definitions**:

1. **LightGBM\_Probability**: Direct model output for predicted L3 category  
2. **Alias\_Match\_Score**:  
   * 1.0 if merchant in known alias list  
   * 0.5-0.9 if cosine similarity to known merchant \> 0.80  
   * 0.0 otherwise  
3. **MCC\_Alignment\_Score**:  
   * 1.0 if transaction MCC matches taxonomy MCC for L3  
   * 0.5 if MCC matches L2 category  
   * 0.0 if no match

**Confidence Thresholds**:

* **High Confidence**: ≥ 0.85 (Auto-accept)  
* **Medium Confidence**: 0.70 \- 0.84 (Review if flagged)  
* **Low Confidence**: \< 0.70 (Mandatory human review)

  #### **Stage 6: Feedback & Continuous Learning**

**Review Queue**:

* All transactions with confidence \< 0.70 flagged for review  
* User interface displays: original transaction, predicted category, confidence score, alternative suggestions  
* User provides correction or confirmation

**Retraining Pipeline**:

1. **Nightly Batch**: Aggregate corrected labels from previous day  
2. **Dataset Update**: Append to training corpus with stratified sampling  
3. **Incremental Training**: Fine-tune LightGBM and Sentence-BERT on augmented dataset  
4. **Validation**: Test on held-out validation set (15K transactions)  
5. **Deployment**: If F1 improvement ≥ 0.5%, promote to production  
6. **Monitoring**: Track quarterly accuracy gains (target: 3-5% improvement)  
   ---

   ## **4\. Training Data Synthesis**

   ### **4.1 Synthetic Dataset Construction (100K Parameters)**

**Source 1: Personal Credit Card Statements (50K transactions)**

* **Acquisition**: Real anonymized credit card statements (6 CSV files provided)  
* **Data Characteristics**:  
  * Raw merchant descriptions with payment processor prefixes  
  * Transaction amounts ranging from micro-payments to large purchases  
  * Multi-month temporal coverage showing seasonal patterns  
  * Mix of online, point-of-sale, and recurring transactions  
  * Geographic diversity across multiple cities and regions  
* **Structure**:  
  * Merchant names with variations (e.g., "SWIGGY", "Swiggy\*Food", "SWIGGY DELIVERY")  
  * Amounts following real-world spending distributions  
  * Temporal patterns (commute times, meal hours, weekend entertainment)  
  * Multiple payment channels (UPI, card, online banking)  
* **Categories**: All 15 L1 categories represented in actual spending behavior  
* **Labeling Strategy**:  
  * Initial rule-based categorization using merchant patterns  
  * Manual verification of ambiguous cases (10-15% of dataset)  
  * Cross-validation against MCC codes where available

**Source 2: VISA & Mastercard MCC Documentation (50K mappings)**

* **Acquisition**:  
  * VISA MCC Directory: https://developer.visa.com/capabilities/visa-merchant-data-standards  
  * Mastercard Category Codes: Open documentation  
* **Structure**:  
  * MCC Code → Category Description → L1/L2/L3 mapping  
  * Example: "5411 \- Grocery Stores, Supermarkets" → "Shopping \- Groceries \- Supermarket"  
* **Augmentation**: Generate synthetic transactions for each MCC with realistic merchant names

**Dataset Composition**:

```
Total: 100,000 transactions
├── 70,000 Training
│   ├── 35,000 from personal statements
│   └── 35,000 from MCC mappings
├── 15,000 Validation
│   ├── 7,500 from personal statements
│   └── 7,500 from MCC mappings
└── 15,000 Test (held-out)
    ├── 7,500 from personal statements
    └── 7,500 from MCC mappings
```

**Synthetic Generation Process**:

1. **Merchant Name Variation**: Create 5-10 variants per canonical merchant  
   * "Starbucks" → \["STARBUCKS \#12345", "SBX\*COFFEE", "STARBUCKS STORE"\]  
2. **Amount Sampling**: Draw from category-specific distributions  
   * Coffee: μ=$5.50, σ=$2.00  
   * Groceries: μ=$75.00, σ=$35.00  
3. **Temporal Patterns**: Assign realistic timestamps  
   * Commute rides: 7-9 AM, 5-7 PM weekdays  
   * Dining: 12-1 PM (lunch), 7-9 PM (dinner)  
4. **Geographic Diversity**: Sample from 50 major US cities with population weighting

   ### **4.2 Data Quality Assurance**

* **Balance Check**: Ensure each L3 category has ≥ 500 training samples  
* **Overlap Prevention**: No merchant string appears in both train and test sets  
* **Edge Case Inclusion**: 10% of dataset includes ambiguous/challenging cases  
  * Multi-category merchants (Amazon: Shopping vs. Streaming)  
  * Refunds and credits (negative amounts)  
  * Foreign currency transactions

  ---

  ## **5\. Model Validation & Monitoring**

  ### **5.1 Evaluation Metrics**

**Primary Metrics**:

* **Macro F1 Score**: Unweighted average F1 across all L3 categories (target ≥ 0.90)  
* **Weighted F1 Score**: Accounts for class imbalance  
* **Hierarchical Accuracy**:  
  * L1 Accuracy: % correct primary category  
  * L2 Accuracy: % correct sub-category (given correct L1)  
  * L3 Accuracy: % correct specific type (given correct L2)

**Secondary Metrics**:

* **Precision@K**: Correct category in top-K predictions (K=3, K=5)  
* **Confusion Matrix**: Identify systematic misclassification patterns  
* **Category-wise Performance**: F1 score per L1/L2/L3 category

  ### **5.2 Model Drift Detection**

**Rolling Window Analysis**:

* **Window Size**: 7 days (≈50K transactions at 7K/day)  
* **Trigger**: F1 score drop \> 3% from baseline  
* **Alert**: Email \+ dashboard notification to ML team

**Drift Indicators**:

1. **Concept Drift**: New merchant patterns not in training data  
2. **Data Drift**: Shift in feature distributions (e.g., seasonal spending changes)  
3. **Label Drift**: User corrections indicate systematic taxonomy issues

**Mitigation Actions**:

* **Immediate**: Increase confidence threshold, route more to review  
* **Short-term**: Expedite retraining cycle (within 24 hours)  
* **Long-term**: Taxonomy update, feature engineering improvements

  ### **5.3 Explainability & Auditability**

**Transaction-level Explanation**:

```json
{
  "transaction_id": "TXN_20240115_001",
  "predicted_category": {
    "l1": "Dining",
    "l2": "Dining - Coffee Shops",
    "l3": "Dining - Coffee Shops - Starbucks"
  },
  "confidence": 0.92,
  "explanation": {
    "top_features": [
      {"feature": "merchant_token_starbucks", "importance": 0.45},
      {"feature": "alias_match_similarity", "importance": 0.30},
      {"feature": "mcc_5814_match", "importance": 0.15},
      {"feature": "amount_band_small", "importance": 0.10}
    ],
    "similar_transactions": [
      {"merchant": "Starbucks #5432", "category": "Dining - Coffee Shops - Starbucks", "similarity": 0.94},
      {"merchant": "SBUX*PIKE PLACE", "category": "Dining - Coffee Shops - Starbucks", "similarity": 0.89}
    ]
  }
}
```

**Model-level Dashboards**:

* Feature importance rankings (updated monthly)  
* Category accuracy heatmaps  
* Confidence distribution histograms  
* Retraining impact analysis (before/after F1 comparison)  
  ---

  ## **6\. Technical Implementation**

  ### **6.1 Infrastructure Requirements**

**Compute**:

* **Development**: Google Colab (free tier, 12GB RAM, GPU optional)  
* **Training**: Local machine (16GB RAM, 8-core CPU) or cloud (AWS t3.xlarge)  
* **Inference**: On-premise server (8GB RAM, 4-core CPU) or edge device  
* **Estimated Cost**: \< $50 for development \+ training (using free tiers)

**Storage**:

* **Database**: Supabase (free tier: 500MB, pgvector enabled)  
* **Vector Index**: HNSW for 100K-1M embeddings (≈2GB storage)  
* **Model Artifacts**:  
  * Sentence-BERT: 80MB  
  * LightGBM: 50MB  
  * Taxonomy JSON: 1MB

**Deployment**:

* **API Framework**: FastAPI \+ Uvicorn  
* **Containerization**: Docker (multi-stage build)  
* **Orchestration**: Docker Compose or Kubernetes (optional)  
* **Monitoring**: Prometheus \+ Grafana

  ### **6.2 Technology Stack**

**Core Libraries**:

```py
# NLP & Embeddings
sentence-transformers==2.2.2
transformers==4.35.0

# Machine Learning
lightgbm==4.1.0
scikit-learn==1.3.2

# Data Processing
pandas==2.1.3
numpy==1.24.3

# Vector Database
supabase==2.0.0
pgvector==0.2.3

# API & Deployment
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0

# Monitoring
prometheus-client==0.19.0
```

**HuggingFace Dataset** (Fine-tuning Sentence-BERT):

* **Primary**: `nlptown/financial-transactions` or custom scraped dataset  
* **Fallback**: Synthetic dataset from MCC mappings \+ GPT-generated variants  
* **Size**: 50K labeled merchant descriptions across all L3 categories

  ### **6.3 API Specification**

**Endpoint**: `POST /api/v1/categorize`

**Request**:

```json
{
  "transactions": [
    {
      "transaction_id": "TXN_001",
      "merchant_raw": "AMZN MKTP US*2A3B4C5D6",
      "amount": 49.99,
      "currency": "USD",
      "timestamp": "2024-01-15T14:32:00Z",
      "channel": "online",
      "location": "Seattle, WA",
      "mcc_code": "5942"
    }
  ]
}
```

**Response**:

```json
{
  "results": [
    {
      "transaction_id": "TXN_001",
      "category": {
        "l1": "Shopping",
        "l1_id": "SHP",
        "l2": "Shopping - Online",
        "l2_id": "SHP-ONL",
        "l3": "Shopping - Online - Amazon",
        "l3_id": "SHP-ONL-AMZ"
      },
      "confidence": 0.94,
      "processing_time_ms": 145,
      "explanation": {
        "top_features": [],
        "similar_transactions": []
      }
    }
  ],
  "metadata": {
    "total_processed": 1,
    "avg_confidence": 0.94,
    "low_confidence_count": 0
  }
}
```

---

## **7\. Privacy & Compliance**

### **7.1 Data Privacy**

* **On-Premise Processing**: All inference runs locally, no external API calls  
* **Data Minimization**: Only essential fields stored (no PII like cardholder names)  
* **Encryption**: AES-256 for data at rest, TLS 1.3 for data in transit  
* **Retention**: Transaction data deleted after 90 days (configurable)

  ### **7.2 Regulatory Compliance**

* **PCI-DSS**: No storage of sensitive authentication data (CVV, PIN)  
* **GDPR**: Right to erasure, data portability, consent management  
* **SOC 2**: Audit logging, access controls, incident response  
  ---

  ## **8\. Success Criteria & KPIs**

  ### **8.1 Technical KPIs**

* Macro F1 ≥ 0.90 on test set  
* Inference latency \< 200ms per transaction  
* Model retraining cycle \< 24 hours  
* Vector database similarity search \< 50ms

  ### **8.2 Business KPIs**

* Cost per 1M transactions \< $10 (vs. $150-300 for API solutions)  
* Misclassification rate \< 2%  
* User satisfaction score ≥ 4.5/5  
* Quarterly accuracy improvement ≥ 3%

  ### **8.3 Adoption Metrics**

* Time to onboard new taxonomy category \< 10 minutes  
* API response success rate ≥ 99.5%  
* Documentation completeness score ≥ 90%  
  ---

  ## **9\. Roadmap & Milestones**

  ### **Phase 1: Foundation (Weeks 1-4)**

* Data ingestion pipeline \+ schema normalization  
* Pre-processing module with token enrichment  
* Supabase setup with pgvector  
* Synthetic dataset generation (100K transactions)

  ### **Phase 2: Model Development (Weeks 5-8)**

* Fine-tune Sentence-BERT on financial corpus  
* Train LightGBM classifier (70K training samples)  
* Implement confidence scoring algorithm  
* Build taxonomy.json with 15 L1 categories

  ### **Phase 3: Integration & Testing (Weeks 9-10)**

* FastAPI deployment with Docker  
* End-to-end pipeline testing (100K test transactions)  
* Performance benchmarking (latency, F1, cost)  
* Model drift detection system

  ### **Phase 4: Production Readiness (Weeks 11-12)**

* Feedback loop \+ retraining automation  
* Monitoring dashboards (Prometheus \+ Grafana)  
* Documentation \+ developer guides  
* Security audit \+ penetration testing  
  ---

  ## **10\. Risk Assessment**

| Risk | Impact | Probability | Mitigation |
| ----- | ----- | ----- | ----- |
| Overfitting to synthetic data | High | Medium | Include 20% real-world transactions in test set |
| Model drift in production | High | Medium | Weekly drift monitoring \+ automated retraining |
| Cold-start for rare merchants | Medium | High | Alias similarity fallback \+ manual review |
| Taxonomy maintenance overhead | Medium | Medium | Version control \+ change impact analysis |
| Infrastructure costs exceed budget | Low | Low | Monitor resource usage \+ auto-scaling limits |

  ---

  ## **11\. Appendix**

  ### **A. Glossary**

* **MCC (Merchant Category Code)**: 4-digit code classifying business types  
* **Sentence-BERT**: Transformer model optimized for semantic similarity  
* **LightGBM**: Gradient boosting framework for tree-based models  
* **pgvector**: PostgreSQL extension for vector similarity search  
* **HNSW**: Approximate nearest neighbor search algorithm

  ### **B. References**

* Original System Document: "Automated AI-Based Financial Transaction Categorisation"  
* Visa MCC Directory: https://developer.visa.com  
* Sentence-BERT Paper: https://arxiv.org/abs/1908.10084  
* LightGBM Documentation: https://lightgbm.readthedocs.io

  ### **C. Contact & Support**

* **Product Owner**: [Pranav Mudigandur Venkat](mailto:pranav@backbase.com), [Pratima Nemani](mailto:pratima@backbase.com)  
* **Engineering Lead**: [Pranav Mudigandur Venkat](mailto:pranav@backbase.com), [Pratima Nemani](mailto:pratima@backbase.com)  
* **Feedback Channel**: GitHub Issues / Slack \#transaction-categorization


