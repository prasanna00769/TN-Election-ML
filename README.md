# Tamil Nadu Election Winner Prediction (2016-2021-2026)

## Project Goal

This project analyzes Tamil Nadu state assembly election results from 2016 and 2021 to predict potential winners for the 2026 elections. Using machine learning techniques, specifically Random Forest classification, the model learns from historical voting patterns, party performance trends, and constituency-level dynamics to forecast election outcomes.

**Key Objectives:**
- Analyze voting patterns and trends between 2016 and 2021
- Build a predictive model to forecast 2026 election winners by constituency
- Visualize party performance and vote share changes over time
- Understand key factors influencing election outcomes

---

## Dataset Description

### Source Files
- **`data/tn_2016_election.csv`** - Tamil Nadu Assembly Election results from 2016
- **`data/tn_2021_election.csv`** - Tamil Nadu Assembly Election results from 2021

### Data Structure

**2016 Election Data:**
- 233 constituencies
- ~100+ parties (varying by constituency)
- Columns: Constituency name, party-candidate pairs, vote counts, summary statistics
- Each party has two columns: candidate name and votes received

**2021 Election Data:**
- 234 constituencies
- ~100+ parties (varying by constituency)
- Columns: Constituency ID, name, total votes, party names with EVM and postal votes
- Each party has three columns: candidate name, EVM votes, postal votes

**Processed Data (`processed_election_data.csv`):**
Each row represents a party's performance in a specific constituency and year:
- `constituency` - Name of the constituency
- `year` - Election year (2016 or 2021)
- `party` - Political party name
- `votes` - Number of votes received
- `total_votes` - Total votes cast in the constituency
- `winning_party` - Party that won the constituency

---

## File Structure
```
├── data/
│   ├── tn_2016_election.csv           # Raw 2016 election data
│   ├── tn_2021_election.csv           # Raw 2021 election data
│   ├── processed_election_data.csv    # Cleaned and standardized data
│   └── predicted_2026_results.csv     # 2026 predictions (generated)
│
├── preprocess.py                      # Data cleaning and preprocessing
├── train_model.py                     # Feature engineering and model training
├── evaluate.py                        # Model evaluation and metrics
├── predict_2026.py                    # Generate 2026 predictions
├── visualize.py                       # Create visualizations
│
├── model.pkl                          # Trained Random Forest model
├── label_encoder.pkl                  # Encoder for winning party labels
├── constituency_encoder.pkl           # Encoder for constituencies
├── party_encoder.pkl                  # Encoder for parties
│
└── README.md                          # This file
```

---

## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Step-by-Step Execution

#### 1. **Preprocess the Data**
Cleans and standardizes the raw CSV files into a unified format.
```bash
python preprocess.py
```

**Outputs:**
- `data/processed_election_data.csv`

**What it does:**
- Reads both 2016 and 2021 election CSV files
- Extracts constituency names, party names, and vote counts
- Standardizes naming conventions
- Handles missing data (parties that didn't contest)
- Calculates winning party for each constituency
- Combines both years into single dataset

---

#### 2. **Train the Model**
Engineers features and trains a Random Forest classifier.
```bash
python train_model.py
```

**Outputs:**
- `model.pkl` - Trained Random Forest model
- `label_encoder.pkl` - Target variable encoder
- `constituency_encoder.pkl` - Constituency encoder
- `party_encoder.pkl` - Party encoder

**What it does:**
- Creates 27 engineered features per party-constituency combination:
  - Historical performance (2016 & 2021)
  - Vote trends and momentum
  - Constituency competition metrics
  - Statewide party strength
  - Competitive positioning
- Trains Random Forest with 200 estimators
- Reports training and test accuracy
- Shows feature importance

**Expected Performance:**
- Training accuracy: ~95-99%
- Test accuracy: ~70-85%
- Constituency-level accuracy: ~60-75%

---

#### 3. **Evaluate the Model**
Assesses model performance with detailed metrics.
```bash
python evaluate.py
```

**What it does:**
- Loads trained model and encoders
- Makes predictions on available data
- Calculates overall accuracy
- Displays confusion matrix (top 10 parties)
- Shows classification report (precision, recall, F1-score)
- Evaluates constituency-level accuracy
- Lists correctly and incorrectly predicted constituencies

**Metrics displayed:**
- Party-level accuracy (how many party-constituency combinations correct)
- Constituency-level accuracy (how many constituencies have correct winner)
- Per-party precision, recall, and F1-scores

---

#### 4. **Predict 2026 Winners**
Generates predictions for the 2026 election.
```bash
python predict_2026.py
```

**Outputs:**
- `data/predicted_2026_results.csv`

**What it does:**
- Extrapolates 2026 vote counts based on 2016→2021 trends
- Assumes linear continuation of growth/decline rates
- Creates same features used during training
- Predicts winning party for each constituency
- Calculates prediction confidence scores
- Saves results with runner-up parties

**Output CSV contains:**
- `constituency` - Constituency name
- `predicted_winning_party` - Predicted winner
- `confidence` - Model confidence (0-1)
- `runner_up_party` - Second-place prediction
- `num_parties_contested` - Number of parties

---


## Model Features

The prediction model uses 27 engineered features:

**Historical Performance (10 features)**
- Votes and vote share in 2016 and 2021
- Winner status in previous elections
- Whether party contested in both years

**Trend Features (6 features)**
- Absolute and percentage vote change
- Vote share momentum
- Consecutive wins indicator

**Constituency-Level (3 features)**
- Number of competing parties
- Winner's vote share
- Turnout changes

**Party-Level (3 features)**
- Total seats won statewide
- Number of constituencies contested
- Average vote share across state

**Competitive Position (3 features)**
- Party rank in constituency
- Margin from winner/to second place

**Binary Indicators (3 features)**
- Is major party (>5 seats)
- Is national party (BJP/INC/CPI/CPI(M))
- Is Dravidian party (DMK/ADMK/DMDK)

---

## Key Insights from Analysis

**Major Parties (2021):**
- **DMK** - Won majority with ~133 seats
- **ADMK** - Main opposition with ~66 seats
- **BJP, INC, PMK, CPI(M), VCK** - Alliance partners with varying seats

**Trends (2016→2021):**
- DMK gained significant seats (anti-incumbency factor)
- ADMK lost seats after governing 2016-2021
- Regional parties maintained consistent base
- Vote share shifts indicate swing constituencies

---

## Disclaimer & Limitations

### ⚠️ **IMPORTANT: Prediction Limitations**

This model is a **statistical exercise** and should **NOT** be considered an accurate forecast of actual 2026 election results.

**Critical Limitations:**

1. **Missing Political Context:**
   - Alliance formations (DMK+, ADMK+, NDA alliances)
   - Candidate selection and popularity
   - Campaign quality and spending
   - Current political climate and issues

2. **External Factors Not Captured:**
   - Economic conditions (inflation, employment, development)
   - National political trends (Modi wave, anti-incumbency)
   - Local issues (water, infrastructure, farmer concerns)
   - Media coverage and public sentiment
   - Government performance and welfare schemes

3. **Data Assumptions:**
   - Assumes linear trends continue (rarely true)
   - Assumes same parties will contest
   - Ignores new parties or significant party splits
   - Does not account for constituency boundary changes
   - No data on candidate quality or criminal records

4. **Historical Precedent:**
   - 2016: ADMK won majority
   - 2021: DMK won majority (complete reversal)
   - Elections are highly unpredictable

5. **Model Accuracy:**
   - ~70-75% constituency-level accuracy on test data
   - Real-world accuracy likely much lower
   - High confidence scores don't guarantee correctness

### Intended Use

This project is designed for:
- **Educational purposes** - Learning ML classification techniques
- **Data analysis practice** - Working with real-world election data
- **Feature engineering study** - Creating meaningful political features
- **Statistical modeling** - Understanding voting pattern analysis

**NOT intended for:**
- Actual election forecasting
- Political decision-making
- Betting or wagering
- Media reporting without context

### Academic Honesty

Election prediction is an extremely complex problem involving sociology, economics, psychology, and political science. Pure vote-based statistical models have limited predictive power. Professional election forecasters use:
- Extensive polling data
- Demographic analysis
- Expert political commentary
- Alliance arithmetic
- Seat-specific candidate analysis
- Real-time campaign tracking

This project uses none of the above.

---

## Future Improvements

Potential enhancements to increase accuracy:

1. **Include alliance data** - Track pre-poll alliances (DMK+, ADMK+, NDA)
2. **Add demographic features** - Population, caste composition, urban/rural ratio
3. **Incorporate polling data** - Pre-election surveys and opinion polls
4. **Candidate-level analysis** - Incumbency, criminal records, assets
5. **Sentiment analysis** - Social media trends and news coverage
6. **Economic indicators** - Local development metrics, employment data
7. **Temporal proximity** - Weight recent trends more heavily
8. **Ensemble methods** - Combine multiple models (RF + XGBoost + Neural Networks)

---

## Technical Notes

**Model Choice:** Random Forest was selected over Logistic Regression because:
- Handles non-linear relationships
- Captures feature interactions automatically
- Robust to outliers
- Provides feature importance
- No need for extensive feature scaling

**Class Imbalance Handling:**
- Used `class_weight='balanced'` to give equal importance to all parties
- Prevents model from only predicting DMK/ADMK (dominant parties)

**Validation Strategy:**
- 80-20 train-test split with stratification
- Ensures all major parties represented in both sets

---

## License & Attribution

This project is for educational purposes only. Election data is publicly available from the Election Commission of India.

**Data Source:** Tamil Nadu State Election Commission

**Author:** R Prasanna rajan

**Date:** February 2026

---

## Contact

For questions, improvements, or collaboration:
- Open an issue on GitHub
- Email: prasannarajan698@gmail.com

---

**Remember: This is a learning project, not a crystal ball! 🔮**
