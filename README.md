# NBA MVP Prediction Project

## Overview
This project predicts NBA MVP voting outcomes using player statistics. It uses a two-stage machine learning pipeline: first identifying who gets MVP votes, then predicting how many votes they receive.

## Data Collection (scraping.ipynb)
Scrapes historical NBA data from Basketball-Reference.com (1991-2024):
- MVP voting records
- Player statistics (per game averages)
- Advanced player metrics (efficiency, win shares, etc.)
- Team win-loss records

Data is saved as CSV files for analysis.

## Data Preprocessing (preprocessing.ipynb)
Cleans and prepares raw data for machine learning:

**Fixes player data:**
- Handles players who switched teams mid-season (merges their stats)
- Removes duplicate entries and invalid records
- Converts all stats to numerical values
- Fills missing values with 0

**Normalizes statistics:**
- Scales each stat by year to account for league-wide changes (inflation/deflation)
- Ensures stats are comparable across decades
- Uses mean/standard deviation normalization per year

**Combines datasets:**
- Merges regular season stats with advanced metrics
- Adds MVP voting share for each player
- Filters out reserves (players with <40 games started)
- Creates final dataset: 15,818 player-seasons ready for ML

## Stage 1: Classification - Who Gets MVP Votes?
Determines if a player receives MVP votes (yes/no).

**Model: Neural Network + XGBoost**
- Training data: 80% | Testing data: 20%
- Threshold: Players with >0.1% of MVP votes are considered candidates

**Best Results (XGBoost):**
- Accuracy: 96%
- Recall: 88% (catches most real MVP candidates)
- Precision: 74% (most predictions are correct)
- ROC-AUC: 0.9862 (excellent discrimination)

## Stage 2: Regression - How Many Votes?
Predicts the exact MVP vote share for players identified as candidates.

**Model: Ridge Regression**
- Optimized alpha: 42.32 (tested 10,000 values)
- MSE: 0.0358 (very small error)

**Key Stats Affecting MVP Votes:**
1. Win Shares (34% importance) - team success
2. VORP (8%) - player value above replacement
3. BPM (6%) - box plus-minus
4. TS% (5%) - shooting efficiency
5. USG% (5%) - usage rate

## Results
The two-stage pipeline successfully:
- Identifies 88% of actual MVP candidates
- Predicts their vote share with high accuracy
- Win Shares is the dominant factor in MVP voting

## Notebooks
- **scraping.ipynb** - Collects all data from Basketball-Reference
- **preprocessing.ipynb** - Cleans, normalizes, and merges all datasets
- **learning.ipynb** - Tests multiple classification and regression models
- **learning2.ipynb** - Final optimized pipeline with best hyperparameters
