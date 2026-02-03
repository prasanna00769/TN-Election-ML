import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

def create_features(df):
    """
    Create features for election prediction
    Returns: DataFrame with engineered features
    """
    print("Creating features...")
    
    features_list = []
    
    # Get unique constituencies and parties
    constituencies = df['constituency'].unique()
    
    for constituency in constituencies:
        # Get data for this constituency
        const_data = df[df['constituency'] == constituency].copy()
        
        # Get unique parties that contested in this constituency
        parties = const_data['party'].unique()
        
        for party in parties:
            # Get party data for 2016 and 2021
            party_2016 = const_data[(const_data['party'] == party) & (const_data['year'] == 2016)]
            party_2021 = const_data[(const_data['party'] == party) & (const_data['year'] == 2021)]
            
            # Skip if party didn't contest in 2021 (we're predicting based on 2021 data)
            if len(party_2021) == 0:
                continue
            
            # Initialize feature dictionary
            features = {
                'constituency': constituency,
                'party': party
            }
            
            # --- HISTORICAL PERFORMANCE FEATURES (2016) ---
            if len(party_2016) > 0:
                features['votes_2016'] = party_2016.iloc[0]['votes']
                features['total_votes_2016'] = party_2016.iloc[0]['total_votes']
                features['vote_share_2016'] = (party_2016.iloc[0]['votes'] / party_2016.iloc[0]['total_votes']) * 100
                features['was_winner_2016'] = 1 if party_2016.iloc[0]['party'] == party_2016.iloc[0]['winning_party'] else 0
                features['contested_2016'] = 1
            else:
                features['votes_2016'] = 0
                features['total_votes_2016'] = 0
                features['vote_share_2016'] = 0
                features['was_winner_2016'] = 0
                features['contested_2016'] = 0
            
            # --- HISTORICAL PERFORMANCE FEATURES (2021) ---
            features['votes_2021'] = party_2021.iloc[0]['votes']
            features['total_votes_2021'] = party_2021.iloc[0]['total_votes']
            features['vote_share_2021'] = (party_2021.iloc[0]['votes'] / party_2021.iloc[0]['total_votes']) * 100
            features['was_winner_2021'] = 1 if party_2021.iloc[0]['party'] == party_2021.iloc[0]['winning_party'] else 0
            features['contested_2021'] = 1
            
            # --- TREND FEATURES ---
            if features['contested_2016'] == 1:
                features['vote_change_abs'] = features['votes_2021'] - features['votes_2016']
                if features['votes_2016'] > 0:
                    features['vote_change_pct'] = ((features['votes_2021'] - features['votes_2016']) / features['votes_2016']) * 100
                else:
                    features['vote_change_pct'] = 100.0  # New entry, cap at 100%
                features['vote_share_change'] = features['vote_share_2021'] - features['vote_share_2016']
            else:
                features['vote_change_abs'] = features['votes_2021']
                features['vote_change_pct'] = 100.0
                features['vote_share_change'] = features['vote_share_2021']
            
            # --- CONSECUTIVE WINS ---
            features['consecutive_wins'] = features['was_winner_2016'] + features['was_winner_2021']
            
            # --- CONSTITUENCY-LEVEL FEATURES ---
            const_2021 = const_data[const_data['year'] == 2021]
            features['num_parties_2021'] = len(const_2021)
            
            # Get winner's vote share in 2021
            winner_2021 = const_2021[const_2021['party'] == const_2021.iloc[0]['winning_party']]
            if len(winner_2021) > 0:
                features['winner_vote_share_2021'] = (winner_2021.iloc[0]['votes'] / winner_2021.iloc[0]['total_votes']) * 100
            else:
                features['winner_vote_share_2021'] = 0
            
            # Turnout change
            const_2016 = const_data[const_data['year'] == 2016]
            if len(const_2016) > 0:
                total_2016 = const_2016.iloc[0]['total_votes']
                total_2021 = const_2021.iloc[0]['total_votes']
                features['turnout_change'] = ((total_2021 - total_2016) / total_2016) * 100 if total_2016 > 0 else 0
            else:
                features['turnout_change'] = 0
            
            # --- PARTY-LEVEL FEATURES (statewide) ---
            # Count seats won by this party in 2021
            party_wins_2021 = df[(df['year'] == 2021) & (df['party'] == party) & (df['party'] == df['winning_party'])]
            features['party_seats_won_2021'] = len(party_wins_2021)
            
            # Count constituencies where party contested in 2021
            party_contested_2021 = df[(df['year'] == 2021) & (df['party'] == party)]
            features['party_contested_constituencies_2021'] = len(party_contested_2021)
            
            # Party average vote share across all constituencies
            if len(party_contested_2021) > 0:
                features['party_avg_vote_share_2021'] = party_contested_2021.apply(
                    lambda x: (x['votes'] / x['total_votes']) * 100, axis=1
                ).mean()
            else:
                features['party_avg_vote_share_2021'] = 0
            
            # --- COMPETITIVE POSITION FEATURES ---
            # Rank in 2021
            const_2021_sorted = const_2021.sort_values('votes', ascending=False).reset_index(drop=True)
            party_rank = const_2021_sorted[const_2021_sorted['party'] == party].index
            features['rank_2021'] = party_rank[0] + 1 if len(party_rank) > 0 else len(const_2021_sorted) + 1
            
            # Margin from winner or to second place
            if features['was_winner_2021'] == 1:
                # This party won - margin to second place
                if len(const_2021_sorted) > 1:
                    features['margin_votes'] = const_2021_sorted.iloc[0]['votes'] - const_2021_sorted.iloc[1]['votes']
                else:
                    features['margin_votes'] = features['votes_2021']
            else:
                # This party lost - margin to winner
                winner_votes = const_2021_sorted.iloc[0]['votes']
                features['margin_votes'] = features['votes_2021'] - winner_votes
            
            # --- BINARY INDICATORS ---
            features['is_major_party'] = 1 if features['party_seats_won_2021'] >= 5 else 0
            
            # National vs Regional party
            national_parties = ['Bharatiya Janata Party', 'Indian National Congress', 
                              'Communist Party Of India', 'Communist Party Of India (Marxist)']
            features['is_national_party'] = 1 if party in national_parties else 0
            
            # Dravidian parties
            dravidian_parties = ['Dravida Munnetra Kazhagam', 'All India Anna Dravida Munnetra Kazhagam',
                               'Desiya Murpokku Dravida Kazhagam']
            features['is_dravidian_party'] = 1 if party in dravidian_parties else 0
            
            # --- TARGET VARIABLE ---
            features['winning_party'] = party_2021.iloc[0]['winning_party']
            
            features_list.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    print(f"Created {len(features_df)} feature records")
    
    return features_df

def prepare_data(features_df):
    """
    Prepare data for training: encode categorical variables and split features/target
    """
    print("\nPreparing data for training...")
    
    # Encode constituency (simple label encoding)
    const_encoder = LabelEncoder()
    features_df['constituency_encoded'] = const_encoder.fit_transform(features_df['constituency'])
    
    # Encode party (simple label encoding)
    party_encoder = LabelEncoder()
    features_df['party_encoded'] = party_encoder.fit_transform(features_df['party'])
    
    # Encode target variable (winning_party)
    target_encoder = LabelEncoder()
    features_df['target'] = target_encoder.fit_transform(features_df['winning_party'])
    
    # Select feature columns (exclude original text columns and target)
    feature_columns = [
        'constituency_encoded', 'party_encoded',
        'votes_2016', 'total_votes_2016', 'vote_share_2016', 'was_winner_2016', 'contested_2016',
        'votes_2021', 'total_votes_2021', 'vote_share_2021', 'was_winner_2021', 'contested_2021',
        'vote_change_abs', 'vote_change_pct', 'vote_share_change', 'consecutive_wins',
        'num_parties_2021', 'winner_vote_share_2021', 'turnout_change',
        'party_seats_won_2021', 'party_contested_constituencies_2021', 'party_avg_vote_share_2021',
        'rank_2021', 'margin_votes',
        'is_major_party', 'is_national_party', 'is_dravidian_party'
    ]
    
    X = features_df[feature_columns]
    y = features_df['target']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{features_df['winning_party'].value_counts()}")
    
    return X, y, target_encoder, const_encoder, party_encoder

def train_model(X, y):
    """
    Train Random Forest classifier
    """
    print("\nTraining Random Forest model...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Initialize and train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return model, test_accuracy

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Tamil Nadu Election Winner Prediction - Model Training")
    print("=" * 60)
    
    # Load processed data
    print("\nLoading processed election data...")
    df = pd.read_csv('data/processed_election_data.csv')
    print(f"Loaded {len(df)} records")
    print(f"Years: {sorted(df['year'].unique())}")
    print(f"Constituencies: {df['constituency'].nunique()}")
    print(f"Parties: {df['party'].nunique()}")
    
    # Create features
    features_df = create_features(df)
    
    # Prepare data
    X, y, target_encoder, const_encoder, party_encoder = prepare_data(features_df)
    
    # Train model
    model, test_accuracy = train_model(X, y)
    
    # Save model and encoders
    print("\nSaving model and encoders...")
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Model saved to model.pkl")
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(target_encoder, f)
    print("✓ Label encoder saved to label_encoder.pkl")
    
    with open('constituency_encoder.pkl', 'wb') as f:
        pickle.dump(const_encoder, f)
    print("✓ Constituency encoder saved to constituency_encoder.pkl")
    
    with open('party_encoder.pkl', 'wb') as f:
        pickle.dump(party_encoder, f)
    print("✓ Party encoder saved to party_encoder.pkl")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("\nModel files created:")
    print("- model.pkl")
    print("- label_encoder.pkl")
    print("- constituency_encoder.pkl")
    print("- party_encoder.pkl")
    print("=" * 60)

if __name__ == "__main__":
    main()