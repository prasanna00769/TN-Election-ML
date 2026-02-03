import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

def create_features(df):
    """
    Create features for election prediction (same as train_model.py)
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
            
            # Skip if party didn't contest in 2021
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
                    features['vote_change_pct'] = 100.0
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

def prepare_data(features_df, const_encoder, party_encoder, target_encoder):
    """
    Prepare data using pre-fitted encoders
    """
    print("\nPreparing data for evaluation...")
    
    # Encode constituency
    features_df['constituency_encoded'] = const_encoder.transform(features_df['constituency'])
    
    # Encode party
    features_df['party_encoded'] = party_encoder.transform(features_df['party'])
    
    # Encode target variable
    features_df['target'] = target_encoder.transform(features_df['winning_party'])
    
    # Select feature columns
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
    
    return X, y, features_df

def print_confusion_matrix(cm, labels, top_n=10):
    """
    Print confusion matrix in a readable format (showing top N classes)
    """
    print("\nConfusion Matrix (Top {} classes by frequency):".format(top_n))
    print("=" * 80)
    
    # Get indices of top N classes
    class_counts = cm.sum(axis=1)
    top_indices = np.argsort(class_counts)[-top_n:][::-1]
    
    # Filter confusion matrix and labels
    cm_filtered = cm[np.ix_(top_indices, top_indices)]
    labels_filtered = [labels[i] for i in top_indices]
    
    # Print header
    max_label_len = max(len(str(label)) for label in labels_filtered)
    header = " " * (max_label_len + 2) + "Predicted"
    print(header)
    print(" " * (max_label_len + 2) + "-" * (len(labels_filtered) * 8))
    
    # Print column headers (abbreviated)
    col_header = " " * (max_label_len + 2)
    for i, label in enumerate(labels_filtered):
        col_header += f"{i:>7} "
    print(col_header)
    
    # Print rows
    print("Actual")
    for i, label in enumerate(labels_filtered):
        row_label = f"{i:>2}. {label[:max_label_len]}"
        row = f"{row_label:<{max_label_len+4}}"
        for j in range(len(labels_filtered)):
            row += f"{cm_filtered[i, j]:>7} "
        print(row)
    
    # Print legend
    print("\nLegend (Top {} classes):".format(top_n))
    for i, idx in enumerate(top_indices):
        print(f"{i:>2}. {labels[idx]}")

def evaluate_by_constituency(features_df, y_true, y_pred, target_encoder):
    """
    Evaluate model performance at constituency level
    """
    print("\n" + "=" * 80)
    print("Constituency-Level Evaluation")
    print("=" * 80)
    
    # Add predictions to features dataframe
    features_df['predicted_winner'] = target_encoder.inverse_transform(y_pred)
    features_df['actual_winner'] = features_df['winning_party']
    
    # Get one row per constituency (all rows have same winner for same constituency)
    constituency_results = features_df.groupby('constituency').first().reset_index()
    
    # Check if prediction matches actual winner
    constituency_results['correct'] = (
        constituency_results['predicted_winner'] == constituency_results['actual_winner']
    ).astype(int)
    
    # Calculate constituency-level accuracy
    const_accuracy = constituency_results['correct'].mean()
    total_const = len(constituency_results)
    correct_const = constituency_results['correct'].sum()
    
    print(f"\nConstituency-Level Accuracy: {const_accuracy:.4f}")
    print(f"Correctly Predicted: {correct_const}/{total_const} constituencies")
    print(f"Incorrectly Predicted: {total_const - correct_const}/{total_const} constituencies")
    
    # Show some correctly predicted constituencies
    print("\nSample Correctly Predicted Constituencies:")
    correct_preds = constituency_results[constituency_results['correct'] == 1].head(10)
    for _, row in correct_preds.iterrows():
        print(f"  {row['constituency']}: {row['actual_winner']}")
    
    # Show some incorrectly predicted constituencies
    print("\nSample Incorrectly Predicted Constituencies:")
    incorrect_preds = constituency_results[constituency_results['correct'] == 0].head(10)
    for _, row in incorrect_preds.iterrows():
        print(f"  {row['constituency']}: Actual={row['actual_winner']}, Predicted={row['predicted_winner']}")
    
    return const_accuracy

def main():
    """Main evaluation pipeline"""
    print("=" * 80)
    print("Tamil Nadu Election Winner Prediction - Model Evaluation")
    print("=" * 80)
    
    # Load model and encoders
    print("\nLoading model and encoders...")
    
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded")
    
    with open('label_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)
    print("✓ Label encoder loaded")
    
    with open('constituency_encoder.pkl', 'rb') as f:
        const_encoder = pickle.load(f)
    print("✓ Constituency encoder loaded")
    
    with open('party_encoder.pkl', 'rb') as f:
        party_encoder = pickle.load(f)
    print("✓ Party encoder loaded")
    
    # Load processed data
    print("\nLoading processed election data...")
    df = pd.read_csv('data/processed_election_data.csv')
    print(f"✓ Loaded {len(df)} records")
    
    # Create features
    features_df = create_features(df)
    
    # Prepare data
    X, y, features_df = prepare_data(features_df, const_encoder, party_encoder, target_encoder)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X)
    
    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total samples evaluated: {len(y)}")
    print(f"Correctly classified: {(y == y_pred).sum()}")
    print(f"Incorrectly classified: {(y != y_pred).sum()}")
    
    # Get class labels
    class_labels = target_encoder.classes_
    
    # Print confusion matrix
    cm = confusion_matrix(y, y_pred)
    print_confusion_matrix(cm, class_labels, top_n=10)
    
    # Print classification report
    print("\n" + "=" * 80)
    print("Classification Report (Top 10 Classes)")
    print("=" * 80)
    
    # Get top 10 classes by frequency
    class_counts = cm.sum(axis=1)
    top_10_indices = np.argsort(class_counts)[-10:][::-1]
    top_10_labels = [class_labels[i] for i in top_10_indices]
    
    report = classification_report(
        y, y_pred, 
        labels=top_10_indices,
        target_names=top_10_labels,
        digits=4,
        zero_division=0
    )
    print(report)
    
    # Evaluate at constituency level
    const_accuracy = evaluate_by_constituency(features_df, y, y_pred, target_encoder)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Party-Level Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Constituency-Level Accuracy: {const_accuracy:.4f} ({const_accuracy*100:.2f}%)")
    print("\nNote: Constituency-level accuracy shows how many constituencies")
    print("      have the correct winner predicted (more meaningful metric).")
    print("=" * 80)

if __name__ == "__main__":
    main()