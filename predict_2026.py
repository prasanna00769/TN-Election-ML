import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

def extrapolate_2026_data(df):
    """
    Extrapolate 2026 data based on trends from 2016 and 2021
    This simulates what 2026 data might look like based on historical trends
    """
    print("Extrapolating 2026 data from historical trends...")
    
    # Get 2021 data as baseline
    df_2021 = df[df['year'] == 2021].copy()
    df_2016 = df[df['year'] == 2016].copy()
    
    # Create 2026 predictions
    predicted_2026 = []
    
    constituencies = df_2021['constituency'].unique()
    
    for constituency in constituencies:
        const_2021 = df_2021[df_2021['constituency'] == constituency]
        const_2016 = df_2016[df_2016['constituency'] == constituency]
        
        # Get total votes for extrapolation
        total_votes_2021 = const_2021.iloc[0]['total_votes']
        
        # Calculate turnout trend
        if len(const_2016) > 0:
            total_votes_2016 = const_2016.iloc[0]['total_votes']
            turnout_growth_rate = (total_votes_2021 - total_votes_2016) / total_votes_2016
            # Extrapolate total votes for 2026 (5 year gap)
            total_votes_2026 = int(total_votes_2021 * (1 + turnout_growth_rate))
        else:
            # If no 2016 data, assume 2% growth
            total_votes_2026 = int(total_votes_2021 * 1.02)
        
        # Get parties that contested in 2021
        parties_2021 = const_2021['party'].unique()
        
        for party in parties_2021:
            party_2021 = const_2021[const_2021['party'] == party].iloc[0]
            party_2016_data = const_2016[const_2016['party'] == party]
            
            # Calculate vote trend
            votes_2021 = party_2021['votes']
            
            if len(party_2016_data) > 0:
                votes_2016 = party_2016_data.iloc[0]['votes']
                # Calculate growth rate
                if votes_2016 > 0:
                    vote_growth_rate = (votes_2021 - votes_2016) / votes_2016
                    # Extrapolate votes for 2026
                    votes_2026 = int(votes_2021 * (1 + vote_growth_rate))
                else:
                    # Party was new in 2021, assume continued growth
                    votes_2026 = int(votes_2021 * 1.1)
            else:
                # Party didn't contest in 2016, assume moderate growth
                votes_2026 = int(votes_2021 * 1.05)
            
            # Ensure votes don't exceed total or go negative
            votes_2026 = max(0, min(votes_2026, total_votes_2026))
            
            predicted_2026.append({
                'constituency': constituency,
                'year': 2026,
                'party': party,
                'votes': votes_2026,
                'total_votes': total_votes_2026,
                'winning_party': 'Unknown'  # Will be calculated later
            })
    
    df_2026 = pd.DataFrame(predicted_2026)
    
    # Normalize votes so they sum to total_votes per constituency
    for constituency in constituencies:
        const_data = df_2026[df_2026['constituency'] == constituency]
        votes_sum = const_data['votes'].sum()
        total_votes = const_data.iloc[0]['total_votes']
        
        if votes_sum > 0:
            # Scale votes proportionally
            scale_factor = total_votes / votes_sum
            df_2026.loc[df_2026['constituency'] == constituency, 'votes'] = (
                df_2026.loc[df_2026['constituency'] == constituency, 'votes'] * scale_factor
            ).astype(int)
    
    print(f"Created {len(df_2026)} extrapolated records for 2026")
    
    return df_2026

def create_features_2026(df, df_2026):
    """
    Create features for 2026 prediction using historical data
    """
    print("Creating features for 2026 prediction...")
    
    features_list = []
    
    constituencies = df_2026['constituency'].unique()
    
    for constituency in constituencies:
        # Get historical data for this constituency
        const_data = df[df['constituency'] == constituency].copy()
        const_2026 = df_2026[df_2026['constituency'] == constituency].copy()
        
        parties = const_2026['party'].unique()
        
        for party in parties:
            # Get party data for 2016 and 2021 (historical)
            party_2016 = const_data[(const_data['party'] == party) & (const_data['year'] == 2016)]
            party_2021 = const_data[(const_data['party'] == party) & (const_data['year'] == 2021)]
            party_2026 = const_2026[const_2026['party'] == party]
            
            if len(party_2026) == 0:
                continue
            
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
            # Use actual 2021 data for features
            if len(party_2021) > 0:
                features['votes_2021'] = party_2021.iloc[0]['votes']
                features['total_votes_2021'] = party_2021.iloc[0]['total_votes']
                features['vote_share_2021'] = (party_2021.iloc[0]['votes'] / party_2021.iloc[0]['total_votes']) * 100
                features['was_winner_2021'] = 1 if party_2021.iloc[0]['party'] == party_2021.iloc[0]['winning_party'] else 0
                features['contested_2021'] = 1
            else:
                features['votes_2021'] = 0
                features['total_votes_2021'] = 0
                features['vote_share_2021'] = 0
                features['was_winner_2021'] = 0
                features['contested_2021'] = 0
            
            # --- TREND FEATURES ---
            if features['contested_2016'] == 1 and features['votes_2016'] > 0:
                features['vote_change_abs'] = features['votes_2021'] - features['votes_2016']
                features['vote_change_pct'] = ((features['votes_2021'] - features['votes_2016']) / features['votes_2016']) * 100
                features['vote_share_change'] = features['vote_share_2021'] - features['vote_share_2016']
            else:
                features['vote_change_abs'] = features['votes_2021']
                features['vote_change_pct'] = 100.0
                features['vote_share_change'] = features['vote_share_2021']
            
            # --- CONSECUTIVE WINS ---
            features['consecutive_wins'] = features['was_winner_2016'] + features['was_winner_2021']
            
            # --- CONSTITUENCY-LEVEL FEATURES (from 2021) ---
            const_2021 = const_data[const_data['year'] == 2021]
            if len(const_2021) > 0:
                features['num_parties_2021'] = len(const_2021)
                
                # Winner's vote share in 2021
                winner_2021 = const_2021[const_2021['party'] == const_2021.iloc[0]['winning_party']]
                if len(winner_2021) > 0:
                    features['winner_vote_share_2021'] = (winner_2021.iloc[0]['votes'] / winner_2021.iloc[0]['total_votes']) * 100
                else:
                    features['winner_vote_share_2021'] = 0
            else:
                features['num_parties_2021'] = 0
                features['winner_vote_share_2021'] = 0
            
            # Turnout change (2016 to 2021)
            const_2016 = const_data[const_data['year'] == 2016]
            if len(const_2016) > 0 and len(const_2021) > 0:
                total_2016 = const_2016.iloc[0]['total_votes']
                total_2021 = const_2021.iloc[0]['total_votes']
                features['turnout_change'] = ((total_2021 - total_2016) / total_2016) * 100 if total_2016 > 0 else 0
            else:
                features['turnout_change'] = 0
            
            # --- PARTY-LEVEL FEATURES (from 2021) ---
            party_wins_2021 = const_data[(const_data['year'] == 2021) & (const_data['party'] == party) & 
                                         (const_data['party'] == const_data['winning_party'])]
            features['party_seats_won_2021'] = len(party_wins_2021)
            
            party_contested_2021 = const_data[(const_data['year'] == 2021) & (const_data['party'] == party)]
            features['party_contested_constituencies_2021'] = len(party_contested_2021)
            
            if len(party_contested_2021) > 0:
                features['party_avg_vote_share_2021'] = party_contested_2021.apply(
                    lambda x: (x['votes'] / x['total_votes']) * 100, axis=1
                ).mean()
            else:
                features['party_avg_vote_share_2021'] = 0
            
            # --- COMPETITIVE POSITION FEATURES (from 2021) ---
            if len(const_2021) > 0:
                const_2021_sorted = const_2021.sort_values('votes', ascending=False).reset_index(drop=True)
                party_rank = const_2021_sorted[const_2021_sorted['party'] == party].index
                features['rank_2021'] = party_rank[0] + 1 if len(party_rank) > 0 else len(const_2021_sorted) + 1
                
                # Margin
                if features['was_winner_2021'] == 1 and len(const_2021_sorted) > 1:
                    features['margin_votes'] = const_2021_sorted.iloc[0]['votes'] - const_2021_sorted.iloc[1]['votes']
                elif len(const_2021_sorted) > 0:
                    winner_votes = const_2021_sorted.iloc[0]['votes']
                    this_party_votes = party_2021.iloc[0]['votes'] if len(party_2021) > 0 else 0
                    features['margin_votes'] = this_party_votes - winner_votes
                else:
                    features['margin_votes'] = 0
            else:
                features['rank_2021'] = 99
                features['margin_votes'] = 0
            
            # --- BINARY INDICATORS ---
            features['is_major_party'] = 1 if features['party_seats_won_2021'] >= 5 else 0
            
            national_parties = ['Bharatiya Janata Party', 'Indian National Congress', 
                              'Communist Party Of India', 'Communist Party Of India (Marxist)']
            features['is_national_party'] = 1 if party in national_parties else 0
            
            dravidian_parties = ['Dravida Munnetra Kazhagam', 'All India Anna Dravida Munnetra Kazhagam',
                               'Desiya Murpokku Dravida Kazhagam']
            features['is_dravidian_party'] = 1 if party in dravidian_parties else 0
            
            features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    print(f"Created {len(features_df)} feature records for 2026")
    
    return features_df

def prepare_prediction_data(features_df, const_encoder, party_encoder):
    """
    Prepare data for prediction
    """
    print("\nPreparing data for prediction...")
    
    # Encode constituency
    features_df['constituency_encoded'] = const_encoder.transform(features_df['constituency'])
    
    # Encode party
    features_df['party_encoded'] = party_encoder.transform(features_df['party'])
    
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
    
    return X

def main():
    """Main prediction pipeline for 2026"""
    print("=" * 80)
    print("Tamil Nadu 2026 Election Winner Prediction")
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
    
    # Load historical data
    print("\nLoading historical election data...")
    df = pd.read_csv('data/processed_election_data.csv')
    print(f"✓ Loaded {len(df)} historical records")
    
    # Extrapolate 2026 data
    df_2026 = extrapolate_2026_data(df)
    
    # Create features for 2026
    features_2026 = create_features_2026(df, df_2026)
    
    # Prepare data for prediction
    X_2026 = prepare_prediction_data(features_2026, const_encoder, party_encoder)
    
    # Make predictions
    print("\nMaking predictions for 2026...")
    y_pred = model.predict(X_2026)
    y_pred_proba = model.predict_proba(X_2026)
    
    # Add predictions to features dataframe
    features_2026['predicted_winner_encoded'] = y_pred
    features_2026['predicted_winner'] = target_encoder.inverse_transform(y_pred)
    features_2026['prediction_confidence'] = y_pred_proba.max(axis=1)
    
    # Determine actual winner for each constituency (party with most predicted probability)
    print("\nDetermining winners per constituency...")
    
    constituency_results = []
    
    for constituency in features_2026['constituency'].unique():
        const_predictions = features_2026[features_2026['constituency'] == constituency]
        
        # Group by predicted winner and sum probabilities (or pick highest confidence)
        winner_row = const_predictions.loc[const_predictions['prediction_confidence'].idxmax()]
        
        constituency_results.append({
            'constituency': constituency,
            'predicted_winning_party': winner_row['predicted_winner'],
            'confidence': winner_row['prediction_confidence'],
            'runner_up_party': const_predictions.nlargest(2, 'prediction_confidence').iloc[1]['party'] if len(const_predictions) > 1 else 'None',
            'num_parties_contested': len(const_predictions)
        })
    
    results_df = pd.DataFrame(constituency_results)
    
    # Save results
    output_path = 'data/predicted_2026_results.csv'
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS FOR 2026")
    print("=" * 80)
    
    # Summary statistics
    print(f"\nTotal constituencies: {len(results_df)}")
    print(f"\nPredicted seat distribution:")
    seat_distribution = results_df['predicted_winning_party'].value_counts()
    for party, seats in seat_distribution.items():
        percentage = (seats / len(results_df)) * 100
        print(f"  {party}: {seats} seats ({percentage:.1f}%)")
    
    print(f"\nAverage prediction confidence: {results_df['confidence'].mean():.4f}")
    print(f"Minimum confidence: {results_df['confidence'].min():.4f}")
    print(f"Maximum confidence: {results_df['confidence'].max():.4f}")
    
    # Show sample predictions
    print("\nSample Predictions (Top 10 by confidence):")
    print("-" * 80)
    top_predictions = results_df.nlargest(10, 'confidence')
    for _, row in top_predictions.iterrows():
        print(f"  {row['constituency']:<30} → {row['predicted_winning_party']:<40} ({row['confidence']:.3f})")
    
    print("\nSample Predictions (Bottom 10 by confidence - most uncertain):")
    print("-" * 80)
    bottom_predictions = results_df.nsmallest(10, 'confidence')
    for _, row in bottom_predictions.iterrows():
        print(f"  {row['constituency']:<30} → {row['predicted_winning_party']:<40} ({row['confidence']:.3f})")
    
    print("\n" + "=" * 80)
    print(f"✓ Predictions saved to {output_path}")
    print("=" * 80)
    
    print("\nNote: These predictions are based on extrapolated trends from 2016-2021")
    print("      Actual 2026 results will depend on many factors not captured in this model")
    print("      including alliances, candidates, campaigns, and current events.")
    print("=" * 80)

if __name__ == "__main__":
    main()