import pandas as pd
import re

def standardize_constituency_name(name):
    """Standardize constituency names for consistent matching"""
    if pd.isna(name):
        return name
    
    # Strip whitespace
    name = str(name).strip()
    
    # Remove extra quotes
    name = name.replace('"', '')
    
    # Normalize multiple spaces to single space
    name = re.sub(r'\s+', ' ', name)
    
    # Add space after Dr. if missing
    name = re.sub(r'Dr\.([A-Z])', r'Dr. \1', name)
    
    # Title case
    name = name.title()
    
    return name

def standardize_party_name(name):
    """Standardize party names for consistent tracking"""
    if pd.isna(name):
        return name
    
    # Strip whitespace
    name = str(name).strip()
    
    # Remove trailing commas
    name = name.rstrip(',')
    
    # Normalize multiple spaces to single space
    name = re.sub(r'\s+', ' ', name)
    
    return name

def process_2016_election(filepath):
    """
    Process 2016 election data
    Returns: list of tuples (constituency, year, party, votes, total_votes)
    """
    print("Processing 2016 election data...")
    
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Get column names
    columns = df.columns.tolist()
    
    # Find the index where summary data starts (after "Ind Vote")
    summary_start_idx = None
    for idx, col in enumerate(columns):
        if 'Ind Vote' in str(col) or col == 'Ind Vote':
            summary_start_idx = idx
            break
    
    results = []
    
    # Process each constituency row
    for _, row in df.iterrows():
        # Extract constituency name (first column)
        constituency = standardize_constituency_name(row.iloc[0])
        
        # Extract total votes (last column named "Total")
        total_votes = None
        if 'Total' in columns:
            total_votes = row['Total']
        
        # Skip if no valid constituency or total
        if pd.isna(constituency) or pd.isna(total_votes):
            continue
        
        total_votes = int(total_votes)
        
        # Iterate through columns to find party-vote pairs
        i = 1  # Start after constituency column
        while i < len(columns) - 1:
            # Stop at summary section
            if summary_start_idx and i >= summary_start_idx:
                break
            
            party_col = columns[i]
            next_col = columns[i + 1] if i + 1 < len(columns) else None
            
            # Check if this is a valid party column (next column should be "Votes")
            if next_col == 'Votes':
                party_name = standardize_party_name(party_col)
                votes_value = row.iloc[i + 1]
                
                # Skip NOTA and invalid parties
                if party_name and 'None of the Above' not in party_name and party_name != 'None Of The Above':
                    # Only add if votes are present (not empty)
                    if pd.notna(votes_value) and str(votes_value).strip() != '':
                        try:
                            votes = int(votes_value)
                            results.append((constituency, 2016, party_name, votes, total_votes))
                        except (ValueError, TypeError):
                            pass
                
                # Move to next party (skip the "Votes" column)
                i += 2
            else:
                i += 1
    
    print(f"Extracted {len(results)} records from 2016 data")
    return results

def process_2021_election(filepath):
    """
    Process 2021 election data
    Returns: list of tuples (constituency, year, party, votes, total_votes)
    """
    print("Processing 2021 election data...")
    
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Get column names
    columns = df.columns.tolist()
    
    results = []
    
    # Process each constituency row
    for _, row in df.iterrows():
        # Extract constituency information
        constituency = standardize_constituency_name(row.iloc[1])  # Column 1: Constituency Name
        total_votes = row.iloc[2]  # Column 2: Total
        
        # Skip if no valid constituency or total
        if pd.isna(constituency) or pd.isna(total_votes):
            continue
        
        total_votes = int(total_votes)
        
        # Iterate through columns to find party-EVM-Postal triplets
        i = 3  # Start after Constituency Id, Name, Total
        while i < len(columns) - 2:
            party_col = columns[i]
            next_col1 = columns[i + 1] if i + 1 < len(columns) else None
            next_col2 = columns[i + 2] if i + 2 < len(columns) else None
            
            # Check if this is a valid party column (next columns should be "EVM Votes", "PostalVotes")
            if next_col1 == 'EVM Votes' and next_col2 == 'PostalVotes':
                party_name = standardize_party_name(party_col)
                evm_votes = row.iloc[i + 1]
                postal_votes = row.iloc[i + 2]
                
                # Skip NOTA, empty parties, and ALL IND
                if party_name and 'None of the Above' not in party_name and party_name != 'Nota' and 'ALL  IND' not in party_name:
                    # Calculate total votes (EVM + Postal)
                    evm = 0 if pd.isna(evm_votes) or str(evm_votes).strip() == '' else int(evm_votes)
                    postal = 0 if pd.isna(postal_votes) or str(postal_votes).strip() == '' else int(postal_votes)
                    votes = evm + postal
                    
                    # Only add if party actually got votes
                    if votes > 0:
                        results.append((constituency, 2021, party_name, votes, total_votes))
                
                # Move to next party (skip EVM Votes and PostalVotes columns)
                i += 3
            else:
                # Stop if we hit "ALL IND" or empty headers
                if party_col and ('ALL  IND' in str(party_col) or str(party_col).strip() == ''):
                    break
                i += 1
    
    print(f"Extracted {len(results)} records from 2021 data")
    return results

def calculate_winning_party(df):
    """
    Calculate winning party for each constituency-year combination
    Returns: DataFrame with winning_party column added
    """
    print("Calculating winning parties...")
    
    # Group by constituency and year, find party with max votes
    winning_parties = df.loc[df.groupby(['constituency', 'year'])['votes'].idxmax()]
    winning_parties = winning_parties[['constituency', 'year', 'party']].rename(columns={'party': 'winning_party'})
    
    # Merge winning party back to original dataframe
    df = df.merge(winning_parties, on=['constituency', 'year'], how='left')
    
    return df

def main():
    """Main preprocessing pipeline"""
    print("=" * 60)
    print("Tamil Nadu Election Data Preprocessing")
    print("=" * 60)
    
    # Process 2016 election data
    results_2016 = process_2016_election('data/tn_2016_election.csv')
    
    # Process 2021 election data
    results_2021 = process_2021_election('data/tn_2021_election.csv')
    
    # Combine all results
    all_results = results_2016 + results_2021
    
    # Create DataFrame
    print("\nCreating combined DataFrame...")
    df = pd.DataFrame(all_results, columns=['constituency', 'year', 'party', 'votes', 'total_votes'])
    
    # Calculate winning party for each constituency-year
    df = calculate_winning_party(df)
    
    # Validation
    print("\n" + "=" * 60)
    print("Data Validation")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    print(f"Unique constituencies (2016): {df[df['year'] == 2016]['constituency'].nunique()}")
    print(f"Unique constituencies (2021): {df[df['year'] == 2021]['constituency'].nunique()}")
    print(f"Unique parties (2016): {df[df['year'] == 2016]['party'].nunique()}")
    print(f"Unique parties (2021): {df[df['year'] == 2021]['party'].nunique()}")
    print(f"Missing values: {df.isna().sum().sum()}")
    
    # Check if votes <= total_votes
    invalid_votes = df[df['votes'] > df['total_votes']]
    if len(invalid_votes) > 0:
        print(f"WARNING: {len(invalid_votes)} records where party votes > total votes")
    else:
        print("✓ All party votes <= total votes")
    
    # Save to CSV
    output_path = 'data/processed_election_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Processed data saved to {output_path}")
    print("=" * 60)
    
    # Display sample
    print("\nSample data:")
    print(df.head(10))

if __name__ == "__main__":
    main()