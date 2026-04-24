"""
Script to download and prepare the Coat Shopping dataset
Dataset source: https://www.cs.cornell.edu/~schnabts/mnar/
"""

import urllib.request
import os
import numpy as np
import pandas as pd

def download_coat_dataset():
    """Download Coat dataset from Cornell website"""
    
    # Create data directory if it doesn't exist
    data_dir = '../data/coat_data'
    os.makedirs(data_dir, exist_ok=True)
    
    # URLs for the Coat dataset files
    base_url = "https://www.cs.cornell.edu/~schnabts/mnar/"
    files = {
        'train.ascii': 'train.ascii',
        'test.ascii': 'test.ascii'
    }
    
    print("Downloading Coat Shopping dataset...")
    print("="*60)
    
    for filename, local_name in files.items():
        url = base_url + filename
        local_path = os.path.join(data_dir, local_name)
        
        if os.path.exists(local_path):
            print(f"✓ {filename} already exists, skipping download")
        else:
            try:
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, local_path)
                print(f"✓ Successfully downloaded {filename}")
            except Exception as e:
                print(f"✗ Error downloading {filename}: {e}")
                print(f"  Please manually download from: {url}")
                print(f"  And save to: {local_path}")
    
    print("="*60)
    print("Download complete!")
    return data_dir

def load_coat_ascii(filepath):
    """
    Load Coat dataset ASCII file and convert to DataFrame
    
    Format: ASCII matrix where:
    - Rows = users
    - Columns = items
    - Values = ratings (0 = missing)
    """
    print(f"Loading {filepath}...")
    
    # Read the ASCII file
    matrix = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                row = [float(x) for x in line.strip().split()]
                matrix.append(row)
    
    matrix = np.array(matrix)
    print(f"  Matrix shape: {matrix.shape} (users x items)")
    
    # Convert to DataFrame format (userId, itemId, rating)
    ratings_list = []
    n_users, n_items = matrix.shape
    
    for user_idx in range(n_users):
        for item_idx in range(n_items):
            rating = matrix[user_idx, item_idx]
            if rating > 0:  # Only include non-zero ratings
                ratings_list.append({
                    'userId': user_idx,
                    'itemId': item_idx,
                    'rating': rating
                })
    
    df = pd.DataFrame(ratings_list)
    print(f"  Total ratings: {len(df)}")
    print(f"  Unique users: {df['userId'].nunique()}")
    print(f"  Unique items: {df['itemId'].nunique()}")
    
    return df

if __name__ == "__main__":
    # Download dataset
    data_dir = download_coat_dataset()
    
    # Load and convert to CSV format
    train_path = os.path.join(data_dir, 'train.ascii')
    test_path = os.path.join(data_dir, 'test.ascii')
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("\nConverting ASCII files to CSV format...")
        print("="*60)
        
        # Load train (biased) and test (unbiased) data
        train_df = load_coat_ascii(train_path)
        test_df = load_coat_ascii(test_path)
        
        # Save as CSV
        train_csv = os.path.join(data_dir, 'train.csv')
        test_csv = os.path.join(data_dir, 'test.csv')
        
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        
        print("\n" + "="*60)
        print("Dataset preparation complete!")
        print(f"Train CSV: {train_csv}")
        print(f"Test CSV: {test_csv}")
        print("\nDataset Statistics:")
        print(f"Train - Users: {train_df['userId'].nunique()}, Items: {train_df['itemId'].nunique()}, Ratings: {len(train_df)}")
        print(f"Test  - Users: {test_df['userId'].nunique()}, Items: {test_df['itemId'].nunique()}, Ratings: {len(test_df)}")
    else:
        print("\n⚠ Warning: Some files are missing. Please download manually if needed.")


