import dask.dataframe as dd
import pandas as pd
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import IsolationForest

# --- Configuration ---
CERT_DATA_PATH = 'C:/Users/dipto/Downloads/CERT_Dataset/r4.2'
ANSWERS_PATH = 'C:/Users/dipto/Downloads/CERT_Dataset/answers' # Path to the 'answers' folder

# --- Feature Calculation Functions ---

def get_logon_counts(base_path):
    """Calculates total and after-hours logon counts."""
    print("Calculating logon features...")
    ddf = dd.read_csv(os.path.join(base_path, 'logon.csv'))
    ddf['datetime'] = dd.to_datetime(ddf['date'])
    
    total_logons = ddf.groupby('user').id.count()
    
    after_hours_mask = (ddf['datetime'].dt.hour < 8) | (ddf['datetime'].dt.hour >= 18)
    after_hours_logons = ddf[after_hours_mask].groupby('user').id.count()
    
    return total_logons.compute(), after_hours_logons.compute()

def get_device_counts(base_path):
    """Calculates device connection counts."""
    print("Calculating device connection features...")
    ddf = dd.read_csv(os.path.join(base_path, 'device.csv'))
    connections = ddf[ddf['activity'] == 'Connect'].groupby('user').id.count()
    return connections.compute()

def get_sentiment_scores(base_path):
    """Calculates average email sentiment."""
    print("Calculating sentiment features...")
    ddf = dd.read_csv(os.path.join(base_path, 'email.csv'), usecols=['user', 'content'], dtype={'content': 'string'})
    
    def get_sentiment(df_partition):
        analyzer = SentimentIntensityAnalyzer()
        # Ensure content is a string to prevent errors
        df_partition['sentiment'] = df_partition['content'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])
        return df_partition
        
    sentiment_ddf = ddf.map_partitions(get_sentiment, meta={'user': 'string', 'content': 'string', 'sentiment': 'float64'})
    mean_sentiment = sentiment_ddf.groupby('user').sentiment.mean()
    return mean_sentiment.compute()

# --- Main Script ---
if __name__ == "__main__":
    # --- 1. Calculate all features ---
    total_logons, after_hours_logons = get_logon_counts(CERT_DATA_PATH)
    device_connections = get_device_counts(CERT_DATA_PATH)
    sentiment_scores = get_sentiment_scores(CERT_DATA_PATH)

    # --- 2. Create the Master User Profile ---
    print("\nBuilding master user profile...")
    master_profile = pd.DataFrame({
        'total_logons': total_logons,
        'after_hours_logons': after_hours_logons,
        'device_connections': device_connections,
        'sentiment': sentiment_scores
    })
    master_profile = master_profile.fillna(0)
    
    print("Master profile created successfully!")
    print(master_profile.head())

    # --- 3. Train the Isolation Forest Model ---
    print("\nTraining Isolation Forest model...")
    model = IsolationForest(contamination='auto', random_state=42)
    model.fit(master_profile)
    master_profile['anomaly_score'] = model.decision_function(master_profile)
    print("Model training and prediction complete!")

    # --- 4. Evaluate the Results ---
    insiders_df = pd.read_csv(os.path.join(ANSWERS_PATH, 'insiders.csv'))
    
    # Find the correct user ID column
    user_col = 'user' # We confirmed this is the correct column name
    
    # Mark the true insiders in our profile
    master_profile['is_insider'] = master_profile.index.isin(insiders_df[user_col])
    
    print("\n--- Top 10 Most Anomalous Users ---")
    top_anomalies = master_profile.sort_values('anomaly_score').head(10)
    print(top_anomalies)

    insiders_caught = top_anomalies['is_insider'].sum()
    print(f"\nFound {insiders_caught} true insider(s) in the top 10 anomalies.")

    # --- 5. Save the results to a file for the dashboard ---
    print("\nSaving results to model_results.csv...")
    top_anomalies.to_csv("model_results.csv")
    print("Results saved successfully!")

