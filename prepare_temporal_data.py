import dask.dataframe as dd
import pandas as pd
import os

# --- Configuration ---
CERT_DATA_PATH = 'C:/Users/dipto/Downloads/CERT_Dataset/r4.2'
OUTPUT_FILE = 'temporal_events.csv'

# --- Main Script ---
if __name__ == "__main__":
    print("Starting data preparation for the advanced temporal model...")

    # --- 1. Load the relevant datasets ---
    # We only need user, date, and activity type
    print("Loading logon data...")
    logon_df = dd.read_csv(os.path.join(CERT_DATA_PATH, 'logon.csv'), usecols=['user', 'date', 'activity'], blocksize='64MB')
    
    print("Loading device data...")
    device_df = dd.read_csv(os.path.join(CERT_DATA_PATH, 'device.csv'), usecols=['user', 'date', 'activity'], blocksize='64MB')
    
    print("Loading email data...")
    # For email, the 'activity' is implicitly 'sent_email'
    email_df = dd.read_csv(os.path.join(CERT_DATA_PATH, 'email.csv'), usecols=['user', 'date'], blocksize='64MB')
    email_df['activity'] = 'sent_email'
    
    # --- 2. Combine all events into one dataframe ---
    print("Combining all event dataframes...")
    combined_df = dd.concat([logon_df, device_df, email_df])
    
    # --- 3. Convert date to datetime and set as index ---
    # This step prepares the data for a time-based sort without loading it all into memory.
    print("Converting to datetime and preparing for sort...")
    combined_df['datetime'] = dd.to_datetime(combined_df['date'])
    time_indexed_df = combined_df.set_index('datetime')

    # --- 4. Save the processed data to a new file ---
    # Dask will handle the sorting and writing partition-by-partition, which is memory-safe.
    # This is the most computationally intensive step.
    print(f"\nSaving the processed and sorted temporal data to {OUTPUT_FILE}...")
    # We select the final columns we need
    final_cols = time_indexed_df[['user', 'activity']]
    
    # Dask's to_csv can write to a single file and handle the large data efficiently.
    # This will take a significant amount of time.
    final_cols.to_csv(OUTPUT_FILE, single_file=True, header=True)

    print("\nData preparation complete!")
    print(f"The file '{OUTPUT_FILE}' is now ready to be uploaded to Google Colab.")

    # We can't easily preview the head now without loading the large file,
    # so we'll inspect it after it's created.
    print("You can inspect the first few lines of the CSV file manually to preview it.")

