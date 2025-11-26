import pandas as pd
from dateutil import parser
from datetime import datetime
import os

# --- Configuration ---
# NOTE: Replace 'your_input_file.csv' with the actual path to your 290k-row CSV file.
INPUT_FILENAME = 'sp500_news_290k_articles.csv'
OUTPUT_FILENAME = 'sp500_news_290k_articles_cleaned.csv'
# Replace 'Date' with the exact name of your date column.
DATE_COLUMN_NAME = 'date'
# ---------------------


def parse_mixed_dates_explicit(date_str):
    """
    Tries to parse two specific date formats using datetime.strptime.
    Returns a datetime object if successful, or pd.NaT if parsing fails.
    """
    if pd.isna(date_str) or str(date_str).strip() == '':
        return pd.NaT
    date_str = str(date_str)
    
    # 1. Try Format: Abbreviated (e.g., 'Jan-21-22')
    try:
        return datetime.strptime(date_str, '%b-%d-%y')
    except ValueError:
        pass

    # 2. Try Format: ISO 8601 (e.g., '2022-01-14T00:00:00.000Z')
    try:
        # Clean the string by removing 'T' and 'Z' for strptime compatibility
        if date_str.endswith('Z') and 'T' in date_str:
             # Convert '2022-01-14T00:00:00.000Z' to '2022-01-14 00:00:00.000'
             date_str_clean = date_str.replace('T', ' ').rstrip('Z')
             # Use %f for the milliseconds part
             return datetime.strptime(date_str_clean, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        pass
        
    return pd.NaT # Return Not a Time if neither format works

def process_date_column(df, column_name):
    """
    Processes a column with mixed date formats using row-by-row explicit parsing
    and ensures the resulting column is a consistent datetime type.
    """
    print(f"Starting explicit processing for column: '{column_name}'...")
    print(f"Note: Row-by-row processing via .apply is slower for 290k rows, but necessary.")
    
    # 1. Apply the explicit parsing function row by row
    df['datetime_obj'] = df[column_name].apply(parse_mixed_dates_explicit)
    
    # 2. Crucial Step: Explicitly convert the resulting column back to a datetime Series.
    # This guarantees the dtype is consistent, resolving the '.dt accessor' error.
    df['datetime_obj'] = pd.to_datetime(df['datetime_obj'], errors='coerce')
    
    print("Date parsing complete.")

    # 3. Extract the components from the new datetime object column
    # Using 'Int64' ensures that missing/unparsable dates (NaT) result in NaN.
    df['day'] = df['datetime_obj'].dt.day.astype('Int64')
    df['month'] = df['datetime_obj'].dt.month.astype('Int64')
    df['year'] = df['datetime_obj'].dt.year.astype('Int64')
    
    # Clean up the temporary datetime object column
    df = df.drop(columns=['datetime_obj'])
    
    return df

def main():
    """Main function to load, process, and save the data."""
    if not os.path.exists(INPUT_FILENAME):
        print(f"ERROR: Input file '{INPUT_FILENAME}' not found.")
        print("Please replace 'your_input_file.csv' with your actual file name and run again.")
        return

    try:
        # Read the CSV file
        df = pd.read_csv(INPUT_FILENAME)
        print(f"File '{INPUT_FILENAME}' loaded successfully. Rows: {len(df)}")
        
        # Check if the required column exists
        if DATE_COLUMN_NAME not in df.columns:
            print(f"ERROR: Column '{DATE_COLUMN_NAME}' not found in the file.")
            print("Please check the column name and update the DATE_COLUMN_NAME variable.")
            print(f"Available columns: {list(df.columns)}")
            return

        # Process the data
        df_processed = process_date_column(df, DATE_COLUMN_NAME)

        # Print a sample of the results
        print("\n--- Sample of Processed Data (First 5 rows) ---")
        print(df_processed[[DATE_COLUMN_NAME, 'day', 'month', 'year']].head())

        # Save the output
        df_processed.to_csv(OUTPUT_FILENAME, index=False)
        print(f"\nProcessing complete! The new file is saved as '{OUTPUT_FILENAME}'.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()