import dask.dataframe as dd
import pandas as pd
import numpy as np
import glob
import os
import time

# When reading the CSV files, specify dtypes
dtypes = {
    'STATION': 'object',
    'DATE': 'object',  # Start as string, convert to datetime later
    'LATITUDE': 'float64',
    'LONGITUDE': 'float64',
    'ELEVATION': 'float64',
    'NAME': 'object',
    'PRCP': 'object',  # Keep as object initially to handle missing values
    'PRCP_ATTRIBUTES': 'object',
    'TMAX': 'object',
    'TMAX_ATTRIBUTES': 'object',
    'TMIN': 'object',
    'TMIN_ATTRIBUTES': 'object',
    'TAVG': 'object',
    'TAVG_ATTRIBUTES': 'object',
    # If SNWD columns exist in some files:
    'SNWD': 'object',
    'SNWD_ATTRIBUTES': 'object',
}

# 1. Load Ukrainian station data
ukrainian_files = glob.glob('D:/daily-summaries-latest-upm/*.csv')
print(f"Found {len(ukrainian_files)} data files")

# Limit to 100 files max
max_files = 50
ukrainian_files = ukrainian_files[:max_files]
print(f"Reading {len(ukrainian_files)} files")

# 2. Read and combine files
ddf = dd.read_csv(ukrainian_files, dtype=dtypes, assume_missing=True)
print(f"Data loaded with {len(ddf.columns)} columns")

# 3. Apply preprocessing to sample for metadata
def preprocess(df):
    df = df.copy()
    
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Ensure all expected columns exist, filling with NaN if necessary
    expected_columns = ['PRCP', 'TMAX', 'TMIN', 'TAVG', 'SNWD']
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan  # Ensure column exists
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'TMAX' in df.columns and 'TMIN' in df.columns:
        df['TEMP_RANGE'] = df['TMAX'] - df['TMIN']
    
    df['MONTH'] = df['DATE'].dt.month
    df['DAY'] = df['DATE'].dt.day
    df['YEAR'] = df['DATE'].dt.year
    df['DAY_OF_YEAR'] = df['DATE'].dt.dayofyear
    df['SEASON'] = ((df['MONTH'] % 12) // 3 + 1).astype(int)

    cols_to_drop = [col for col in df.columns if '_ATTRIBUTES' in col]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Ensure column order matches the metadata
    expected_order = ['STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 
                      'PRCP', 'SNWD', 'TMAX', 'TMIN', 'TAVG', 'TEMP_RANGE', 
                      'MONTH', 'DAY', 'YEAR', 'DAY_OF_YEAR', 'SEASON']
    
    df = df.reindex(columns=expected_order)

    return df


# Create metadata by applying function to the first partition
meta_df = preprocess(ddf.partitions[0].head())

# 4. Apply preprocessing with metadata
print("Preprocessing data...")
ddf = ddf.map_partitions(preprocess, meta=meta_df)

# 5. Convert Dask dataframe to Pandas for further processing
# This will load all data into memory (ensure you have enough RAM)
print("Converting to pandas dataframe...")
df = ddf.compute()

# 6. Create a time-indexed dataset
# Group by station and apply resampling
print("Resampling data by station...")
processed_stations = []

for station_id, station_df in df.groupby('STATION'):
    # Process each station's data
    try:
        # Sort by date
        station_df = station_df.sort_values('DATE')
        
        # Basic quality check - ensure we have enough data
        if len(station_df) < 365:  # At least a year of data
            continue
            
        # Get station metadata that won't change over time
        station_meta = {
            'STATION': station_id,
            'NAME': station_df['NAME'].iloc[0],
            'LATITUDE': station_df['LATITUDE'].iloc[0],
            'LONGITUDE': station_df['LONGITUDE'].iloc[0],
            'ELEVATION': station_df['ELEVATION'].iloc[0]
        }
        
        # Set date as index
        station_df = station_df.set_index('DATE')
        
        # Remove duplicate indices if any
        station_df = station_df[~station_df.index.duplicated(keep='first')]
        
        # Resample to daily frequency, filling missing values
        resampled_df = station_df.resample('D').first()
        
        # For numeric columns, apply interpolation for small gaps, then forward/backward fill
        numeric_cols = ['PRCP', 'TMAX', 'TMIN', 'TAVG', 'TEMP_RANGE', 'SNWD']
        numeric_cols = [col for col in numeric_cols if col in resampled_df.columns]
        
        # Interpolate for small gaps (up to 3 days)
        resampled_df[numeric_cols] = resampled_df[numeric_cols].interpolate(method='linear', limit=3)
        
        # Fill remaining gaps with forward then backward fill
        resampled_df[numeric_cols] = resampled_df[numeric_cols].ffill().bfill()
        
        # Add back station metadata
        for key, value in station_meta.items():
            resampled_df[key] = value
            
        # Add to list of processed stations
        processed_stations.append(resampled_df)
        print(f"Processed station {station_id} with {len(resampled_df)} days of data")
        
    except Exception as e:
        print(f"Error processing station {station_id}: {e}")

# Combine all processed stations
print("Combining processed stations...")
if processed_stations:
    processed_df = pd.concat(processed_stations)
    print(f"Combined dataset has {len(processed_df)} rows")
else:
    raise ValueError("No stations were successfully processed")

# 7. Split data into training/validation/test sets
print("Splitting data into train/validation/test sets...")

# Create a function to prepare data for time series forecasting
def prepare_time_series_data(df, target_col='TMAX', seq_length=14, forecast_horizon=7):
    """
    Prepare data for time series forecasting.
    
    Args:
        df: DataFrame with time series data
        target_col: Column to predict
        seq_length: Number of days to use as input
        forecast_horizon: Number of days to predict
    
    Returns:
        X: Input sequences
        y: Target sequences
    """
    # Make sure data is sorted by date
    df = df.sort_index()
    
    # For each station, create sequences
    station_X = []
    station_y = []
    
    for station_id, station_df in df.groupby('STATION'):
        # Select only numeric features
        numeric_df = station_df.select_dtypes(include=[np.number])
        
        # Drop any rows with NaN
        numeric_df = numeric_df.dropna()
        numeric_df = numeric_df.astype(np.float32)
        
        if len(numeric_df) <= seq_length + forecast_horizon:
            continue
            
        # Create sequences
        for i in range(len(numeric_df) - seq_length - forecast_horizon + 1):
            # Input sequence
            X_seq = numeric_df.iloc[i:i+seq_length]
            
            # Target sequence
            y_seq = numeric_df[target_col].iloc[i+seq_length:i+seq_length+forecast_horizon]
            
            station_X.append(np.array(X_seq.values, dtype=np.float32))
            station_y.append(np.array(y_seq.values, dtype=np.float32))
    
    return np.array(station_X), np.array(station_y)

# For each station, perform temporal split
train_data = {}
val_data = {}
test_data = {}

for station_id, station_df in processed_df.groupby('STATION'):
    # Calculate split points (e.g., 70% train, 15% validation, 15% test)
    n = len(station_df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    # Sort by date
    station_df = station_df.sort_index()
    
    train = station_df.iloc[:train_end]
    val = station_df.iloc[train_end:val_end]
    test = station_df.iloc[val_end:]
    
    train_data[station_id] = train
    val_data[station_id] = val
    test_data[station_id] = test

# 8. Normalize data (for each station separately)
print("Normalizing data...")
train_norm = {}
val_norm = {}
test_norm = {}
normalization_params = {}

for station_id in train_data.keys():
    # Get numeric columns for this station
    numeric_cols = train_data[station_id].select_dtypes(include=[np.number]).columns
    
    # Calculate mean and std from training data only
    means = train_data[station_id][numeric_cols].mean()
    stds = train_data[station_id][numeric_cols].std()
    stds = stds.replace(0, 1)  # Avoid division by zero
    
    # Store normalization parameters
    normalization_params[station_id] = {'mean': means, 'std': stds}
    
    # Apply normalization to all sets
    train_numeric = (train_data[station_id][numeric_cols] - means) / stds
    val_numeric = (val_data[station_id][numeric_cols] - means) / stds
    test_numeric = (test_data[station_id][numeric_cols] - means) / stds
    
    # Copy normalized values back to original dataframes
    train_norm[station_id] = train_data[station_id].copy()
    val_norm[station_id] = val_data[station_id].copy()
    test_norm[station_id] = test_data[station_id].copy()
    
    train_norm[station_id][numeric_cols] = train_numeric
    val_norm[station_id][numeric_cols] = val_numeric
    test_norm[station_id][numeric_cols] = test_numeric


# 9. Create sequences for time series forecasting
print("Creating sequences for forecasting...")
start_time = time.time()

# Choose target variable
target_col = 'TMAX'  # Can be changed to TMIN, PRCP, etc.
seq_length = 14  # Use 14 days of history
forecast_horizon = 7  # Predict 7 days ahead

# Combine normalized data from all stations for each split
train_combined = pd.concat([df for df in train_norm.values()])
val_combined = pd.concat([df for df in val_norm.values()])
test_combined = pd.concat([df for df in test_norm.values()])

print("Processing training data...")
X_train, y_train = prepare_time_series_data(train_combined, target_col, seq_length, forecast_horizon)
print(f"Training data processed in {time.time() - start_time:.2f} seconds")

print("Processing validation data...")
X_val, y_val = prepare_time_series_data(val_combined, target_col, seq_length, forecast_horizon)
print(f"Validation data processed in {time.time() - start_time:.2f} seconds")

print("Processing test data...")
X_test, y_test = prepare_time_series_data(test_combined, target_col, seq_length, forecast_horizon)
print(f"Test data processed in {time.time() - start_time:.2f} seconds")

print(f"Created {len(X_train)} training sequences, {len(X_val)} validation sequences, {len(X_test)} test sequences")

# 10. Save the prepared data
print("Saving prepared data...")
data_dir = "D:/Dev/python-projects/weather-forecasting/prepared_data_small"
os.makedirs(data_dir, exist_ok=True)

def save_in_batches(data, filename, batch_size=1000):
    """Save large arrays in batches to avoid memory errors."""
    n_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        
        batch = data[start_idx:end_idx]
        
        # For the first batch, create a new file
        if i == 0:
            np.save(filename, batch)
        else:
            # For subsequent batches, append to the existing file
            with open(filename, 'ab') as f:
                np.save(f, batch)
        
        # Free memory
        del batch
        
    print(f"Saved {len(data)} items to {filename} in {n_batches} batches")

# Save sequences
#save_in_batches(X_train, f'{data_dir}/X_train.npy')
#save_in_batches(X_val, f'{data_dir}/X_val.npy')
#save_in_batches(X_test, f'{data_dir}/X_test.npy')

np.save(f'{data_dir}/X_train.npy', X_train)
np.save(f'{data_dir}/X_val.npy', X_val)
np.save(f'{data_dir}/X_test.npy', X_test)

np.save(f'{data_dir}/y_train.npy', y_train)
np.save(f'{data_dir}/y_val.npy', y_val)
np.save(f'{data_dir}/y_test.npy', y_test)

# Save normalization parameters
pd.to_pickle(normalization_params, f'{data_dir}/normalization_params.pkl')

print("Data preparation complete!")
print("Final dataset shapes:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")











print(f"X_train dtype: {X_train.dtype}")
print(f"X_train sample type: {type(X_train[0])}")
print(f"X_train shape: {X_train.shape}")

print(f"X_train sample:\n{X_train[0]}")
print(f"X_train first element type: {type(X_train[0, 0, 0])}")

print(f"X_train element shapes: {[x.shape for x in X_train[:5]]}") 

print(X_train.dtype, X_val.dtype, X_test.dtype)
print(y_train.dtype, y_val.dtype, y_test.dtype)

