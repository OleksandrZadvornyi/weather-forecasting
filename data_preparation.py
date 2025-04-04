import dask.dataframe as dd
import pandas as pd
import numpy as np
import glob
import os
from datasets import Dataset, Features, Value, Sequence, DatasetDict
from gluonts.itertools import Map

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
ukrainian_files = glob.glob('D:/Dev/python-projects/weather-forecasting/daily-summaries-latest-upm/*.csv')
print(f"Found {len(ukrainian_files)} data files")

# Set files count limit
max_files = 20
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

def process_station_data(station_df):
    # Sort by date
    station_df = station_df.sort_values('DATE')
    
    # Basic quality check - ensure we have enough data
    if len(station_df) < 365:  # At least a year of data
        return None
        
    # Get station metadata that won't change over time
    station_meta = {
        'STATION': station_df['STATION'].iloc[0],
        'NAME': station_df['NAME'].iloc[0],
        'LATITUDE': station_df['LATITUDE'].iloc[0],
        'LONGITUDE': station_df['LONGITUDE'].iloc[0],
        'ELEVATION': station_df['ELEVATION'].iloc[0]
    }
    
    # Set date as index
    station_df = station_df.set_index('DATE')
    
    # Remove duplicate indices if any
    station_df = station_df[~station_df.index.duplicated(keep='first')]
    
    # Create a complete date range for the station
    date_range = pd.date_range(start=station_df.index.min(), end=station_df.index.max(), freq='D')
    
    # Resample to daily frequency, filling missing values
    resampled_df = station_df.reindex(date_range)
    
    # Numeric columns to interpolate
    numeric_cols = ['PRCP', 'TMAX', 'TMIN', 'TAVG', 'TEMP_RANGE', 'SNWD']
    numeric_cols = [col for col in numeric_cols if col in resampled_df.columns]
    
    # Enhanced interpolation strategy
    for col in numeric_cols:
        # Step 1: Seasonal interpolation with machine learning fallback
        def seasonal_interpolation(x):
            # First try linear interpolation within season
            interpolated = x.interpolate(method='linear', limit=7)
            
            # If still NaNs, use seasonal median
            if interpolated.isna().any():
                interpolated = interpolated.fillna(x.median())
            
            return interpolated
        
        resampled_df[col] = resampled_df.groupby(resampled_df.index.month)[col].transform(seasonal_interpolation)
        
        # Step 2: Advanced interpolation techniques
        if col in ['TMAX', 'TMIN', 'TAVG']:
            # Use rolling median for temperature
            window_size = 15  # Adjust based on your data
            rolling_median = resampled_df[col].rolling(window=window_size, center=True, min_periods=1).median()
            resampled_df[col] = resampled_df[col].fillna(rolling_median)
        
        elif col == 'PRCP':
            # Log interpolation with non-zero floor
            log_interpolated = np.log1p(resampled_df[col] + 0.01).interpolate(method='cubic', limit=10)
            resampled_df[col] = np.maximum(np.expm1(log_interpolated), 0)
        
        elif col == 'SNWD':
            # More robust interpolation for snow depth
            resampled_df[col] = resampled_df[col].interpolate(method='spline', order=3, limit=10)
        
        # Step 3: Final fallback strategies
        # Forward and backward fill with extended limits
        resampled_df[col] = resampled_df[col].ffill(limit=60)
        resampled_df[col] = resampled_df[col].bfill(limit=60)
        
        # Step 4: Last resort - replace remaining NaNs with global median
        global_median = station_df[col].median()
        resampled_df[col] = resampled_df[col].fillna(global_median)
    
    # Outlier and extreme value handling
    for col in numeric_cols:
        if col in ['TMAX', 'TMIN', 'TAVG']:
            # Remove extreme temperature outliers
            temp_mean = resampled_df[col].mean()
            temp_std = resampled_df[col].std()
            lower_bound = temp_mean - 3 * temp_std
            upper_bound = temp_mean + 3 * temp_std
            resampled_df.loc[(resampled_df[col] < lower_bound) | (resampled_df[col] > upper_bound), col] = global_median
        
        elif col == 'PRCP':
            # Ensure non-negative precipitation
            resampled_df.loc[resampled_df[col] < 0, col] = 0
    
    # Add back station metadata
    for key, value in station_meta.items():
        resampled_df[key] = value
    
    # Recalculate derived columns for ALL rows
    resampled_df = resampled_df.reset_index().rename(columns={'index': 'DATE'})
    resampled_df['MONTH'] = resampled_df['DATE'].dt.month
    resampled_df['DAY'] = resampled_df['DATE'].dt.day
    resampled_df['YEAR'] = resampled_df['DATE'].dt.year
    resampled_df['DAY_OF_YEAR'] = resampled_df['DATE'].dt.dayofyear
    resampled_df['SEASON'] = ((resampled_df['MONTH'] % 12) // 3 + 1).astype(int)
    
    # Recalculate TEMP_RANGE for ALL rows
    if 'TMAX' in resampled_df.columns and 'TMIN' in resampled_df.columns:
        resampled_df['TEMP_RANGE'] = resampled_df['TMAX'] - resampled_df['TMIN']
    
    # Final NaN check and global median replacement
    for col in numeric_cols:
        if resampled_df[col].isna().any():
            # print(f"Warning: Remaining NaNs in {col} column")
            global_median = station_df[col].median()
            resampled_df[col] = resampled_df[col].fillna(global_median)
    
    return resampled_df

# Create metadata by applying function to the first partition
meta_df = preprocess(ddf.partitions[0].head())

# 4. Apply preprocessing with metadata
print("Preprocessing data...")
ddf = ddf.map_partitions(preprocess, meta=meta_df)

# 5. Convert Dask dataframe to Pandas for further processing
# This will load all data into memory (ensure you have enough RAM)
print("Converting to pandas dataframe...")
df = ddf.compute()

# Drop rows with NaN values
print("Dropping rows with NaN values...")
print(f"Total rows: {len(df)}")
df = df.dropna(subset=['TMAX'])
print(f"Rows remaining after dropping NaNs: {len(df)}")

# 6. Create a time-indexed dataset
# Group by station and apply resampling
print("Resampling data by station...")
processed_stations = []
for station_id, station_df in df.groupby('STATION'):
    try:
        processed_station = process_station_data(station_df)
        if processed_station is not None:
            processed_stations.append(processed_station)
            print(f"Processed station {station_id} with {len(processed_station)} days of data")
    except Exception as e:
        print(f"Error processing station {station_id}: {e}")

# Combine all processed stations
print("Combining processed stations...")
if processed_stations:
    processed_df = pd.concat(processed_stations)
    print(f"Combined dataset has {len(processed_df)} rows")
else:
    raise ValueError("No stations were successfully processed")

# 7. Split data into training/validation/test sets based on time
print("Splitting data into train/validation/test sets...")

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

# 8. Prepare data for GluonTS format
# First, let's convert the data into the appropriate format for each split
print("Converting data to GluonTS format...")

def prepare_gluonts_data(data_dict, target_column='TMAX'):
    """
    Convert dictionary of dataframes to GluonTS dataset format.
    
    Args:
        data_dict: Dictionary of dataframes, keyed by station_id
        target_column: Column to use as target variable
    
    Returns:
        List of dictionaries in GluonTS format
    """
    gluonts_data = []
    
    # Create a mapping of station IDs to unique categorical indices
    station_ids = list(data_dict.keys())
    station_id_to_cat = {station_id: idx for idx, station_id in enumerate(station_ids)}
    
    for station_id, df in data_dict.items():
        # Sort by date
        df = df.sort_index()
        
        # Extract target variable
        target = df[target_column].values.astype(np.float32)
        
        # Get start timestamp
        start = df.index[0]
        
        # Create static categorical feature
        # Use the pre-mapped index to ensure it's within the expected range
        static_cat = [station_id_to_cat[station_id]]
        
        # Create optional static real features
        static_real = [
            df['LATITUDE'].iloc[0],
            df['LONGITUDE'].iloc[0],
            df['ELEVATION'].iloc[0]
        ]
        
        # Create dictionary in GluonTS format
        ts_data = {
            "start": start,
            "target": target,
            "feat_static_cat": static_cat,
            "feat_static_real": static_real,
            "item_id": station_id
        }
        
        gluonts_data.append(ts_data)
    
    return gluonts_data

# Choose target variable
target_column = 'TMAX'  # Can be changed to TMIN, PRCP, etc.

# Convert data to GluonTS format
train_gluonts = prepare_gluonts_data(train_data, target_column)
val_gluonts = prepare_gluonts_data(val_data, target_column)
test_gluonts = prepare_gluonts_data(test_data, target_column)

print(f"Created {len(train_gluonts)} training series, {len(val_gluonts)} validation series, {len(test_gluonts)} test series")

# 9. Create HuggingFace Datasets
print("Creating HuggingFace Datasets...")

# Process start field for HuggingFace Dataset
class ProcessStartField:
    ts_id = 0
    
    def __call__(self, data):
        # Ensure start is a timestamp
        if isinstance(data["start"], pd.Timestamp):
            data["start"] = data["start"].to_pydatetime()
        
        # Assign a unique ID to each time series
        if "feat_static_cat" not in data:
            data["feat_static_cat"] = [self.ts_id]
        
        self.ts_id += 1
        return data

# Apply processing to each split
process_start = ProcessStartField()
train_processed = list(Map(process_start, train_gluonts))
process_start = ProcessStartField()  # Reset ID counter
val_processed = list(Map(process_start, val_gluonts))
process_start = ProcessStartField()  # Reset ID counter
test_processed = list(Map(process_start, test_gluonts))

# Define dataset features
features = Features(
    {    
        "start": Value("timestamp[s]"),
        "target": Sequence(Value("float32")),
        "feat_static_cat": Sequence(Value("int64")),
        "feat_static_real": Sequence(Value("float32")),
        "item_id": Value("string"),
    }
)

# Create HuggingFace Datasets
train_dataset = Dataset.from_list(train_processed, features=features)
val_dataset = Dataset.from_list(val_processed, features=features)
test_dataset = Dataset.from_list(test_processed, features=features)

# 10. Save the prepared datasets
print("Saving prepared datasets...")
data_dir = "D:/Dev/python-projects/weather-forecasting/prepared_datasets"
os.makedirs(data_dir, exist_ok=True)

# Save datasets

# Create a DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Save the DatasetDict
dataset.save_to_disk(f"{data_dir}/dataset")

# Save target column information
with open(f"{data_dir}/metadata.txt", "w") as f:
    f.write(f"target_column={target_column}\n")
    f.write("freq=D\n")  # Daily frequency

print("Data preparation complete!")
print("Datasets saved to:", data_dir)
print("Final dataset sizes:")
print(f"Train: {len(train_dataset)} series")
print(f"Validation: {len(val_dataset)} series")
print(f"Test: {len(test_dataset)} series")





import matplotlib.pyplot as plt

def plot_sequential_time_series(train_example, validation_example, test_example, 
                                 title=None, target_column="TMAX"):
    """
    Plot time series data sequentially from train, validation, and test sets.
    
    Parameters:
    - train_example: First time series example from train dataset
    - validation_example: First time series example from validation dataset
    - test_example: First time series example from test dataset
    - title: Optional title for the plot
    - target_column: Name of the target column (default is "TMAX")
    """
    # Create a figure and axis
    plt.figure(figsize=(15, 6))
    
    # Extract target values
    train_target = train_example["target"]
    validation_target = validation_example["target"]
    test_target = test_example["target"]
    
    # Calculate the x-axis values for each array
    train_x = np.arange(0, len(train_target))
    validation_x = np.arange(len(train_target), len(train_target) + len(validation_target))
    test_x = np.arange(len(train_target) + len(validation_target), 
                       len(train_target) + len(validation_target) + len(test_target))
    
    # Plot each array with its corresponding x-axis values
    plt.plot(train_x, train_target, color="blue", label="Train")
    plt.plot(validation_x, validation_target, color="red", label="Validation")
    plt.plot(test_x, test_target, color="green", label="Test")
    
    # Add labels and title
    plt.xlabel("Time Index")
    plt.ylabel(f"{target_column} Temperature")
    plt.title(title or f"Sequential Time Series Plot of {target_column} ({train_example['item_id']})")
    plt.legend()
    
    # Add vertical lines to separate datasets
    plt.axvline(x=len(train_target), color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=len(train_target) + len(validation_target), color='gray', linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
for x in range(max_files):     
    # Usage in your existing code:
    plot_sequential_time_series(
        train_dataset[x], 
        val_dataset[x], 
        test_dataset[x], 
        target_column=target_column
    )