import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datasets import load_from_disk
from transformers import TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig
from functools import partial

# Load metadata
data_dir = "D:/Dev/python-projects/weather-forecasting/prepared_datasets"
with open(f"{data_dir}/metadata.txt", "r") as f:
    metadata = {}
    for line in f:
        key, value = line.strip().split("=")
        metadata[key] = value

# Load datasets
test_dataset = load_from_disk(f"{data_dir}/test")

# Metadata parameters
target_column = metadata.get("target_column", "TMAX")
freq = metadata.get("freq", "D")
prediction_length = int(metadata.get("prediction_length", 7))
context_length = prediction_length * 2

# Helper function to convert start field
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch

# Set transform for test dataset
test_dataset.set_transform(partial(transform_start_field, freq=freq))

# Reload saved model configuration and model
config = TimeSeriesTransformerConfig.from_json_file("./weather_model/config/config.json")
model = TimeSeriesTransformerForPrediction(config)
model.load_state_dict(torch.load("./weather_model/time_series_model.pth"))
model.eval()

# Function to generate forecast for a single time series
def generate_forecast(model, time_series, device='cpu'):
    # Prepare input features
    inputs = {
        "past_values": torch.tensor(time_series["target"][:-prediction_length]).unsqueeze(0).float(),
        "past_time_features": torch.tensor(time_series["time_features"][:, :-prediction_length]).unsqueeze(0).float(),
        "past_observed_mask": torch.ones_like(torch.tensor(time_series["target"][:-prediction_length])).unsqueeze(0).bool(),
        "future_time_features": torch.tensor(time_series["time_features"][:, -prediction_length:]).unsqueeze(0).float(),
    }
    
    # Add static features if they exist
    if "static_categorical_features" in time_series:
        inputs["static_categorical_features"] = torch.tensor(time_series["static_categorical_features"]).unsqueeze(0)
    if "static_real_features" in time_series:
        inputs["static_real_features"] = torch.tensor(time_series["static_real_features"]).unsqueeze(0).float()
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate forecast
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    return outputs.sequences.cpu().numpy()

# Function to plot forecast
def plot_forecast(time_series, forecasts, target_column, freq='D', prediction_length=7):
    # Create timestamp index
    index = pd.period_range(
        start=time_series['start'],
        periods=len(time_series['target']),
        freq=freq,
    ).to_timestamp()

    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(index[:-prediction_length], time_series['target'][:-prediction_length], 
             label='Historical', color='blue')
    
    # Plot ground truth for forecast period
    plt.plot(index[-prediction_length:], time_series['target'][-prediction_length:], 
             label='Ground Truth', color='green', linestyle='--')
    
    # Plot forecast
    forecast_median = np.median(forecasts, axis=0)
    plt.plot(index[-prediction_length:], forecast_median, 
             label='Forecast Median', color='red')
    
    # Plot forecast uncertainty
    plt.fill_between(
        index[-prediction_length:],
        np.percentile(forecasts, 25, axis=0),
        np.percentile(forecasts, 75, axis=0),
        alpha=0.3, color='red', label='50% Confidence Interval'
    )
    
    plt.title(f'{target_column} Forecast')
    plt.xlabel('Date')
    plt.ylabel(target_column)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Visualization for a few time series
num_plots = 3
for i in range(num_plots):
    time_series = test_dataset[i]
    
    # Generate forecasts
    forecasts = generate_forecast(model, time_series)
    
    # Plot forecast
    plot_forecast(time_series, forecasts, target_column, freq, prediction_length)

# Calculate and print some forecast metrics
from evaluate import load

mase_metric = load("evaluate-metric/mase")
smape_metric = load("evaluate-metric/smape")

mase_metrics = []
smape_metrics = []

for item_id, ts in enumerate(test_dataset[:50]):  # Limit to first 50 series for computation time
    training_data = ts["target"][:-prediction_length]
    ground_truth = ts["target"][-prediction_length:]
    
    # Generate forecast
    forecasts = generate_forecast(model, ts)
    forecast_median = np.median(forecasts, axis=0)
    
    # Compute metrics
    mase = mase_metric.compute(
        predictions=forecast_median, 
        references=np.array(ground_truth), 
        training=np.array(training_data), 
        periodicity=1  # For daily data
    )
    mase_metrics.append(mase["mase"])
    
    smape = smape_metric.compute(
        predictions=forecast_median, 
        references=np.array(ground_truth)
    )
    smape_metrics.append(smape["smape"])

print(f"Average MASE: {np.mean(mase_metrics):.4f}")
print(f"Average sMAPE: {np.mean(smape_metrics):.4f}")

# Scatter plot of metrics
plt.figure(figsize=(10, 6))
plt.scatter(mase_metrics, smape_metrics, alpha=0.5)
plt.xlabel("MASE")
plt.ylabel("sMAPE")
plt.title("Forecast Accuracy Metrics")
plt.tight_layout()
plt.show()