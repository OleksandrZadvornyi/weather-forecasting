import os
from datasets import load_from_disk
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from evaluate import load
from gluonts.time_feature import get_seasonality, time_features_from_frequency_str
from transformers import TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig
from gluonts.dataset.common import ListDataset
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    InstanceSplitter,
    RemoveFields,
    TestSplitSampler,
    VstackFeatures,
    RenameFields,
)
from gluonts.dataset.field_names import FieldName

# Load model and configuration
model_dir = "./weather_model"
model_path = os.path.join(model_dir, "time_series_model.pth")
config_path = os.path.join(model_dir, "config")

# Load configuration
config = TimeSeriesTransformerConfig.from_pretrained(config_path)

# Load metadata
metadata = {}
with open(os.path.join(model_dir, "config/metadata.txt"), "r") as f:
    for line in f:
        key, value = line.strip().split("=")
        metadata[key] = value

freq = metadata["freq"]
prediction_length = int(metadata["prediction_length"])
target_column = metadata.get("target_column", "TMAX")

# Initialize model
model = TimeSeriesTransformerForPrediction(config)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load test dataset
data_dir = "D:/Dev/python-projects/weather-forecasting/prepared_datasets"
dataset = load_from_disk(f"{data_dir}/dataset")
test_dataset = dataset["test"]

# Convert test dataset to GluonTS ListDataset format
def convert_to_gluonts_dataset(hf_dataset, freq):
    data = []
    for item in hf_dataset:
        data.append({
            FieldName.START: pd.Period(item["start"], freq=freq),
            FieldName.TARGET: item["target"],
            FieldName.FEAT_STATIC_CAT: [item["feat_static_cat"][0]],
            FieldName.FEAT_STATIC_REAL: item["feat_static_real"],
            FieldName.ITEM_ID: item["item_id"]
        })
    return ListDataset(data, freq=freq)

gluonts_test_dataset = convert_to_gluonts_dataset(test_dataset, freq)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create test transformation
def create_test_transformation(freq: str, config: TimeSeriesTransformerConfig):
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    return Chain(
        [RemoveFields(field_names=remove_field_names)]
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                expected_ndim=1,
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE],
            ),
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )

# Create test instance splitter
def create_test_instance_splitter(config):
    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=TestSplitSampler(),
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )

# Create test dataloader
def create_test_dataloader(dataset, config, freq, batch_size=64):
    transformation = create_test_transformation(freq, config)
    transformed_data = transformation.apply(dataset, is_train=False)
    
    instance_splitter = create_test_instance_splitter(config)
    testing_instances = instance_splitter.apply(transformed_data, is_train=False)
    
    from gluonts.dataset.loader import as_stacked_batches
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=[
            "past_time_features",
            "past_values",
            "past_observed_mask",
            "future_time_features",
            "static_categorical_features",
            "static_real_features",
        ],
    )

# Generate forecasts
forecasts = []
test_dataloader = create_test_dataloader(gluonts_test_dataset, config, freq)

for batch in test_dataloader:
    outputs = model.generate(
        static_categorical_features=batch["static_categorical_features"].to(device),
        static_real_features=batch["static_real_features"].to(device),
        past_time_features=batch["past_time_features"].to(device),
        past_values=batch["past_values"].to(device),
        future_time_features=batch["future_time_features"].to(device),
        past_observed_mask=batch["past_observed_mask"].to(device),
    )
    forecasts.append(outputs.sequences.cpu().numpy())

forecasts = np.vstack(forecasts)
print(f"Generated forecasts shape: {forecasts.shape}")

# Calculate evaluation metrics
mase_metric = load("evaluate-metric/mase")
smape_metric = load("evaluate-metric/smape")

forecast_median = np.median(forecasts, axis=1)

mase_metrics = []
smape_metrics = []

for item_id, ts in enumerate(test_dataset):
    training_data = ts["target"][:-prediction_length]
    ground_truth = ts["target"][-prediction_length:]
    
    mase = mase_metric.compute(
        predictions=forecast_median[item_id], 
        references=np.array(ground_truth), 
        training=np.array(training_data), 
        periodicity=get_seasonality(freq))
    mase_metrics.append(mase["mase"])
    
    smape = smape_metric.compute(
        predictions=forecast_median[item_id], 
        references=np.array(ground_truth),
    )
    smape_metrics.append(smape["smape"])

print(f"\nEvaluation Metrics:")
print(f"MASE (mean): {np.mean(mase_metrics):.4f}")
print(f"sMAPE (mean): {np.mean(smape_metrics):.4f}")

# Plot metrics distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(mase_metrics, bins=20, color='blue', alpha=0.7)
plt.title("MASE Distribution")
plt.xlabel("MASE")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(smape_metrics, bins=20, color='green', alpha=0.7)
plt.title("sMAPE Distribution")
plt.xlabel("sMAPE")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# Plot some sample forecasts
def plot_forecast(ts_index, prediction_length=30):
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Get the test example
    ts = test_dataset[ts_index]
    
    # Create date range
    dates = pd.date_range(
        start=pd.Period(ts["start"], freq=freq).start_time,
        periods=len(ts["target"]),
        freq=freq,
    )
    
    # Plot historical data
    ax.plot(
        dates[:-prediction_length], 
        ts["target"][:-prediction_length], 
        label="Historical Data",
        color="blue",
        linewidth=2
    )
    
    # Plot actual values in forecast period
    ax.plot(
        dates[-prediction_length:], 
        ts["target"][-prediction_length:], 
        label="Actual Future Values",
        color="green",
        linestyle="--",
        linewidth=2
    )
    
    # Plot median forecast
    ax.plot(
        dates[-prediction_length:], 
        forecast_median[ts_index], 
        label="Median Forecast",
        color="red",
        linewidth=2
    )
    
    # Plot forecast uncertainty with wider confidence interval
    ax.fill_between(
        dates[-prediction_length:],
        np.percentile(forecasts[ts_index], 10, axis=0),
        np.percentile(forecasts[ts_index], 90, axis=0),
        color="red",
        alpha=0.2,
        label="80% Confidence Interval"
    )
    
    ax.set_title(f"Forecast for Station {ts['item_id']} - {target_column}", fontsize=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(f"{target_column} (°C)", fontsize=12)
    ax.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Plot with longer prediction length
for i in range(3):
    plot_forecast(i, prediction_length=90)  # Try 30 or 90 days