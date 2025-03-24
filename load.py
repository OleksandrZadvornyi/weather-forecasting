from datasets import load_dataset
from functools import lru_cache
from functools import partial

from gluonts.time_feature import (
    time_features_from_frequency_str,
    get_seasonality
)
from gluonts.dataset.field_names import FieldName

from gluonts.dataset.loader import as_stacked_batches

from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    InstanceSplitter,
    TestSplitSampler,
    Transformation,
    VstackFeatures,
    RenameFields,
    RemoveFields
)

from transformers import (
    TimeSeriesTransformerConfig, 
    TimeSeriesTransformerForPrediction,
    PretrainedConfig
)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import torch
import os

# Load saved metadata
config_path = "./saved_model/config"
model_path = "./saved_model/time_series_model.pth"

# Read metadata
freq = None
prediction_length = None
lags_sequence = None

with open(os.path.join(config_path, "metadata.txt"), "r") as f:
    for line in f:
        if line.startswith("freq="):
            freq = line.strip().split("=")[1]
        elif line.startswith("prediction_length="):
            prediction_length = int(line.strip().split("=")[1])
        elif line.startswith("lags_sequence="):
            # Parse list from string representation
            lags_str = line.strip().split("=")[1]
            lags_sequence = eval(lags_str)  # Be careful with eval

print(f"Loaded metadata: freq={freq}, prediction_length={prediction_length}")

# Load the configuration
config = TimeSeriesTransformerConfig.from_json_file(os.path.join(config_path, "config.json"))

# Create model with the configuration
model = TimeSeriesTransformerForPrediction(config)

# Load the model weights
model.load_state_dict(torch.load(model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Model loaded successfully")

# Load test dataset
dataset = load_dataset("monash_tsf", "tourism_monthly")
test_dataset = dataset["test"]

@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch

test_dataset.set_transform(partial(transform_start_field, freq=freq))

# Define transformation functions (same as in training)
def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    return Chain(
        [RemoveFields(field_names=remove_field_names)]
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                expected_ndim=1 if config.input_size == 1 else 2,
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
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
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

def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )

def create_test_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    instance_sampler = create_instance_splitter(config, "test")

    testing_instances = instance_sampler.apply(transformed_data, is_train=False)
    
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )

# Create test dataloader
test_dataloader = create_test_dataloader(
    config=config,
    freq=freq,
    data=test_dataset,
    batch_size=64,
)

# Generate forecasts
forecasts = []

with torch.no_grad():
    for batch in test_dataloader:
        outputs = model.generate(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        )
        forecasts.append(outputs.sequences.cpu().numpy())

forecasts = np.vstack(forecasts)
print(f"Generated forecasts shape: {forecasts.shape}")

# Optional: Evaluate the forecasts
try:
    from evaluate import load
    
    mase_metric = load("evaluate-metric/mase")
    smape_metric = load("evaluate-metric/smape")

    forecast_median = np.median(forecasts, 1)

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
        
    print(f"MASE: {np.mean(mase_metrics)}")
    print(f"sMAPE: {np.mean(smape_metrics)}")
    
    # Optional: Visualize metrics
    plt.scatter(mase_metrics, smape_metrics, alpha=0.3)
    plt.xlabel("MASE")
    plt.ylabel("sMAPE")
    plt.title("Forecast Error Metrics")
    plt.savefig("./saved_model/metrics_scatter.png")
    plt.show()
except ImportError:
    print("Evaluation metrics not available. Skipping evaluation.")

# Function to plot forecasts
def plot_forecast(ts_index, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    index = pd.period_range(
        start=test_dataset[ts_index][FieldName.START],
        periods=len(test_dataset[ts_index][FieldName.TARGET]),
        freq=freq,
    ).to_timestamp()

    # Major ticks every half year, minor ticks every month,
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    # Plot actual values (context + ground truth)
    ax.plot(
        index[-2*prediction_length:], 
        test_dataset[ts_index]["target"][-2*prediction_length:],
        label="Actual",
        color="blue",
        linewidth=2
    )

    # Plot forecast median
    ax.plot(
        index[-prediction_length:], 
        np.median(forecasts[ts_index], axis=0),
        label="Forecast (median)",
        color="red",
        linewidth=2
    )
    
    # Plot confidence interval
    ax.fill_between(
        index[-prediction_length:],
        forecasts[ts_index].mean(0) - forecasts[ts_index].std(axis=0), 
        forecasts[ts_index].mean(0) + forecasts[ts_index].std(axis=0), 
        alpha=0.3, 
        color="red",
        interpolate=True,
        label="+/- 1-std"
    )
    
    # Add vertical line separating historical data and forecasts
    ax.axvline(x=index[-prediction_length], color="black", linestyle="--")
    
    # Add labels and legend
    ax.set_title(f"Time Series #{ts_index} Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    plt.show()
    
# Plot some example forecasts
print("Plotting example forecasts...")
os.makedirs("./saved_model/plots", exist_ok=True)

# Plot a few examples
for i in [0, 50, 100, 334]:
    if i < len(test_dataset):
        plot_forecast(i, f"./saved_model/plots/forecast_{i}.png")

print("Forecasting process completed!")