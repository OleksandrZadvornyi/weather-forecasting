from datasets import load_from_disk
from functools import lru_cache
from functools import partial

from gluonts.time_feature import (
    get_lags_for_frequency, 
    time_features_from_frequency_str
)
from gluonts.dataset.field_names import FieldName
from gluonts.transform.sampler import InstanceSampler
from typing import Optional, Iterable

from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches

from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)

from transformers import (
    TimeSeriesTransformerConfig, 
    TimeSeriesTransformerForPrediction,
    PretrainedConfig
)

import matplotlib.pyplot as plt
import pandas as pd
import torch
import os

from accelerate import Accelerator
from torch.optim import AdamW

@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch

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
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
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

def create_train_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
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

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    instance_splitter = create_instance_splitter(config, "train")

    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(stream)
    
    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )

# Load datasets
data_dir = "D:/Dev/python-projects/weather-forecasting/prepared_datasets"

dataset = load_from_disk(f"{data_dir}/dataset")

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
test_dataset = dataset["test"]

print(f"Train dataset: {len(train_dataset)} time series")
print(f"Validation dataset: {len(validation_dataset)} time series")
print(f"Test dataset: {len(test_dataset)} time series")

# Load metadata
with open(f"{data_dir}/metadata.txt", "r") as f:
    metadata = {}
    for line in f:
        key, value = line.strip().split("=")
        metadata[key] = value

target_column = metadata.get("target_column", "TMAX")
freq = metadata.get("freq", "D")  # Daily frequency

# Example of a time series from training set
train_example = train_dataset[0]
validation_example = validation_dataset[0]
test_example = test_dataset[0]

# Define prediction parameters
prediction_length = 7  # Predict 7 days ahead
context_length = prediction_length * 2  # Use 14 days of context

# assert len(train_example["target"]) + prediction_length == len(
#     validation_example["target"]
# )

# figure, axes = plt.subplots()
# axes.plot(test_example["target"], color="green")
# axes.plot(validation_example["target"], color="red")
# axes.plot(train_example["target"], color="blue")
# plt.show()

import numpy as np

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
    plt.title(title or f"Sequential Time Series Plot of {target_column}")
    plt.legend()
    
    # Add vertical lines to separate datasets
    plt.axvline(x=len(train_target), color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=len(train_target) + len(validation_target), color='gray', linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Usage in your existing code:
plot_sequential_time_series(
    train_example, 
    validation_example, 
    test_example, 
    target_column=target_column
)

# Set transforms
train_dataset.set_transform(partial(transform_start_field, freq=freq))
validation_dataset.set_transform(partial(transform_start_field, freq=freq))
test_dataset.set_transform(partial(transform_start_field, freq=freq))

# Configure model
lags_sequence = get_lags_for_frequency(freq)
time_features = time_features_from_frequency_str(freq)

config = TimeSeriesTransformerConfig(
    prediction_length=prediction_length,
    context_length=context_length,
    lags_sequence=lags_sequence,
    num_time_features=len(time_features) + 1,  # Add 1 for age feature
    num_static_categorical_features=1,
    num_static_real_features=3,  # Latitude, longitude, elevation
    cardinality=[len(train_dataset)],  # Number of unique time series
    embedding_dimension=[2],  # Dimension of categorical embedding
    encoder_layers=4,
    decoder_layers=4,
    d_model=64,
)

model = TimeSeriesTransformerForPrediction(config)

# Create data loaders
train_dataloader = create_train_dataloader(
    config=config,
    freq=freq,
    data=train_dataset,
    batch_size=128,
    num_batches_per_epoch=50,
)

# Set up training
accelerator = Accelerator()
device = accelerator.device

model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-2)

model, optimizer, train_dataloader = accelerator.prepare(
    model,
    optimizer,
    train_dataloader,
)

# Training loop
model.train()
num_epochs = 30
print(f"Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0
    
    for idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            future_values=batch["future_values"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
            future_observed_mask=batch["future_observed_mask"].to(device),
        )
        loss = outputs.loss
        
        # Backpropagation
        accelerator.backward(loss)
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1

        if idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {idx}, Loss: {loss.item():.4f}")
    
    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")

# Save the model and configuration
os.makedirs("./weather_model", exist_ok=True)
os.makedirs("./weather_model/config", exist_ok=True)

model_path = "./weather_model/time_series_model.pth"
config_path = "./weather_model/config"

# Get the unwrapped model if using accelerator
unwrapped_model = accelerator.unwrap_model(model)

# Save model state dictionary
torch.save(unwrapped_model.state_dict(), model_path)

# Save the configuration
unwrapped_model.config.to_json_file(os.path.join("./weather_model/config/config.json"))

# Save frequency and prediction length for later use
with open(os.path.join("./weather_model/config/metadata.txt"), "w") as f:
    f.write(f"freq={freq}\n")
    f.write(f"prediction_length={prediction_length}\n")
    f.write(f"target_column={target_column}\n")
    f.write(f"lags_sequence={lags_sequence}\n")

print(f"Model saved to {model_path}")
print(f"Configuration saved to {config_path}")