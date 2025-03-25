from datasets import load_dataset
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
import numpy as np
import torch
import os

from accelerate import Accelerator
from torch.optim import AdamW

dataset = load_dataset("monash_tsf", "tourism_monthly")

train_example = dataset['train'][0]
validation_example = dataset['validation'][0]
test_example = dataset['test'][0]

freq = "1M"
prediction_length = 24

assert len(train_example["target"]) + prediction_length == len(
    validation_example["target"]
)

# Optional: Visualize training data
# figure, axes = plt.subplots()
# axes.plot(train_example["target"], color="blue")
# axes.plot(validation_example["target"], color="red", alpha=0.5)
# plt.show()

train_dataset = dataset["train"]
test_dataset = dataset["test"]

@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch

train_dataset.set_transform(partial(transform_start_field, freq=freq))
test_dataset.set_transform(partial(transform_start_field, freq=freq))

lags_sequence = get_lags_for_frequency(freq)
time_features = time_features_from_frequency_str(freq)

config = TimeSeriesTransformerConfig(
    prediction_length=prediction_length,
    context_length=prediction_length * 2,
    lags_sequence=lags_sequence,
    num_time_features=len(time_features) + 1,
    num_static_categorical_features=1,
    cardinality=[len(train_dataset)],
    embedding_dimension=[2],
    encoder_layers=4,
    decoder_layers=4,
    d_model=32,
)

model = TimeSeriesTransformerForPrediction(config)

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

train_dataloader = create_train_dataloader(
    config=config,
    freq=freq,
    data=train_dataset,
    batch_size=256,
    num_batches_per_epoch=100,
)

# Optional: Test a forward pass before training
# batch = next(iter(train_dataloader))
# for k, v in batch.items():
#     print(k, v.shape, v.type())
# outputs = model(
#     past_values=batch["past_values"],
#     past_time_features=batch["past_time_features"],
#     past_observed_mask=batch["past_observed_mask"],
#     static_categorical_features=batch["static_categorical_features"]
#     if config.num_static_categorical_features > 0
#     else None,
#     static_real_features=batch["static_real_features"]
#     if config.num_static_real_features > 0
#     else None,
#     future_values=batch["future_values"],
#     future_time_features=batch["future_time_features"],
#     future_observed_mask=batch["future_observed_mask"],
#     output_hidden_states=True,
# )
# print("Loss:", outputs.loss.item())

# Set up training
accelerator = Accelerator()
device = accelerator.device

model.to(device)
optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)

model, optimizer, train_dataloader = accelerator.prepare(
    model,
    optimizer,
    train_dataloader,
)

# Training loop
model.train()
for epoch in range(40):
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

        if idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {idx}, Loss: {loss.item():.4f}")
    
    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")

# Save the model and configuration

# Create directory for model artifacts
os.makedirs("./saved_model", exist_ok=True)
os.makedirs("./saved_model/config", exist_ok=True)

model_path = "./saved_model/time_series_model.pth"
config_path = "./saved_model/config"

# Get the unwrapped model if using accelerator
unwrapped_model = accelerator.unwrap_model(model)

# Save model state dictionary
torch.save(unwrapped_model.state_dict(), model_path)

# Save the configuration
unwrapped_model.config.to_json_file(os.path.join("./saved_model/config/config.json"))

# Save frequency and prediction length for later use
with open(os.path.join("./saved_model/config/metadata.txt"), "w") as f:
    f.write(f"freq={freq}\n")
    f.write(f"prediction_length={prediction_length}\n")
    f.write(f"lags_sequence={lags_sequence}\n")

print(f"Model saved to {model_path}")
print(f"Configuration saved to {config_path}")