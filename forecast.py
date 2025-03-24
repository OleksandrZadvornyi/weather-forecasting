import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = "D:/Dev/python-projects/weather-forecasting/prepared_data_small"
OUTPUT_DIR = "D:/Dev/python-projects/weather-forecasting/model_output"
MODEL_NAME = "weather-transformer"
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 2e-4
CONTEXT_LENGTH = 14  # seq_length in your data preparation
PREDICTION_LENGTH = 7  # forecast_horizon in your data preparation
WARMUP_STEPS = 500
FP16 = False  # Set to True if your GPU supports mixed precision

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the data
logger.info(f"Loading data from {DATA_DIR}")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

logger.info("Data loaded successfully")
logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Check data types and convert if necessary
if X_train.dtype != np.float32:
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

if y_train.dtype != np.float32:
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)

logger.info(f"X_train dtype: {X_train.dtype}, y_train dtype: {y_train.dtype}")

# Create PyTorch Dataset
class WeatherDataset(Dataset):
    def __init__(self, X, y, target_idx=None):
        """
        Initialize the dataset.
        
        Args:
            X: Input features, shape (n_samples, seq_length, n_features)
            y: Target values, shape (n_samples, forecast_horizon)
            target_idx: Index of the target feature in X (if None, assumed to be a separate y array)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.target_idx = target_idx
        
        # For the time series transformer, we need to ensure the input
        # has a consistent feature dimension
        self.num_features = self.X.shape[2]
        self.seq_length = self.X.shape[1]
        self.pred_length = self.y.shape[1]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Get the input sequence and target sequence
        past_values = self.X[idx]
        future_targets = self.y[idx]
        
        # Transformer expects past_values to be [seq_len, n_features]
        # and future_values to be [prediction_len, n_features]
        # But we only have future_values as [prediction_len], so we need to expand it
        
        # Create a tensor of zeros for all features
        expanded_future = torch.zeros((future_targets.shape[0], past_values.shape[1]), dtype=torch.float32)
        
        # If target_idx is provided, we assume this is the index of the target feature in X
        # Otherwise, we place the target in the first position
        target_position = self.target_idx if self.target_idx is not None else 0
        
        # Place the targets in the appropriate position
        expanded_future[:, target_position] = future_targets
        
        # Create past_time_features and observed mask
        # For TimeSeriesTransformer, we need to add time features and masks
        past_time_features = torch.zeros((self.seq_length, 0), dtype=torch.float32)  # Empty tensor with 0 time features
        past_observed_mask = torch.ones((self.seq_length, self.num_features), dtype=torch.float32)  # All values observed
        
        # Future time features and future observed mask (for training)
        future_time_features = torch.zeros((self.pred_length, 0), dtype=torch.float32)  # Empty tensor with 0 time features
        future_observed_mask = torch.ones((self.pred_length, self.num_features), dtype=torch.float32)
        
        return {
            "past_values": past_values,
            "past_time_features": past_time_features,
            "past_observed_mask": past_observed_mask,
            "future_values": expanded_future,
            "future_time_features": future_time_features,
            "future_observed_mask": future_observed_mask
        }

# Create datasets and dataloaders
train_dataset = WeatherDataset(X_train, y_train)
val_dataset = WeatherDataset(X_val, y_val)
test_dataset = WeatherDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

logger.info("Created datasets and dataloaders")
logger.info(f"Train dataset size: {len(train_dataset)}")
logger.info(f"Validation dataset size: {len(val_dataset)}")
logger.info(f"Test dataset size: {len(test_dataset)}")

# Verify data format by checking first batch
sample_batch = next(iter(train_dataloader))
logger.info(f"Sample batch past_values shape: {sample_batch['past_values'].shape}")
logger.info(f"Sample batch past_time_features shape: {sample_batch['past_time_features'].shape}")
logger.info(f"Sample batch past_observed_mask shape: {sample_batch['past_observed_mask'].shape}")
logger.info(f"Sample batch future_values shape: {sample_batch['future_values'].shape}")

# Create model configuration
config = TimeSeriesTransformerConfig(
    prediction_length=PREDICTION_LENGTH,
    context_length=CONTEXT_LENGTH,
    lags_sequence=[1],  # Just use a simple lag
    num_time_features=0,  # We're not using time features
    num_static_categorical_features=0,  # We're not using categorical features
    num_static_real_features=0,  # We're not using static features
    num_dynamic_real_features=X_train.shape[2],  # Number of features in our sequence data
    cardinality=[],  # No categorical features
    embedding_dimension=[],  # No categorical features
    d_model=64,  # Dimension of the model
    encoder_layers=2,
    decoder_layers=2,
    encoder_attention_heads=2,
    decoder_attention_heads=2,
    encoder_ffn_dim=128,
    decoder_ffn_dim=128,
    activation_function="gelu",
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0.1,
    encoder_layerdrop=0.1,
    decoder_layerdrop=0.1,
    use_cache=True,
    is_encoder_decoder=True,
    distribution_output="normal",  # Use normal distribution output
)

logger.info("Created model configuration")
logger.info(f"Model config: {config}")

# Create model
model = TimeSeriesTransformerForPrediction(config)
logger.info(f"Created model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
model = model.to(device)

# Define compute_loss function for our trainer
def compute_loss(model, inputs, num_items_in_batch=None):
    outputs = model(
        past_values=inputs["past_values"],
        past_time_features=inputs["past_time_features"],
        past_observed_mask=inputs["past_observed_mask"],
        future_values=inputs["future_values"],
        future_time_features=inputs["future_time_features"],
        future_observed_mask=inputs["future_observed_mask"]
    )
    return outputs.loss

# Define training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, MODEL_NAME),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=FP16,
    learning_rate=LEARNING_RATE,
    save_total_limit=3,  # Only keep the 3 best checkpoints
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    greater_is_better=False,  # Lower loss is better
    report_to="none",  # Disable reporting to wandb, tensorboard etc.
)

# Custom evaluation function
def compute_metrics(eval_pred):
    # This will be called with EvalPrediction objects
    # which contain predictions and label_ids
    # But for TimeSeriesTransformer, we need to process differently
    # We'll calculate metrics manually in a separate evaluation step
    return {}

# Custom Trainer class to handle the TimeSeriesTransformer model
class TimeSeriesTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            past_values=inputs["past_values"],
            past_time_features=inputs["past_time_features"],
            past_observed_mask=inputs["past_observed_mask"],
            future_values=inputs["future_values"],
            future_time_features=inputs["future_time_features"],
            future_observed_mask=inputs["future_observed_mask"],
            static_categorical_features=None,
            static_real_features=None
        )
        
        # Debugging: Check if static_feat exists
        if hasattr(outputs, "static_feat"):
            print(f"static_feat shape: {outputs.static_feat.shape}")
            
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# Create trainer
trainer = TimeSeriesTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
logger.info("Starting model training")
start_time = time.time()
train_result = trainer.train()
end_time = time.time()
logger.info(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")
logger.info(f"Train metrics: {train_result.metrics}")

# Save model
model_save_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}-final")
trainer.save_model(model_save_path)
logger.info(f"Model saved to {model_save_path}")

# Evaluate on test set
logger.info("Evaluating on test set")
test_results = trainer.evaluate(test_dataset)
logger.info(f"Test metrics: {test_results}")

# Function to make predictions
def make_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move all inputs to device
            past_values = batch["past_values"].to(device)
            past_time_features = batch["past_time_features"].to(device)
            past_observed_mask = batch["past_observed_mask"].to(device)
            future_values = batch["future_values"].to(device)
            future_time_features = batch["future_time_features"].to(device)
            future_observed_mask = batch["future_observed_mask"].to(device)
            
            outputs = model(
                past_values=past_values,
                past_time_features=past_time_features,
                past_observed_mask=past_observed_mask,
                future_values=None,  # Set to None for inference mode
                future_time_features=future_time_features,
                future_observed_mask=None  # Set to None for inference mode
            )
            
            # Get mean of the distribution for predictions
            preds = outputs.predictions  # Should be [batch, prediction_length, features]
            
            # Get target feature predictions (first feature)
            target_preds = preds[:, :, 0].cpu().numpy()
            target_labels = future_values[:, :, 0].cpu().numpy()
            
            all_preds.append(target_preds)
            all_labels.append(target_labels)
    
    return np.vstack(all_preds), np.vstack(all_labels)

# Make predictions on test set
logger.info("Making predictions on test set")
test_preds, test_labels = make_predictions(model, test_dataloader, device)

# Calculate metrics on test set
mse = mean_squared_error(test_labels.flatten(), test_preds.flatten())
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_labels.flatten(), test_preds.flatten())

# Avoid division by zero for MAPE
mask = test_labels != 0
mape = np.mean(np.abs((test_labels[mask] - test_preds[mask]) / test_labels[mask])) * 100

logger.info("Test set metrics:")
logger.info(f"MSE: {mse:.4f}")
logger.info(f"RMSE: {rmse:.4f}")
logger.info(f"MAE: {mae:.4f}")
logger.info(f"MAPE: {mape:.4f}%")

# Create a figure to visualize predictions
logger.info("Visualizing predictions")
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# Plot some examples
examples_to_plot = [0, len(test_dataset)//2, len(test_dataset)-1]  # First, middle, last

for i, idx in enumerate(examples_to_plot):
    ax = axes[i]
    
    # Get the past, future, and predictions
    if idx < len(test_dataset):
        sample = test_dataset[idx]
        past = sample['past_values'][:, 0].numpy()  # Target feature (first column)
        future = sample['future_values'][:, 0].numpy()  # Target feature (first column)
        pred = test_preds[idx]
        
        # Time steps
        past_timesteps = np.arange(0, len(past))
        future_timesteps = np.arange(len(past), len(past) + len(future))
        
        # Plot
        ax.plot(past_timesteps, past, 'b-', label='Historical')
        ax.plot(future_timesteps, future, 'g-', label='Actual')
        ax.plot(future_timesteps, pred, 'r--', label='Prediction')
        ax.set_title(f'Example {idx}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_examples.png'))
logger.info(f"Prediction visualization saved to {os.path.join(OUTPUT_DIR, 'prediction_examples.png')}")

# Save a summary of experiment configuration and results
summary = {
    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_config": {k: str(v) if not isinstance(v, (int, float, bool, list, dict, type(None))) else v 
                    for k, v in config.to_dict().items()},
    "train_samples": len(train_dataset),
    "val_samples": len(val_dataset),
    "test_samples": len(test_dataset),
    "num_features": X_train.shape[2],
    "context_length": CONTEXT_LENGTH,
    "prediction_length": PREDICTION_LENGTH,
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "test_metrics": {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape)
    },
    "training_time_minutes": (end_time - start_time) / 60
}

import json
with open(os.path.join(OUTPUT_DIR, 'experiment_summary.json'), 'w') as f:
    json.dump(summary, f, indent=4)

logger.info(f"Experiment summary saved to {os.path.join(OUTPUT_DIR, 'experiment_summary.json')}")
logger.info("Training script completed successfully")