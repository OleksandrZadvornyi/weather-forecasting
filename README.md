# Weather Forecasting with Time Series Transformers

This repository contains a deep learning pipeline for forecasting daily maximum temperatures (`TMAX`) using the **Time Series Transformer** architecture. The project utilizes HuggingFace `transformers`, `gluonts` and `accelerator` to preprocess historical weather data, train a model and visualize 30-day forecasts.

## 📂 Repository Structure

* **`daily-summaries-latest-upm/`**: Directory containing raw CSV data files (Daily Summaries) for Ukrainian weather stations.
* **`prepare_dataset.ipynb`**: Data preprocessing pipeline. It reads raw CSVs using Dask, handles missing values via seasonal interpolation, feature engineers (e.g., seasons, temperature ranges) and saves the data as HuggingFace Datasets.
* **`train.ipynb`**: Model training pipeline. It loads the prepared dataset, configures the `TimeSeriesTransformer` and trains the model using PyTorch and Accelerate.
* **`test_model.ipynb`**: Evaluation pipeline. It loads the trained model, performs backtesting on the test set, calculates metrics (MASE, sMAPE) and visualizes forecast plots with confidence intervals.
* **`check_for_nan.ipynb`**: Utility notebook to audit datasets for missing values (NaN) before training.

## 🚀 Key Features

* **Model**: Time Series Transformer (via HuggingFace).
* **Target**: Daily Maximum Temperature (TMAX).
* **Prediction Horizon**: 30 days.
* **Data Processing**: 
    * Efficient large-scale data handling with **Dask**.
    * Robust missing value imputation using seasonal medians and linear interpolation.
    * Conversion to GluonTS and HuggingFace Dataset formats.
* **Evaluation**: Metrics include Mean Absolute Scaled Error (**MASE**) and Symmetric Mean Absolute Percentage Error (**sMAPE**).

## 🛠️ Tech Stack

* **Python**
* **HuggingFace Transformers** & **Datasets**
* **GluonTS** (Probabilistic Time Series Modeling)
* **PyTorch** & **Accelerate**
* **Dask** (Parallel computing)
* **Evaluate** & **Matplotlib**

## ⚙️ Usage

To reproduce the results, execute the notebooks in the following order:

1.  **Data Preparation**:
    Run `prepare_dataset.ipynb` to process the raw CSV files in `daily-summaries-latest-upm/` and generate the train/validation/test datasets.

2.  **Training**:
    Run `train.ipynb` to train the transformer model. This notebook will save the model weights and configuration to disk.

3.  **Testing & Visualization**:
    Run `test_model.ipynb` to load the saved model, generate predictions on unseen data and plot the results against ground truth.

## 📊 Results

The model generates probabilistic forecasts, providing a median prediction along with confidence intervals. Performance is evaluated using standard time-series metrics to ensure accuracy across different seasons and stations.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
