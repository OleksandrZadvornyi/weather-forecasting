from datasets import load_from_disk
import numpy as np

# Load the dataset
data_dir = "D:/Dev/python-projects/weather-forecasting/prepared_datasets/dataset"
dataset = load_from_disk(data_dir)

def check_nan_in_dataset(dataset):
    """
    Check for NaN values in a HuggingFace Dataset
    
    Parameters:
    dataset (datasets.Dataset): The dataset to check
    
    Returns:
    dict: A summary of NaN findings
    """
    nan_summary = {
        'total_series': len(dataset),
        'series_with_nans': 0,
        'total_nans': 0,
        'nan_details': []
    }
    
    for idx, item in enumerate(dataset):
        # Check target values
        target = item['target']
        target_nans = np.isnan(target).sum()
        
        if target_nans > 0:
            nan_summary['series_with_nans'] += 1
            nan_summary['total_nans'] += target_nans
            nan_summary['nan_details'].append({
                'item_id': item.get('item_id', idx),
                'nan_count': int(target_nans),
                'total_length': len(target)
            })
    
    return nan_summary

# Check NaN in each split
splits = ['train', 'validation', 'test']
for split in splits:
    print(f"\nChecking NaNs in {split} dataset:")
    nan_result = check_nan_in_dataset(dataset[split])
    
    print(f"Total series: {nan_result['total_series']}")
    print(f"Series with NaNs: {nan_result['series_with_nans']}")
    print(f"Total NaN values: {nan_result['total_nans']}")
    
    if nan_result['series_with_nans'] > 0:
        print("\nDetailed NaN information:")
        for detail in nan_result['nan_details']:
            print(f"Item ID: {detail['item_id']}")
            print(f"  NaN count: {detail['nan_count']}")
            print(f"  Total series length: {detail['total_length']}")
            print(f"  NaN percentage: {detail['nan_count']/detail['total_length']*100:.2f}%")