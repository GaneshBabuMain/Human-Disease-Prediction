import pandas as pd
import os

# File paths
raw_data_path = 'data/dataset.csv'
cleaned_data_path = 'data/dataset_cleaned.csv'

# Load dataset
dataset = pd.read_csv(raw_data_path)

# Fill missing values with 'None'
dataset.fillna('None', inplace=True)

# Normalize symptom strings
symptom_cols = [col for col in dataset.columns if col.startswith('Symptom_')]
for col in symptom_cols:
    dataset[col] = dataset[col].apply(
        lambda x: x.strip().lower() if x != 'None' else x)

# Save only if changed
if os.path.exists(cleaned_data_path):
    existing = pd.read_csv(cleaned_data_path)
    if not dataset.equals(existing):
        dataset.to_csv(cleaned_data_path, index=False)
else:
    dataset.to_csv(cleaned_data_path, index=False)
