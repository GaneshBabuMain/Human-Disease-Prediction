import pandas as pd
# Load datasets
dataset = pd.read_csv('data/dataset.csv')
severity = pd.read_csv('data/Symptom-severity.csv')

# Preview first few rows
print("===== Dataset Preview =====")
print(dataset.head())

print("\n===== Symptom Severity Preview =====")
print(severity.head())

# Show dataset info
print("\n===== Dataset Info =====")
print(dataset.info())

print("\n===== Severity Info =====")
print(severity.info())

# Check for missing values
print("\n===== Missing Values in Dataset =====")
print(dataset.isnull().sum())

print("\n===== Missing Values in Severity =====")
print(severity.isnull().sum())
