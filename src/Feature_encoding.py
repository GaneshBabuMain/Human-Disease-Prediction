import pandas as pd

# Load the cleaned dataset
dataset = pd.read_csv('data/dataset_cleaned.csv')

# Identify symptom columns
symptom_cols = [col for col in dataset.columns if col.startswith('Symptom_')]

# Step 1: Get all unique symptoms (excluding 'None')
all_symptoms = set()

for col in symptom_cols:
    all_symptoms.update(dataset[col].unique())

all_symptoms = {symptom for symptom in all_symptoms if isinstance(
    symptom, str) and symptom != 'None'}
all_symptoms = sorted(all_symptoms)  # Keep order for consistent encoding

# Step 2: Function to encode one patient


def encode_symptoms(row):
    symptoms = set(row[symptom_cols])
    return [1 if symptom in symptoms else 0 for symptom in all_symptoms]


# Step 3: Apply encoding to each row
X = dataset.apply(encode_symptoms, axis=1, result_type='expand')
X.columns = all_symptoms  # Set column names as symptoms

# Step 4: Prepare target labels
y = dataset['Disease']

# Optional: Save features and labels for reuse
X.to_csv('data/features.csv', index=False)
y.to_csv('data/labels.csv', index=False)

print("✅ Feature matrix shape:", X.shape)
print("✅ Target label count:", y.nunique())
