import pandas as pd
import joblib

# Load model
model = joblib.load('outputs/RF_model.pkl')
# Load symptom list

all_symptoms = pd.read_csv(
    'data/features.csv', nrows=1).columns.str.lower().tolist()


def get_user_input():
    print("\n🩺 Enter symptoms (comma-separated), e.g., headache, stomach pain, fatigue")
    input_raw = input(">> ").strip().lower()
    symptom_list = [s.strip().replace(" ", "_")
                    for s in input_raw.split(",") if s.strip()]

    # Check for unknown symptoms
    unknown = [sym for sym in symptom_list if sym not in all_symptoms]
    if unknown:
        print(f"\n⚠️ Unknown symptoms ignored: {', '.join(unknown)}")

    # Filter valid symptoms
    valid_symptoms = [sym for sym in symptom_list if sym in all_symptoms]
    return valid_symptoms


def encode_input(symptom_list):
    encoded = [1 if symptom in symptom_list else 0 for symptom in all_symptoms]
    return pd.DataFrame([encoded], columns=all_symptoms)


def main():
    print("\n🔍 Disease Prediction Based on Symptoms")
    print("📋 Type `list` to see all possible symptoms or start entering symptoms.\n")

    # Option to show available symptoms
    peek = input("Do you want to see the symptom list? (y/n): ").lower()
    if peek == "y":
        print("\n🧾 Available symptoms:\n" + ", ".join(all_symptoms))

    symptoms = get_user_input()
    if not symptoms:
        print("\n⚠️ No valid symptoms provided. Exiting.")
        return

    encoded_input = encode_input(symptoms)
    prediction = model.predict(encoded_input)[0]

    print(f"\n✅ Predicted Disease: **{prediction}**")


if __name__ == "__main__":
    main()
