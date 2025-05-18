import pandas as pd
import csv
import pickle
import numpy as np

def predict_disease(user_symptoms, days):
    with open('model/disease_model.pkl', 'rb') as f:
        model, mlb = pickle.load(f)
    user_symptoms = [s.strip().lower() for s in user_symptoms if s.strip()]
    known_symptoms = set(mlb.classes_)
    filtered_symptoms = [s for s in user_symptoms if s in known_symptoms]
    if not filtered_symptoms:  # If no valid symptoms are found
        return "No valid symptoms detected. Please enter known symptoms.", {}
    input_vector = mlb.transform([filtered_symptoms])
    predicted_diseases = model.predict(input_vector)
    unique, counts = np.unique(predicted_diseases, return_counts=True)
    sorted_diseases = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:2]
    print("Predicted Diseases:", sorted_diseases)

    # Load descriptions & precautions
    description_dict, precautions_dict = {}, {}
    with open("dataset/symptom_description.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)  # Skip header
        for row in csv_reader:
            description_dict[row[0].strip().lower()] = row[1].strip()
    with open("dataset/symptom_precaution.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)  # Skip header
        for row in csv_reader:
            disease_name = row[0].strip().lower()
            precautions_dict[disease_name] = [p.strip().capitalize() for p in row[1:] if p.strip()]

    # Build response
    output = {}
    for disease, _ in sorted_diseases:
        disease_lower = disease.strip().lower()
        output[disease] = {
            "desc": description_dict.get(disease_lower, "No description available"),
            "prec": precautions_dict.get(disease_lower, ["No precautions available"]),
            "drugs": get_drugs_for_disease(disease_lower)  # Pass lowercase disease name
        }
    return "If symptoms persist, consult a doctor.", output

def get_drugs_for_disease(disease):
    df = pd.read_csv("dataset/medicine.csv")
    df["Disease"] = df["Disease"].str.lower()
    disease_lower = disease.strip().lower()
    disease_data = df[df["Disease"].str.contains(disease_lower, na=False, case=False)]
    if disease_data.empty:
        print(f"No medicine found for {disease_lower}")
        return {"Medications": ["No drug found"], "Diet": ["No dietary recommendations"]}
    medications = eval(disease_data.iloc[0]["Medication"])
    diet = eval(disease_data.iloc[0]["Diet"])

    return {"Medications": medications, "Diet": diet}