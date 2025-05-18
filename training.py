import pandas as pd
import pickle
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score, f1_score, root_mean_squared_error,
    confusion_matrix, ConfusionMatrixDisplay
)

def save_metrics(metrics, filename):
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=2)

def Train_Crop_Recommendation():
    data = pd.read_csv("dataset/Crop_recommendation.csv")
    features = ['N','P','K','temperature','humidity','ph','rainfall']
    target = 'label'
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_value = f1_score(y_test, y_pred, average='weighted')
    metrics = {
        "accuracy": round(accuracy, 4)*100,
        "f1_score": round(f1_value, 4)*100
    }
    save_metrics(metrics, "analytics/crop_recommendation_metrics.json")
    with open("model/crop_predict_model.pkl", "wb") as f:
        pickle.dump(model, f)
    # Confusion Matrix
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(include_values=False, cmap='Blues', ax=plt.gca(), xticks_rotation=45)
    plt.title("Crop Recommendation Confusion Matrix")
    plt.tight_layout()
    plt.savefig("maps/crop_recommendation_confusion_matrix.png")
    plt.close()

def Train_Crop_Price():
    data = pd.read_csv("dataset/Crop_Price.csv")
    le_state = LabelEncoder()
    le_crop = LabelEncoder()
    data['State'] = le_state.fit_transform(data['State'])
    data['Crop'] = le_crop.fit_transform(data['Crop'])
    features = ['State', 'Crop', 'CostCultivation', 'CostCultivation2', 'Production', 'Yield', 'Temperature', 'RainFall Annual']
    target = 'Price'
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "r2_score": round(r2, 4)
    }
    save_metrics(metrics, "analytics/crop_price_metrics.json")
    with open("model/crop_price_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/state_encoder.pkl", "wb") as f:
        pickle.dump(le_state, f)
    with open("model/crop_encoder.pkl", "wb") as f:
        pickle.dump(le_crop, f)
    # Actual vs Predicted
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Crop Price: Actual vs Predicted")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    plt.savefig("maps/crop_price_actual_vs_pred.png")
    plt.close()
    # Residual Matrix
    residuals = y_test - y_pred
    plt.figure(figsize=(8,6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Crop Price: Residual Matrix")
    plt.tight_layout()
    plt.savefig("maps/crop_price_residual_matrix.png")
    plt.close()

def Train_Fertilizer():
    data = pd.read_csv("dataset/fertilizer.csv")
    data['Soil'] = data['Soil'].str.strip().str.title()
    data['Crop'] = data['Crop'].str.strip().str.lower()
    data['Fertilizer'] = data['Fertilizer'].str.strip()
    le_soil = LabelEncoder()
    le_crop = LabelEncoder()
    le_fertilizer = LabelEncoder()
    data['Soil'] = le_soil.fit_transform(data['Soil'])
    data['Crop'] = le_crop.fit_transform(data['Crop'])
    data['Fertilizer'] = le_fertilizer.fit_transform(data['Fertilizer'])
    joblib.dump(le_soil, 'model/soil_encoder.pkl')
    joblib.dump(le_crop, 'model/fertilizer_crop_encoder.pkl')
    joblib.dump(le_fertilizer, 'model/fertilizer_encoder.pkl')
    X = data.drop(['Fertilizer', 'Remark'], axis=1)
    y = data['Fertilizer']
    scaler = StandardScaler()
    scaler.fit(X) 
    X_scaled = scaler.transform(X)
    joblib.dump(scaler, 'model/fertilizer_scaler.pkl')
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_value = f1_score(y_test, y_pred, average='weighted')
    metrics = {
        "accuracy": round(accuracy, 4)*100,
        "f1_score": round(f1_value, 4)*100
    }
    save_metrics(metrics, "analytics/fertilizer_metrics.json")
    joblib.dump(model, 'model/fertilizer_model.pkl')
    # Confusion Matrix
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(include_values=False, cmap='Blues', ax=plt.gca())
    plt.title("Fertilizer Confusion Matrix")
    plt.tight_layout()
    plt.savefig("maps/fertilizer_confusion_matrix.png")
    plt.close()

def Train_Disease():
    df = pd.read_csv("dataset/symptoms_df.csv")
    df['Symptoms'] = df[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']].values.tolist()
    df['Symptoms'] = df['Symptoms'].apply(lambda x: list(set(s.strip().lower() for s in x if pd.notnull(s) and s.strip())))
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df['Symptoms'])
    y = df['Disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_value = f1_score(y_test, y_pred, average='weighted')
    metrics = {
        "accuracy": round(accuracy, 4)*100,
        "f1_score": round(f1_value, 4)*100
    }
    save_metrics(metrics, "analytics/disease_metrics.json")
    with open('model/disease_model.pkl', 'wb') as f:
        pickle.dump((model, mlb), f)
    # Confusion Matrix
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(include_values=False, cmap='Blues', ax=plt.gca(), xticks_rotation=45)
    plt.title("Disease Confusion Matrix")
    plt.tight_layout()
    plt.savefig("maps/disease_confusion_matrix.png")
    plt.close()

def Train_Breast_Cancer():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    top_features = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension'
    ]
    top_indices = [list(feature_names).index(f) for f in top_features]
    X_top = X[:, top_indices]
    X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics = {
        "accuracy": round(acc, 4)*100,
        "f1_score": round(f1, 4)*100
    }
    save_metrics(metrics, "analytics/breast_cancer_metrics.json")
    with open("model/breast_cancer_model.pkl", "wb") as f:
        pickle.dump(model, f)
    # Confusion Matrix
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(include_values=False, cmap='Blues', ax=plt.gca())
    plt.title("Breast Cancer Confusion Matrix")
    plt.tight_layout()
    plt.savefig("maps/breast_cancer_confusion_matrix.png")
    plt.close()

# call function
# Train_Crop_Recommendation()
# Train_Crop_Price()
# Train_Disease()
# Train_Breast_Cancer()
# Train_Fertilizer()