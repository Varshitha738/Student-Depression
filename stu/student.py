import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st

#Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

#Preprocess data
def preprocess_data(data):
    data_cleaned = data.drop(["City", "Profession", "Unnamed: 7"], axis=1, errors="ignore").dropna(subset=["Depression"])
    le = LabelEncoder()
    data_cleaned["Gender"] = le.fit_transform(data_cleaned["Gender"])
    data_cleaned.fillna({ 
        "Age": data_cleaned["Age"].mean(), 
        "Academic Pressure": data_cleaned["Academic Pressure"].mean(), 
        "CGPA": data_cleaned["CGPA"].mean(), 
        "Study Satisfaction": data_cleaned["Study Satisfaction"].mean(), 
        "Work/Study Hours": data_cleaned["Work/Study Hours"].mean(), 
        "Financial Stress": data_cleaned["Financial Stress"].mean() 
    }, inplace=True)
    return data_cleaned

#Train model
def train_model(data_cleaned):
    selected_features = ["Age", "Gender", "Academic Pressure", "CGPA", "Study Satisfaction", "Work/Study Hours", "Financial Stress"]
    X = data_cleaned[selected_features]
    y = data_cleaned["Depression"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    return knn, scaler

#Predict depression
def predict_depression(knn, scaler, user_data):
    user_data_scaled = scaler.transform(user_data)
    prediction = knn.predict(user_data_scaled)
    prediction_binary = int(prediction >= 0.5)
    return prediction_binary

#Main function
def main():
    st.title("Student Depression Prediction")
    file_path = "Student Depression Dataset.csv"
    data = load_data(file_path)
    data_cleaned = preprocess_data(data)
    knn, scaler = train_model(data_cleaned)

    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender == "Male" else 0
    academic_pressure = st.slider("Academic Pressure (1-10)", 1, 10)
    cgpa = st.number_input("CGPA")
    study_satisfaction = st.slider("Study Satisfaction (1-10)", 1, 10)
    work_hours = st.number_input("Work/Study Hours per day")
    financial_stress = st.slider("Financial Stress (1-10)", 1, 10)

    if st.button("Predict"):
        user_data = np.array([[age, gender, academic_pressure, cgpa, study_satisfaction, work_hours, financial_stress]])
        prediction_binary = predict_depression(knn, scaler, user_data)
        result_text = "⚠️ The student is likely experiencing depression." if prediction_binary == 1 else "✅ The student is not experiencing depression."
        st.write(result_text)

if __name__ == "__main__":
    main()