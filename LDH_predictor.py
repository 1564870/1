import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('SVM_label.pkl')

# Feature names
feature_names = [
    'PfirrmannGrade',
    'ResNet50-sagittal-DTL1864',
    'ResNet50-sagittal-DTL1013',
    'ResNet50-sagittal-DTL1176',
    'ResNet50-sagittal-DTL355',
    'ResNet50-transverse-DTL1058',
    'Osteoporosis',
    'ResNet50-transverse-DTL1701',
    'ResNet50-transverse-DTL1360',
    'ResNet50-transverse-DTL472',
    'ResNet50-transverse-DTL1869',
    'ResNet50-transverse-DTL114',
    'ResNet50-sagittal-DTL1087',
    'ResNet50-sagittal-DTL1243',
    'ResNet50-transverse-DTL503',
    'ResNet50-transverse-DTL970',
    'ResNet50-transverse-DTL834',
    'ResNet50-sagittal-DTL1397',
    'Hypertension',
    'ResNet50-sagittal-DTL1236'
]

# Title and description
st.title("Post-Surgery Sensory Improvement Predictor")
st.write("Enter the following details to predict post-surgery numbness improvement.")

# Input fields
input_data = {}

# Dropdowns for categorical features
input_data['Osteoporosis'] = st.selectbox('Osteoporosis (0=No, 1=Yes):', options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
input_data['Hypertension'] = st.selectbox('Hypertension (0=No, 1=Yes):', options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
input_data['PfirrmannGrade'] = st.selectbox('Pfirrmann Grade:', options=[1, 2, 3, 4, 5], format_func=lambda x: f"Grade {x}")

# Numeric input fields for other features
for feature in feature_names[1:]:
    if feature not in ["Osteoporosis", "Hypertension", "PfirrmannGrade"]:
        input_data[feature] = st.number_input(f"{feature}:", value=0.0)

# Prediction button
if st.button("Predict"):
    # Convert inputs to a format suitable for the model
    try:
        feature_values = [input_data[feature] for feature in feature_names]
        features = np.array([feature_values])

        # Make prediction
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # Generate prediction results and advice
        result = f"Predicted Class: {predicted_class}\n"
        result += f"Prediction Probabilities: {predicted_proba}\n"
        advice = ""
        
        probability = predicted_proba[predicted_class] * 100
        if predicted_class == 1:
            advice = (
                f"\nBased on our model's prediction, your numbness improvement after the surgery is significant. "
                f"The model predicts a {probability:.1f}% likelihood of significant postoperative improvement in numbness."
            )
        else:
            advice = (
                f"\nBased on our model's prediction, your numbness improvement after the surgery is limited. "
                f"The model predicts a {probability:.1f}% likelihood of limited improvement."
            )
        
        # Display result and advice
        st.success(result + advice)

    except Exception as e:
        st.error(f"An error occurred: {e}")
