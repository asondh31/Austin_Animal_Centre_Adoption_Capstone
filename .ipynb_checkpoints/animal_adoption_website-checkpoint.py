import streamlit as st
import pandas as pd
import joblib

# --- Load Model ---
try:
    model = joblib.load("gs_rf.joblib")
    model_features = joblib.load("model_features.joblib")
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please run the training script first to create 'gs_rf.joblib' and 'model_features.joblib'.")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="Animal Adoption Predictor", layout="centered")

st.title("🐾 Animal Adoption Predictor")
st.markdown("Enter an animal's characteristics to predict its likelihood of adoption at the Austin Animal Center.")

# --- Sidebar for Inputs ---
st.sidebar.header("Animal Characteristics")

animal_type = st.sidebar.selectbox("Animal Type", ["Dog", "Cat", "Other"])
age_group = st.sidebar.selectbox("Age Group", ["Puppy/Kitten", "Young Adult", "Senior"])
spayed_neutered = st.sidebar.selectbox("Spayed/Neutered", ["Yes", "No"])
breed = st.sidebar.selectbox("Breed", ["Mix", "Purebred", "Other"])
color = st.sidebar.selectbox("Primary Colour", ["Black", "White", "Brown", "Other"])
intake_condition = st.sidebar.selectbox("Intake Condition", ["Normal", "Sick", "Injured", "Other"])
intake_type = st.sidebar.selectbox("Intake Type", ["Stray", "Owner Surrender", "Public Assist", "Other"])
animal_sex = st.sidebar.selectbox("Animal Sex", ["Male", "Female"])

# --- Prediction ---
if st.button("🔮 Predict Adoption Likelihood"):

    # Encode user inputs into model features
    input_data = {
        "is_spayed_neutered": 1 if spayed_neutered == "Yes" else 0,

        # Animal type
        "animal_type_Dog": 1 if animal_type == "Dog" else 0,
        "animal_type_Cat": 1 if animal_type == "Cat" else 0,
        "animal_type_Other": 0 if animal_type == "Other" else 1,

        # Breed
        "breed_Purebred": 1 if breed == "Purebred" else 0,
        "breed_Mix": 1 if breed == "Mix" else 0,
        "breed_Other": 0 if breed == "Other" else 1,

        # Colour
        "color_Black": 0 if color == "Black" else 1,
        "color_White": 1 if color == "White" else 0,
        "color_Brown": 1 if color == "Brown" else 0,
        "color_Other": 0 if color == "Other" else 1,

        # Intake condition
        "intake_condition_Normal": 1 if intake_condition == "Normal" else 0,
        "intake_condition_Sick": 0 if intake_condition == "Sick" else 1,
        "intake_condition_Injured": 0 if intake_condition == "Injured" else 1,
        "intake_condition_Other": 1 if intake_condition == "Other" else 0,

        # Intake type
        "intake_type_Stray": 1 if intake_type == "Stray" else 0,
        "intake_type_Owner Surrender": 1 if intake_type == "Owner Surrender" else 0,
        "intake_type_Public Assist": 1 if intake_type == "Public Assist" else 0,
        "intake_type_Other": 0 if intake_type == "Other" else 1,

        # Age group
        "age_upon_intake_age_group_Puppy/Kitten": 1 if age_group == "Puppy/Kitten" else 0,
        "age_upon_intake_age_group_Young Adult": 1 if age_group == "Young Adult" else 0,
        "age_upon_intake_age_group_Senior": 0 if age_group == "Senior" else 1,

        # Sex
        "animal_sex_Male": 1 if animal_sex == "Male" else 0,
        "animal_sex_Female": 1 if animal_sex == "Female" else 0,
    }

    # Convert to DataFrame with correct model columns
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    # Predict probability of adoption
    prediction_proba = (model.predict_proba(input_df)[0][1])
    prediction_percentage = round(prediction_proba * 100, 1)

    # --- Show Results ---
    st.subheader("📊 Prediction Result")
    if prediction_proba >= 0.7:
        st.success(f"This animal has a **{prediction_percentage}%** chance of being adopted. Looks very promising! 🐶🐱")
    elif prediction_proba >= 0.4:
        st.warning(f"This animal has a **{prediction_percentage}%** chance of being adopted. It may need some extra attention. 🐾")
    else:
        st.error(f"This animal has a **{prediction_percentage}%** chance of being adopted. It may be at risk. 💔")
