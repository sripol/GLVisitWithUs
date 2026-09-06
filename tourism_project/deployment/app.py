import streamlit as st
import pandas as pd
import joblib
import os

# App loads model from deployment/
model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(model_path)

st.title("Tourism Purchase Prediction")

# Feature values to be collected from streamlit page 
st.header("Enter Customer Details")

Age = st.number_input("Age", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
Passport = st.selectbox("Passport", ["No", "Yes"])
OwnCar = st.selectbox("Own Car", ["No", "Yes"])
DurationOfPitch = st.number_input("Duration of Pitch")
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting")
NumberOfFollowups = st.number_input("Number of Follow-ups" )
NumberOfTrips = st.number_input("Number of Trips")
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting")
MonthlyIncome = st.number_input("Monthly Income")
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score")
PreferredPropertyStar = st.number_input("Preferred Hotel Rating")

# Create input dataframe
input_df = pd.DataFrame([{
"Age" : Age,
"TypeofContact" : TypeofContact,
"CityTier" : CityTier,
"Occupation" : Occupation,
"Gender" : Gender,
"NumberOfPersonVisiting" : NumberOfPersonVisiting,
"PreferredPropertyStar" : PreferredPropertyStar,
"MaritalStatus" : MaritalStatus,
"NumberOfTrips" : NumberOfTrips,
"Passport" : Passport,
"OwnCar" : OwnCar,
"NumberOfChildrenVisiting" : NumberOfChildrenVisiting,
"Designation" : Designation,
"MonthlyIncome" : MonthlyIncome,
"PitchSatisfactionScore" : PitchSatisfactionScore,
"ProductPitched" : ProductPitched,
"NumberOfFollowups" : NumberOfFollowups,
"DurationOfPitch" : DurationOfPitch
}])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)
    result = "Will Purchase" if prediction[0] == 1 else "Will Not Purchase"
    st.subheader(f"Result: {result}")
