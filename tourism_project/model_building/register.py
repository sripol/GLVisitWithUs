import pandas as pd
import os

# Updated path with GlVisitWithUs prefix
DATA_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(DATA_PATH)

expected_columns = ["CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "Occupation", "Gender",
                    "NumberOfPersonVisiting", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
                    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome", "PitchSatisfactionScore",
                    "ProductPitched", "NumberOfFollowups", "DurationOfPitch"]

for col in expected_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

print("Dataset registered successfully")
