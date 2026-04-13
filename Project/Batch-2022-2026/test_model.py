import pandas as pd
import joblib

# Load trained model
model = joblib.load('Accident_model.pkl')

# Features expected by the model
FEATURE_COLUMNS = model.feature_names_in_

def predict_accident_severity(input_data: dict):
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # Add missing columns with default value (0)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training
    df = df[FEATURE_COLUMNS]

    # Prediction
    prediction = model.predict(df)

    severity_map = {
        0: "Slight Injury",
        1: "Serious Injury",
        2: "Fatal Injury"
    }

    return severity_map.get(prediction[0], "Unknown")

# ---------------- TEST ----------------
if __name__ == "__main__":
    sample_input = {
        'Age_band_of_driver': 2,
        'Driving_experience': 1,
        'Type_of_vehicle': 3,
        'Area_accident_occured': 4,
        'Road_allignment': 1,
        'Type_of_collision': 2,
        'Vehicle_movement': 1,
        'Lanes_or_Medians': 2,
        'Types_of_Junction': 1,
        'Age_band_of_casualty': 2,
        'Casualty_class': 1,
        'Cause_of_accident': 3,
        'Light_conditions': 1,
        'Number_of_casualties': 1,
        'Weather_conditions': 2,
        'Road_surface_conditions': 1
        # Missing columns will be auto-added safely
    }

    result = predict_accident_severity(sample_input)
    print("Predicted Accident Severity:", result)
