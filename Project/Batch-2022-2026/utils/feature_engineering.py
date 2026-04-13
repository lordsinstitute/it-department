from typing import List
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np


class CustomCategoricalEncoder:
    def __init__(self, columns: List[str], case: str = "nominal"):
        self.columns = columns
        self.case = case

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = X.copy()
        if self.case == "nominal":
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col]).astype(np.uint8)
        else:
            oe = OrdinalEncoder()
            output[self.columns] = oe.fit_transform(output[self.columns]).astype(
                np.uint8
            )

        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def preprocess_cat_data(data: pd.DataFrame):
    # Preprocess categorical values

    # selected this approach, because CustomCategoricalEncoder(OrdinalEncoder) gives higher values in frequent ones.
    # convert np.uint64 --> np.uint8 for less memory usage.
    data["Gender"] = data["Gender"].map({"Female": 0, "Male": 1}).astype(np.uint16)
    data["FamHist"] = data["FamHist"].map({"no": 0, "yes": 1}).astype(np.uint16)
    data["FAVC"] = data["FAVC"].map({"no": 0, "yes": 1}).astype(np.uint16)
    data["SMOKE"] = data["SMOKE"].map({"no": 0, "yes": 1}).astype(np.uint16)
    data["SCC"] = data["SCC"].map({"no": 0, "yes": 1}).astype(np.uint16)
    data["CAEC"] = (
        data["CAEC"]
        .map({"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3})
        .astype(np.uint16)
    )
    data["CALC"] = (
        data["CALC"]
        .map({"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 2})
        .astype(np.uint16)
    )


def preprocess_num_data(data: pd.DataFrame):
    # Numerical ones
    numerical_columns = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    for col in numerical_columns:
        data[col] = data[col].astype(np.float32)


def feature_engineering(data: pd.DataFrame):
    # first preprocess
    preprocess_cat_data(data)
    preprocess_num_data(data)

    # IsFemale
    data["IsFemale"] = np.where(data["Gender"] == 0, 1, 0).astype(np.uint16)

    # BMI
    data["BMI"] = data["Weight"] / (data["Height"] ** 2)

    # BMIClass (https://www.ncbi.nlm.nih.gov/books/NBK541070/)
    data["BMIClass"] = pd.cut(
        data["BMI"],
        bins=[0, 16.5, 18.5, 24.9, 29.9, 34.9, 39.9, float("inf")],
        labels=[0, 1, 2, 3, 4, 5, 6],
    ).astype(np.uint16)

    # BMI_FAF
    data["BMI_FAF"] = (data["BMI"] * data["FAF"]) / 25.0

    # AgeGroup
    data["Age_Group"] = pd.cut(
        data["Age"], bins=[0, 20, 25, 30, 40, float("inf")], labels=[0, 1, 2, 3, 4]
    ).astype(np.uint16)

    # RiskFactor
    data["RiskFactor"] = (data["BMI"] + data["Age_Group"]) * data["FamHist"]

    # NCPGroup
    data["NCPGroup"] = pd.cut(
        data["NCP"], bins=[0, 1, 2, 3, float("inf")], labels=[0, 1, 2, 3]
    ).astype(np.uint16)

    # MTRANS_Factor --> assign higher values to physical activities
    data["MTRANS"] = (
        data["MTRANS"]
        .map(
            {
                "Automobile": 0,
                "Motorbike": 0,
                "Public_Transportation": 1,
                "Walking": 2,
                "Bike": 2,
            }
        )
        .astype(np.uint16)
    )

    # HealthyActivity
    data["HealthyActivity"] = (
        (data["FAF"] * data["MTRANS"])
        + (data["NCP"] * data["FCVC"] * data["CH2O"])
        - (data["CALC"] + data["SMOKE"])
    )

    # FAVC_ratio_FCVC
    data["FAVC_ratio_FCVC"] = data["FAVC"] / data["FCVC"]

    # NCP_CAEC
    data["NCP_CAEC"] = data["NCP"] * data["CAEC"]

    # FoodTrackFactor
    data["FoodTrackFactor"] = data["FAF"] * data["SCC"]

    # Activity_TechUse
    data["Activity_TechUse"] = data["FAF"] - data["TUE"]

    # HydrateEfficiency
    data["HydrateEfficiency"] = data["CH2O"] / data["FCVC"]

    # TechUsageScore
    data["TechUsageScore"] = data["TUE"] / data["Age"]

    # BMIbyNCP
    data["BMIbyNCP"] = np.log1p(data["BMI"]) - np.log1p(data["NCP"])

    return data


def standarize_data(data: pd.DataFrame):
    standardscl = StandardScaler()
    scaled_data = standardscl.fit_transform(data)
    return scaled_data, standardscl