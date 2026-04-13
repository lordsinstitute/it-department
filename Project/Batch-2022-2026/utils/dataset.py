import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from utils.feature_engineering import feature_engineering, preprocess_cat_data, preprocess_num_data


class ObesityDataset:
    def __init__(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame, seed: int = 42
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.seed = seed
        np.random.seed(self.seed)

    def build_data(self, validation_size: float = 0.2):
        # firstly, we need to drop the ID column (it's not needed)
        self.train_data.drop("id", axis=1, inplace=True)

        # get X_features and Targets
        target = self.train_data.NObeyesdad
        x_features = self.train_data.drop("NObeyesdad", axis=1)

        # Label Encode the target variable
        # we will also return this, for decoding the test predictions!
        le = LabelEncoder()

        # split first and then apply preprocessing and feature engineering steps
        # to avoid Data Leakage train-test
        if validation_size > 0:
            x_train, x_valid, y_train, y_valid = self.make_splits(
                x_features, target, validation_size
            )

            # first fit in y_train, and the transform the y_valid
            y_train = le.fit_transform(y_train).astype(np.uint8)
            y_valid = le.transform(y_valid).astype(np.uint8)

            # apply feature engineering
            x_train = self.feature_engineering(x_train)
            x_valid = self.feature_engineering(x_valid)

            # Standarize unbalanced data for models that requires Normalized data
        #             x_train_scaled, scaler = self.standarize_data(x_train)
        #             x_valid_scaled = scaler.transform(x_valid)

        else:

            x_valid, y_valid = None, None

            # first fit in y_train, and the transform the y_valid
            y_train = le.fit_transform(target).astype(np.uint8)

            # apply feature engineering
            x_train = self.feature_engineering(x_features)

            # Standarize unbalanced data
        #             x_train_scaled, scaler = self.standarize_data(x_train)

        # apply all aforementioned steps to test data
        test_ids = self.test_data.id
        test_features = self.test_data.drop("id", axis=1)

        x_test = feature_engineering(test_features)
        #         x_test = scaler.transform(x_test)

        print(
            "\n------------------------------------------------------------------------"
        )
        print(f"Train samples: {len(x_train)} | Train targets: {len(y_train)}")
        print(
            f"Validation samples: {len(x_valid) if x_valid is not None else 0} | Validation targets: {len(y_valid) if x_valid is not None else 0}"
        )
        print(f"Test samples: {len(x_test)}")
        print(
            "\n------------------------------------------------------------------------"
        )

        return x_train, y_train, x_valid, y_valid, x_test, test_ids, le

    def make_splits(
        self, x_train: pd.DataFrame, y_train: pd.Series, test_size: float = 0.2
    ):
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=test_size, random_state=self.seed
        )
        return x_train, x_valid, y_train, y_valid

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

    def feature_engineering(self, data: pd.DataFrame):
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

