import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DROP_COLUMNS = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours",
]


def load_data(path):
    df = pd.read_csv(path)

    df = df.drop(columns=DROP_COLUMNS)

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"].map({"No": 0, "Yes": 1})

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ]
    )

    return X, y, preprocessor