from pathlib import Path

import joblib
import numpy as np
import shap
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


app = FastAPI(
    title="ModelScope API",
    description="API for exploring machine learning model predictions.",
    version="1.0.0",
)


MODELS = {
    "logistic_regression": joblib.load(
        MODEL_DIR / "logistic_regression.joblib"
    ),
    "random_forest": joblib.load(
        MODEL_DIR / "random_forest.joblib"
    ),
    "gradient_boosting": joblib.load(
        MODEL_DIR / "gradient_boosting.joblib"
    ),
}

@app.get("/schema")
def get_schema():
    return {
        "features": {
            "Age": {
                "type": "number",
                "min": 18,
                "max": 100
            },
            "BusinessTravel": {
                "type": "select",
                "options": [
                    "Travel_Rarely",
                    "Travel_Frequently",
                    "Non-Travel"
                ]
            },
            "DailyRate": {
                "type": "number"
            },
            "Department": {
                "type": "select",
                "options": [
                    "Sales",
                    "Research & Development",
                    "Human Resources"
                ]
            },
            "DistanceFromHome": {
                "type": "number",
                "min": 1
            },
            "Education": {
                "type": "number",
                "min": 1,
                "max": 5
            },
            "EducationField": {
                "type": "select",
                "options": [
                    "Life Sciences",
                    "Medical",
                    "Marketing",
                    "Technical Degree",
                    "Human Resources",
                    "Other"
                ]
            },
            "EnvironmentSatisfaction": {
                "type": "number",
                "min": 1,
                "max": 4
            },
            "Gender": {
                "type": "select",
                "options": [
                    "Male",
                    "Female"
                ]
            },
            "JobInvolvement": {
                "type": "number",
                "min": 1,
                "max": 4
            },
            "JobLevel": {
                "type": "number",
                "min": 1,
                "max": 5
            },
            "JobRole": {
                "type": "select",
                "options": [
                    "Sales Executive",
                    "Research Scientist",
                    "Laboratory Technician",
                    "Manufacturing Director",
                    "Healthcare Representative",
                    "Manager",
                    "Sales Representative",
                    "Research Director",
                    "Human Resources"
                ]
            },
            "JobSatisfaction": {
                "type": "number",
                "min": 1,
                "max": 4
            },
            "MaritalStatus": {
                "type": "select",
                "options": [
                    "Single",
                    "Married",
                    "Divorced"
                ]
            },
            "MonthlyIncome": {
                "type": "number",
                "min": 0
            },
            "MonthlyRate": {
                "type": "number",
                "min": 0
            },
            "NumCompaniesWorked": {
                "type": "number",
                "min": 0
            },
            "OverTime": {
                "type": "select",
                "options": [
                    "Yes",
                    "No"
                ]
            },
            "PercentSalaryHike": {
                "type": "number",
                "min": 0
            },
            "PerformanceRating": {
                "type": "number",
                "min": 1,
                "max": 5
            },
            "RelationshipSatisfaction": {
                "type": "number",
                "min": 1,
                "max": 4
            },
            "StockOptionLevel": {
                "type": "number",
                "min": 0,
                "max": 3
            },
            "TotalWorkingYears": {
                "type": "number",
                "min": 0
            },
            "TrainingTimesLastYear": {
                "type": "number",
                "min": 0
            },
            "WorkLifeBalance": {
                "type": "number",
                "min": 1,
                "max": 4
            },
            "YearsAtCompany": {
                "type": "number",
                "min": 0
            },
            "YearsInCurrentRole": {
                "type": "number",
                "min": 0
            },
            "YearsSinceLastPromotion": {
                "type": "number",
                "min": 0
            },
            "YearsWithCurrManager": {
                "type": "number",
                "min": 0
            },
            "HourlyRate": {
                "type": "number",
                "min": 0
            }
        }
    }

def get_feature_names(model):
    preprocessor = model.named_steps["preprocessor"]

    return preprocessor.get_feature_names_out()


def get_shap_values(model, input_data):

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    data_path = (
        BASE_DIR
        / "data"
        / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    background_df = pd.read_csv(data_path)

    background_df = background_df.drop(
        columns=[
            "EmployeeCount",
            "EmployeeNumber",
            "Over18",
            "StandardHours",
            "Attrition",
        ]
    )

    background_sample = background_df.sample(
        n=min(100, len(background_df)),
        random_state=42,
    )

    transformed_background = preprocessor.transform(
        background_sample
    )

    transformed_input = preprocessor.transform(
        input_data
    )

    if hasattr(transformed_background, "toarray"):
        transformed_background = transformed_background.toarray()

    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()

    feature_names = preprocessor.get_feature_names_out()

    if isinstance(
        classifier,
        (
            __import__(
                "sklearn.linear_model",
                fromlist=["LogisticRegression"],
            ).LogisticRegression,
        ),
    ):
        explainer = shap.LinearExplainer(
            classifier,
            transformed_background,
        )
    else:
        explainer = shap.TreeExplainer(
            classifier
        )

    explanation = explainer(
        transformed_input
    )

    values = np.asarray(
        explanation.values
    )

    if values.ndim == 3:
        values = values[:, :, 1]

    values = values[0]

    # Aggregate one-hot encoded features back
    # into their original feature names.
    aggregated = {}

    for feature_name, value in zip(
        feature_names,
        values,
    ):
        value = float(value)

        if feature_name.startswith("num__"):
            original_name = feature_name.replace(
                "num__",
                "",
                1,
            )

        elif feature_name.startswith("cat__"):
            encoded_name = feature_name.replace(
                "cat__",
                "",
                1,
            )

            original_name = encoded_name.split(
                "_",
                1,
            )[0]

        else:
            original_name = feature_name

        if original_name not in aggregated:
            aggregated[original_name] = 0.0

        aggregated[original_name] += value

    return aggregated

class PredictionRequest(BaseModel):
    model: str
    features: dict


@app.get("/")
def root():
    return {
        "name": "ModelScope API",
        "status": "running",
    }


@app.get("/models")
def get_models():
    return {
        "models": list(MODELS.keys())
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    if request.model not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {request.model}",
        )

    model = MODELS[request.model]

    try:
        input_data = pd.DataFrame([request.features])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return {
            "model": request.model,
            "prediction": "Yes" if prediction == 1 else "No",
            "probability": round(float(probability), 4),
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )
    
@app.post("/compare")
def compare(request: PredictionRequest):

    input_data = pd.DataFrame([request.features])

    results = []

    for model_name, model in MODELS.items():

        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            results.append({
                "model": model_name,
                "prediction": "Yes" if prediction == 1 else "No",
                "probability": round(float(probability), 4),
            })

        except Exception as error:
            raise HTTPException(
                status_code=400,
                detail=f"{model_name}: {str(error)}",
            )

    return {
        "results": results
    }

@app.post("/explain")
def explain(request: PredictionRequest):

    if request.model not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {request.model}",
        )

    model = MODELS[request.model]

    try:
        input_data = pd.DataFrame(
            [request.features]
        )

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        contributions = get_shap_values(
            model,
            input_data,
        )

        explanation = []

        for feature, value in contributions.items():

            explanation.append({
                "feature": feature,
                "contribution": round(
                    value,
                    6,
                ),
                "direction": (
                    "pushes_toward_attrition"
                    if value > 0
                    else "pushes_away_from_attrition"
                ),
            })

        explanation.sort(
            key=lambda x: abs(
                x["contribution"]
            ),
            reverse=True,
        )

        return {
            "model": request.model,
            "prediction": (
                "Yes"
                if prediction == 1
                else "No"
            ),
            "probability": round(
                float(probability),
                4,
            ),
            "top_features": explanation[:10],
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

def generate_counterfactuals(model, input_data):
    original_probability = float(
        model.predict_proba(input_data)[0][1]
    )

    candidates = {
        "OverTime": ["Yes", "No"],
        "JobSatisfaction": [1, 2, 3, 4],
        "EnvironmentSatisfaction": [1, 2, 3, 4],
        "JobInvolvement": [1, 2, 3, 4],
        "RelationshipSatisfaction": [1, 2, 3, 4],
        "WorkLifeBalance": [1, 2, 3, 4],
        "StockOptionLevel": [0, 1, 2, 3],
        "YearsSinceLastPromotion": [0, 1, 2, 3, 4, 5],
        "YearsInCurrentRole": [0, 1, 2, 3, 4, 5],
        "YearsWithCurrManager": [0, 1, 2, 3, 4, 5],
    }

    results = []

    for feature, possible_values in candidates.items():

        current_value = input_data.iloc[0][feature]

        for new_value in possible_values:

            if new_value == current_value:
                continue

            modified_input = input_data.copy()

            modified_input.at[
                modified_input.index[0],
                feature,
            ] = new_value

            new_probability = float(
                model.predict_proba(
                    modified_input
                )[0][1]
            )

            new_prediction = int(
                model.predict(modified_input)[0]
            )

            probability_change = (
                new_probability
                - original_probability
            )

            results.append({
                "feature": feature,
                "from": current_value.item()
                if hasattr(current_value, "item")
                else current_value,
                "to": new_value.item()
                if hasattr(new_value, "item")
                else new_value,
                "original_probability": round(
                    original_probability,
                    4,
                ),
                "new_probability": round(
                    new_probability,
                    4,
                ),
                "change": round(
                    probability_change,
                    4,
                ),
                "new_prediction": (
                    "Yes"
                    if new_prediction == 1
                    else "No"
                ),
                "prediction_changed": (
                    new_prediction
                    != int(model.predict(input_data)[0])
                ),
            })

    results.sort(
        key=lambda x: x["change"]
    )

    return results

@app.post("/counterfactual")
def counterfactual(request: PredictionRequest):

    if request.model not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {request.model}",
        )

    model = MODELS[request.model]

    try:
        input_data = pd.DataFrame(
            [request.features]
        )

        original_prediction = model.predict(
            input_data
        )[0]

        original_probability = float(
            model.predict_proba(
                input_data
            )[0][1]
        )

        results = generate_counterfactuals(
            model,
            input_data,
        )

        # Keep the strongest probability reductions.
        best_changes = results[:5]

        return {
            "model": request.model,
            "original_prediction": (
                "Yes"
                if original_prediction == 1
                else "No"
            ),
            "original_probability": round(
                original_probability,
                4,
            ),
            "counterfactuals": best_changes,
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )
    