# ModelScope

ModelScope is an interactive machine learning interpretability platform for exploring how different classification models make predictions.

The project uses an employee attrition dataset to compare three machine learning models, examine their confidence, explain individual predictions, and simulate counterfactual changes to understand how different inputs affect a prediction.

## Why ModelScope?

A model prediction alone does not explain why the prediction happened.

ModelScope was built to make that process more transparent by showing:

- How different models evaluate the same profile
- How confident each model is
- Which features push a prediction toward or away from attrition
- How changing an individual feature can alter the prediction
- When different models disagree

## Features

### Multi-model comparison

Each employee profile is evaluated by three classification models:

- Logistic Regression
- Random Forest
- Gradient Boosting

The interface displays each model's predicted probability of attrition and final prediction.

### Model agreement

ModelScope compares the three predictions and identifies whether the models agree or disagree.

It also reports the probability spread between the models to show how differently they evaluate the same profile.

### Explainable predictions

ModelScope uses SHAP-based explanations to identify the features that contribute most strongly to an individual prediction.

Features are separated into:

- Pushes toward attrition
- Pushes away from attrition

### Counterfactual analysis

The What If? section simulates changes to individual employee attributes and measures how the model's prediction responds.

For example:

```text
OverTime: Yes → No

53.1% attrition probability
        ↓
15.2% attrition probability

Prediction: ATTRITION → STAY