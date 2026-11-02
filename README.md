# ModelScope

### Interactive ML Interpretability Platform

ModelScope is an interactive machine learning platform for exploring how different classification models make decisions.

Instead of stopping at a prediction, ModelScope lets users compare multiple models, inspect prediction confidence, understand which features influenced an individual prediction, and simulate alternative inputs through counterfactual analysis.

**Live Demo:** https://frontend-six-lac-60.vercel.app/

**API Documentation:** https://modelscope-wmbk.onrender.com/docs

---

## What can ModelScope do?

### Compare models

The same employee profile is evaluated by three different classification models:

- Logistic Regression
- Random Forest
- Gradient Boosting

ModelScope displays each model's prediction and probability, making it possible to see when models agree and when they interpret the same input differently.

### Explain predictions

ModelScope uses SHAP-based feature contributions to show which inputs push an individual prediction toward or away from employee attrition.

Instead of simply showing:

> Prediction: Attrition

the application can show:

> Overtime → pushes toward attrition  
> Stock Option Level → pushes away from attrition

This makes individual model behavior more observable.

### Explore counterfactuals

The What If? analysis simulates changes to individual features and measures how the model responds.

For example:

```text
OverTime
Yes → No

53.1% → 15.2%

Prediction changed:
ATTRITION → STAY
```

Counterfactuals describe how the trained model responds to alternative inputs. They are not causal conclusions or recommendations.

### Detect model disagreement

ModelScope identifies whether the three models agree on the final classification and reports the spread between their predicted probabilities.

For example:

```text
Logistic Regression     53.1%  ATTRITION
Random Forest           17.3%  STAY
Gradient Boosting       12.8%  STAY

2 of 3 models agree
Probability spread: 40.3 pts
```

---

## Machine Learning Results

The models were evaluated using accuracy, precision, recall, F1, and ROC-AUC.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 86.05% | 61.54% | 34.04% | 43.84% | 81.15% |
| Random Forest | 85.03% | 63.64% | 14.89% | 24.14% | 80.08% |
| Gradient Boosting | 86.05% | 71.43% | 21.28% | 32.79% | 80.57% |

Accuracy is presented alongside precision, recall, F1, and ROC-AUC rather than being treated as the sole measure of model performance.

---

## Technical Architecture

```text
                    React Frontend
                         │
                         ▼
                    FastAPI API
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Prediction Pipeline     Explainability
              │                     │
       ┌──────┼──────┐             SHAP
       │      │      │
       ▼      ▼      ▼
   Logistic  Random  Gradient
  Regression Forest  Boosting
       │      │      │
       └──────┼──────┘
              ▼
       Model Comparison
              │
       ┌──────┴─────────┐
       ▼                ▼
   Predictions     Counterfactuals
       │                │
       └───────┬────────┘
               ▼
          React Interface
```

---

## ML Pipeline

```text
Raw Employee Data
        ↓
Feature Preprocessing
        ↓
Categorical Encoding
        ↓
Numerical Feature Scaling
        ↓
Train / Test Split
        ↓
Three Classification Models
        ↓
Prediction + Probability
        ↓
SHAP Explanation
        ↓
Counterfactual Analysis
```

---

## Tech Stack

### Frontend

- React
- Vite
- JavaScript
- CSS

### Backend

- Python
- FastAPI
- Uvicorn
- Pydantic

### Machine Learning

- scikit-learn
- pandas
- NumPy
- SHAP
- joblib

### Deployment

- Vercel
- Render

---

## API

The FastAPI backend exposes endpoints for the major ModelScope capabilities:

```text
GET  /schema
GET  /models
POST /predict
POST /compare
POST /explain
POST /counterfactual
```

Interactive API documentation is available through the deployed Swagger interface:

https://modelscope-wmbk.onrender.com/docs

---

## Project Structure

```text
modelscope/
├── backend/
│   ├── data/
│   ├── models/
│   ├── main.py
│   ├── preprocessing.py
│   ├── requirements.txt
│   └── train.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── package-lock.json
│
├── .gitignore
└── README.md
```

---

## Running Locally

### Backend

```bash
cd backend
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The Vite development server will provide the local frontend URL.

---

## Limitations

ModelScope is an educational machine learning project using employee attrition data.

The predictions should not be interpreted as causal conclusions or used to make real employment decisions.

Counterfactual analysis describes how the trained model responds to simulated input changes. It does not establish that changing a particular employee attribute would cause a different real-world outcome.

---

## Future Improvements

Potential extensions include:

- Probability calibration
- Additional model architectures
- More advanced counterfactual generation
- Global feature importance analysis
- Model monitoring
- Additional datasets
- Expanded evaluation and fairness analysis

---

## Status

ModelScope is a deployed interactive ML prototype with a React frontend, FastAPI backend, multiple classification models, SHAP-based interpretability, and counterfactual analysis.

**Live:** https://frontend-six-lac-60.vercel.app/
