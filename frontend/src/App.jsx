import { useEffect, useState } from "react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";

const FIELD_LABELS = {
  Age: "Age",
  BusinessTravel: "Business Travel",
  DailyRate: "Daily Rate",
  Department: "Department",
  DistanceFromHome: "Distance From Home",
  Education: "Education Level",
  EducationField: "Education Field",
  EnvironmentSatisfaction: "Environment Satisfaction",
  Gender: "Gender",
  JobInvolvement: "Job Involvement",
  JobLevel: "Job Level",
  JobRole: "Job Role",
  JobSatisfaction: "Job Satisfaction",
  MaritalStatus: "Marital Status",
  MonthlyIncome: "Monthly Income",
  MonthlyRate: "Monthly Rate",
  NumCompaniesWorked: "Companies Previously Worked At",
  OverTime: "Overtime",
  PercentSalaryHike: "Percent Salary Hike",
  PerformanceRating: "Performance Rating",
  RelationshipSatisfaction: "Relationship Satisfaction",
  StockOptionLevel: "Stock Option Level",
  TotalWorkingYears: "Total Working Years",
  TrainingTimesLastYear: "Training Times Last Year",
  WorkLifeBalance: "Work-Life Balance",
  YearsAtCompany: "Years At Company",
  YearsInCurrentRole: "Years In Current Role",
  YearsSinceLastPromotion: "Years Since Last Promotion",
  YearsWithCurrManager: "Years With Current Manager",
  HourlyRate: "Hourly Rate",
};

const FIELD_GROUPS = [
  {
    title: "Work & Role",
    fields: [
      "Department",
      "JobRole",
      "JobLevel",
      "BusinessTravel",
      "OverTime",
      "TotalWorkingYears",
      "YearsAtCompany",
      "YearsInCurrentRole",
      "YearsSinceLastPromotion",
      "YearsWithCurrManager",
    ],
  },
  {
    title: "Satisfaction & Performance",
    fields: [
      "JobSatisfaction",
      "EnvironmentSatisfaction",
      "JobInvolvement",
      "RelationshipSatisfaction",
      "WorkLifeBalance",
      "PerformanceRating",
      "TrainingTimesLastYear",
      "PercentSalaryHike",
    ],
  },
  {
    title: "Compensation",
    fields: [
      "MonthlyIncome",
      "MonthlyRate",
      "HourlyRate",
      "StockOptionLevel",
    ],
  },
  {
    title: "Personal",
    fields: [
      "Age",
      "Gender",
      "MaritalStatus",
      "Education",
      "EducationField",
      "DistanceFromHome",
      "NumCompaniesWorked",
      "DailyRate",
    ],
  },
];

const MODEL_LABELS = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  gradient_boosting: "Gradient Boosting",
};

const DEMO_PROFILE = {
  Age: 31,
  BusinessTravel: "Travel_Rarely",
  DailyRate: 500,
  Department: "Research & Development",
  DistanceFromHome: 10,
  Education: 3,
  EducationField: "Life Sciences",
  EnvironmentSatisfaction: 3,
  Gender: "Male",
  JobInvolvement: 3,
  JobLevel: 2,
  JobRole: "Research Scientist",
  JobSatisfaction: 2,
  MaritalStatus: "Single",
  MonthlyIncome: 4200,
  MonthlyRate: 15000,
  NumCompaniesWorked: 1,
  OverTime: "Yes",
  PercentSalaryHike: 13,
  PerformanceRating: 3,
  RelationshipSatisfaction: 3,
  StockOptionLevel: 0,
  TotalWorkingYears: 6,
  TrainingTimesLastYear: 2,
  WorkLifeBalance: 2,
  YearsAtCompany: 3,
  YearsInCurrentRole: 2,
  YearsSinceLastPromotion: 1,
  YearsWithCurrManager: 2,
  HourlyRate: 65,
};


function App() {
  const [schema, setSchema] = useState(null);
  const [formData, setFormData] = useState({});
  const [results, setResults] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [counterfactuals, setCounterfactuals] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API_URL}/schema`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to load schema");
        }

        return response.json();
      })
      .then((data) => {
        setSchema(data);

        const defaults = {};

        Object.entries(data.features).forEach(
          ([feature, config]) => {
            if (config.type === "select") {
              defaults[feature] = config.options[0];
            } else {
              defaults[feature] = config.min ?? 0;
            }
          }
        );

        setFormData(defaults);
      })
      .catch((err) => {
        console.error(err);
        setError("Unable to connect to ModelScope API.");
      });
  }, []);

  function resetExperiment() {
  if (!schema) {
    return;
  }

  const defaults = {};

  Object.entries(schema.features).forEach(
    ([feature, config]) => {
      if (config.type === "select") {
        defaults[feature] = config.options[0];
      } else {
        defaults[feature] = config.min ?? 0;
      }
    }
  );

  setFormData(defaults);
  setResults(null);
  setExplanation(null);
  setCounterfactuals(null);
  setError("");
}

  function handleChange(feature, value) {
    setFormData((previous) => ({
      ...previous,
      [feature]: value,
    }));
  }

  async function runExperiment() {
    setRunning(true);
    setError("");
    setResults(null);
    setExplanation(null);
    setCounterfactuals(null);

    const requestBody = {
      model: "logistic_regression",
      features: formData,
    };

    try {
      const [compareResponse, explainResponse, counterfactualResponse] =
        await Promise.all([
          fetch(`${API_URL}/compare`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(requestBody),
          }),

          fetch(`${API_URL}/explain`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(requestBody),
          }),

          fetch(`${API_URL}/counterfactual`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(requestBody),
          }),
        ]);

      if (
        !compareResponse.ok ||
        !explainResponse.ok ||
        !counterfactualResponse.ok
      ) {
        throw new Error("One or more experiment requests failed.");
      }

      const [compareData, explainData, counterfactualData] =
        await Promise.all([
          compareResponse.json(),
          explainResponse.json(),
          counterfactualResponse.json(),
        ]);

      setResults(compareData.results);
      setExplanation(explainData);
      setCounterfactuals(counterfactualData);
    } catch (err) {
      console.error(err);
      setError(
        "Something went wrong while running the experiment."
      );
    } finally {
      setRunning(false);
    }
  }

  function renderField(feature) {
    const config = schema.features[feature];

    return (
      <div className="field" key={feature}>
        <label htmlFor={feature}>
          {FIELD_LABELS[feature] || feature}
        </label>

        {config.type === "select" ? (
          <select
            id={feature}
            value={formData[feature] ?? ""}
            onChange={(event) =>
              handleChange(feature, event.target.value)
            }
          >
            {config.options.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        ) : (
          <input
            id={feature}
            type="number"
            value={formData[feature] ?? ""}
            min={config.min}
            max={config.max}
            onChange={(event) =>
              handleChange(
                feature,
                event.target.value === ""
                  ? ""
                  : Number(event.target.value)
              )
            }
          />
        )}
      </div>
    );
  }

  function formatContribution(value) {
    return Math.abs(value).toFixed(2);
  }

  function getMaxContribution() {
    if (!explanation?.top_features?.length) {
      return 1;
    }

    return Math.max(
      ...explanation.top_features.map((item) =>
        Math.abs(item.contribution)
      )
    );
  }

  if (!schema && !error) {
    return (
      <main className="loading-screen">
        <div className="loading-spinner" />
        <p>Loading ModelScope...</p>
      </main>
    );
  }

  if (!schema) {
    return (
      <main className="loading-screen">
        <h1>ModelScope</h1>
        <p>{error}</p>
      </main>
    );
  }

  const towardAttrition =
    explanation?.top_features?.filter(
      (item) =>
        item.direction === "pushes_toward_attrition"
    ) || [];

  const awayFromAttrition =
    explanation?.top_features?.filter(
      (item) =>
        item.direction === "pushes_away_from_attrition"
    ) || [];

    function getModelAgreement() {
    if (!results || results.length === 0) {
      return null;
    }

    const predictions = results.map(
      (result) => result.prediction
    );

    const yesCount = predictions.filter(
      (prediction) => prediction === "Yes"
    ).length;

    const noCount = predictions.length - yesCount;

    const majorityPrediction =
      yesCount >= noCount ? "Yes" : "No";

    const majorityCount = Math.max(
      yesCount,
      noCount
    );

    const probabilities = results.map(
      (result) => result.probability
    );

    const highestProbability = Math.max(
      ...probabilities
    );

    const lowestProbability = Math.min(
      ...probabilities
    );

    return {
      majorityPrediction,
      majorityCount,
      totalModels: results.length,
      probabilitySpread:
        highestProbability - lowestProbability,
      unanimous:
        yesCount === results.length ||
        noCount === results.length,
    };
  }

  const agreement = getModelAgreement();

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-content">
          <div className="eyebrow">ML EXPERIMENTATION</div>

          <h1>ModelScope</h1>

          <p>
            Explore how different machine learning models
            interpret the same prediction problem.
          </p>
        </div>
      </header>

      <main className="content">
        <section className="panel experiment-panel">
          <div className="section-heading">
            <div>
              <span className="section-number">01</span>
              <h2>Build an experiment</h2>
            </div>

            <p>
              Adjust the employee profile and see how the
              models respond.
            </p>
          </div>

          <div className="field-groups">
            {FIELD_GROUPS.map((group) => (
              <div className="field-group" key={group.title}>
                <h3>{group.title}</h3>

                <div className="field-grid">
                  {group.fields
                    .filter((field) => schema.features[field])
                    .map(renderField)}
                </div>
              </div>
            ))}
          </div>

          <button
            className="demo-button"
            onClick={() => {
              setFormData(DEMO_PROFILE);
              setResults(null);
              setExplanation(null);
              setCounterfactuals(null);
              setError("");
            }}
            disabled={running}
          >
            Load demonstration profile
          </button> 
          
          <button
            type="button"
            className="secondary-button"
            onClick={resetExperiment}
            disabled={running}
          >
            Reset experiment
          </button>

          <button
            className="run-button"
            onClick={runExperiment}
            disabled={running}
          >
            {running ? (
              <>
                <span className="button-spinner" />
                Running experiment...
              </>
            ) : (
              <>
                Run experiment
                <span>→</span>
              </>
            )}
          </button>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </section>

        {results && (
          <>
            <section className="panel">
              <div className="section-heading">
                <div>
                  <span className="section-number">02</span>
                  <h2>Model comparison</h2>
                </div>

                <p>
                  The same profile evaluated by three different
                  models.
                </p>
              </div>

              <div className="model-grid">
                {results.map((result) => (
                  <div
                    className="model-card"
                    key={result.model}
                  >
                    <div className="model-card-header">
                      <span className="model-name">
                        {MODEL_LABELS[result.model] ||
                          result.model}
                      </span>

                      <span
                        className={`prediction-badge ${
                          result.prediction === "Yes"
                            ? "prediction-yes"
                            : "prediction-no"
                        }`}
                      >
                        {result.prediction === "Yes"
                          ? "ATTRITION"
                          : "STAY"}
                      </span>
                    </div>

                    <div className="probability">
                      {(result.probability * 100).toFixed(1)}
                      <span>%</span>
                    </div>

                    <div className="probability-label">
                      Predicted probability of attrition
                    </div>

                    <div className="probability-track">
                      <div
                        className="probability-fill"
                        style={{
                          width: `${
                            result.probability * 100
                          }%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {agreement && (
              <section className="insight-banner">
                <div className="insight-icon">
                  {agreement.unanimous ? "✓" : "!"}
                </div>

                <div className="insight-content">
                  <strong>
                    {agreement.unanimous
                      ? `All ${agreement.totalModels} models agree`
                      : `${agreement.majorityCount} of ${agreement.totalModels} models agree`}
                  </strong>

                  <p>
                    {agreement.unanimous
                      ? `Every model predicts ${
                          agreement.majorityPrediction === "Yes"
                            ? "attrition"
                            : "the employee will stay"
                        }.`
                      : `The majority predicts ${
                          agreement.majorityPrediction === "Yes"
                            ? "attrition"
                            : "the employee will stay"
                        }, but the models disagree.`}
                  </p>
                </div>

                <div className="spread-stat">
                  <span>Probability spread</span>
                  <strong>
                    {(agreement.probabilitySpread * 100).toFixed(1)}
                    <small> pts</small>
                  </strong>
                </div>
              </section>
            )}
            </>
        )}

        {explanation && (
          <section className="panel">
            <div className="section-heading">
              <div>
                <span className="section-number">03</span>
                <h2>Why did the model decide this?</h2>
              </div>

              <p>
                Feature contributions from{" "}
                {MODEL_LABELS[explanation.model] ||
                  explanation.model}
                .
              </p>
            </div>

            <div className="explanation-grid">
              <div className="explanation-column">
                <h3 className="positive-heading">
                  Pushes toward attrition
                </h3>

                {towardAttrition.map((item) => (
                  <div
                    className="contribution"
                    key={item.feature}
                  >
                    <div className="contribution-header">
                      <span>
                        {FIELD_LABELS[item.feature] ||
                          item.feature}
                      </span>

                      <span className="contribution-value positive">
                        +{formatContribution(
                          item.contribution
                        )}
                      </span>
                    </div>

                    <div className="contribution-track">
                      <div
                        className="contribution-fill positive-fill"
                        style={{
                          width: `${
                            (Math.abs(item.contribution) /
                              getMaxContribution()) *
                            100
                          }%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="explanation-column">
                <h3 className="negative-heading">
                  Pushes away from attrition
                </h3>

                {awayFromAttrition.map((item) => (
                  <div
                    className="contribution"
                    key={item.feature}
                  >
                    <div className="contribution-header">
                      <span>
                        {FIELD_LABELS[item.feature] ||
                          item.feature}
                      </span>

                      <span className="contribution-value negative">
                        {formatContribution(
                          item.contribution
                        )}
                      </span>
                    </div>

                    <div className="contribution-track">
                      <div
                        className="contribution-fill negative-fill"
                        style={{
                          width: `${
                            (Math.abs(item.contribution) /
                              getMaxContribution()) *
                            100
                          }%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="methodology-note">
              These contributions describe how the model
              arrived at its prediction. They do not represent
              causal effects or guarantees about an individual.
            </div>
          </section>
        )}

            {counterfactuals && (
  <section className="panel">
    <div className="section-heading">
      <div>
        <span className="section-number">04</span>
        <h2>What if?</h2>
      </div>

      <p>
        Simulate changes and see how the model's
        prediction responds.
      </p>
    </div>

    <div className="counterfactual-summary">
      <div>
        <span>Current prediction</span>
        <strong>
          {counterfactuals.original_prediction === "Yes"
            ? "ATTRITION"
            : "STAY"}
        </strong>
      </div>

      <div>
        <span>Current probability</span>
        <strong>
          {(
            counterfactuals.original_probability * 100
          ).toFixed(1)}
          %
        </strong>
      </div>
    </div>

    {(() => {
      const simulations = [
        ...(counterfactuals.counterfactuals || []),
      ].sort(
        (a, b) =>
          Math.abs(b.change) - Math.abs(a.change)
      );

      const flips = simulations.filter(
        (item) => item.prediction_changed
      );

      const otherChanges = simulations.filter(
        (item) => !item.prediction_changed
      );

      return (
        <>
          {flips.length > 0 && (
            <div className="counterfactual-group">
              <div className="counterfactual-group-heading">
                <div>
                  <span className="group-kicker">
                    PREDICTION-CHANGING
                  </span>

                  <h3>
                    Changes that flipped the model
                  </h3>
                </div>

                <span className="group-count">
                  {flips.length}
                </span>
              </div>

              <div className="counterfactual-list">
                {flips.map((item, index) => (
                  <div
                    className="counterfactual counterfactual-flip"
                    key={`${item.feature}-${item.to}-flip-${index}`}
                  >
                    <div className="counterfactual-rank">
                      {String(index + 1).padStart(2, "0")}
                    </div>

                    <div className="counterfactual-main">
                      <strong>
                        {FIELD_LABELS[item.feature] ||
                          item.feature}
                      </strong>

                      <div className="change">
                        <span>{String(item.from)}</span>
                        <span>→</span>
                        <span>{String(item.to)}</span>
                      </div>
                    </div>

                    <div className="counterfactual-result">
                      <span>
                        {(
                          item.original_probability * 100
                        ).toFixed(1)}
                        %
                      </span>

                      <span>→</span>

                      <strong>
                        {(
                          item.new_probability * 100
                        ).toFixed(1)}
                        %
                      </strong>
                    </div>

                    <span className="flip-badge">
                      {item.new_prediction === "Yes"
                        ? "→ Attrition"
                        : "→ Stay"}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="counterfactual-group secondary-group">
              <div className="counterfactual-group-heading">
                <div>
                  <span className="group-kicker">
                    PROBABILITY CHANGES
                  </span>

                  <h3>
                    Other simulated outcomes
                  </h3>
                </div>

                <span className="group-count">
                  {otherChanges.length}
                </span>
              </div>

              {otherChanges.length > 0 ? (
                <div className="counterfactual-list">
                  {otherChanges.map((item, index) => (
                    <div
                      className="counterfactual"
                      key={`${item.feature}-${item.to}-other-${index}`}
                    >
                      <div className="counterfactual-rank">
                        {String(index + 1).padStart(2, "0")}
                      </div>

                      <div className="counterfactual-main">
                        <strong>
                          {FIELD_LABELS[item.feature] || item.feature}
                        </strong>

                        <div className="change">
                          <span>{String(item.from)}</span>
                          <span>→</span>
                          <span>{String(item.to)}</span>
                        </div>
                      </div>

                      <div className="counterfactual-result">
                        <span>
                          {(item.original_probability * 100).toFixed(1)}%
                        </span>

                        <span>→</span>

                        <strong>
                          {(item.new_probability * 100).toFixed(1)}%
                        </strong>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="no-counterfactuals">
                  <strong>
                    No additional probability-only changes
                  </strong>

                  <p>
                    Every simulated change in this run was large enough
                    to change the model's final prediction.
                  </p>
                </div>
              )}

              <div className="counterfactual-list">
                {otherChanges.map((item, index) => (
                  <div
                    className="counterfactual"
                    key={`${item.feature}-${item.to}-other-${index}`}
                  >
                    <div className="counterfactual-rank">
                      {String(index + 1).padStart(2, "0")}
                    </div>

                    <div className="counterfactual-main">
                      <strong>
                        {FIELD_LABELS[item.feature] ||
                          item.feature}
                      </strong>

                      <div className="change">
                        <span>{String(item.from)}</span>
                        <span>→</span>
                        <span>{String(item.to)}</span>
                      </div>
                    </div>

                    <div className="counterfactual-result">
                      <span>
                        {(
                          item.original_probability * 100
                        ).toFixed(1)}
                        %
                      </span>

                      <span>→</span>

                      <strong>
                        {(
                          item.new_probability * 100
                        ).toFixed(1)}
                        %
                      </strong>
                    </div>
                  </div>
                ))}
              </div>
            </div>
        </>
      );
    })()}

    <div className="methodology-note">
      Counterfactuals are simulated input changes. They
      show how this model responds to alternative inputs;
      they are not recommendations or causal claims.
    </div>
  </section>
)}

        <footer className="footer">
          <p>
            ModelScope is an educational ML experiment using
            the IBM HR Analytics dataset. Predictions are not
            intended for real employment decisions.
          </p>
        </footer>
      </main>
    </div>
  );
}

export default App;