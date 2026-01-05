# Scikit-learn Integration Guide

This guide shows you how to integrate WhiteBoxAI monitoring with scikit-learn models.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Supported Model Types](#supported-model-types)
3. [Automatic Model Detection](#automatic-model-detection)
4. [Pipeline Integration](#pipeline-integration)
5. [Cross-Validation Monitoring](#cross-validation-monitoring)
6. [GridSearch Monitoring](#gridsearch-monitoring)
7. [Best Practices](#best-practices)
8. [Complete Examples](#complete-examples)

---

## Quick Start

```python
from whiteboxai import WhiteBoxAI
from whiteboxai.integrations.sklearn import SklearnMonitor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Initialize client
client = WhiteBoxAI(api_key='your-api-key')

# Train your model
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
model.fit(X, y)

# Wrap with monitor
monitored_model = SklearnMonitor(
    client=client,
    model=model,
    model_name="iris-classifier",
    auto_register=True
)

# Use as normal - predictions are automatically logged
predictions = monitored_model.predict(X)
```

---

## Supported Model Types

WhiteBoxAI supports all major scikit-learn model types:

### Classification Models
- LogisticRegression
- SVC, LinearSVC
- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier
- AdaBoostClassifier
- KNeighborsClassifier
- GaussianNB, MultinomialNB
- MLPClassifier

### Regression Models
- LinearRegression
- Ridge, Lasso, ElasticNet
- SVR
- DecisionTreeRegressor
- RandomForestRegressor
- GradientBoostingRegressor
- AdaBoostRegressor
- KNeighborsRegressor
- MLPRegressor

### Clustering Models
- KMeans
- DBSCAN
- AgglomerativeClustering
- MeanShift

---

## Automatic Model Detection

WhiteBoxAI automatically detects model types and configuration:

```python
from whiteboxai.integrations.sklearn import SklearnWrapper

# Automatic detection
wrapper = SklearnWrapper(client=client)

# Classifier
classifier = RandomForestClassifier(n_estimators=100, max_depth=10)
classifier.fit(X_train, y_train)
wrapped_classifier = wrapper.wrap(classifier, name="rf-classifier")

# Regressor
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, y_train)
wrapped_regressor = wrapper.wrap(regressor, name="rf-regressor")

# Clustering
clusterer = KMeans(n_clusters=3)
clusterer.fit(X_train)
wrapped_clusterer = wrapper.wrap(clusterer, name="kmeans")
```

### Detected Metadata

The wrapper automatically extracts:
- Model type (classification/regression/clustering)
- Framework version
- Hyperparameters
- Feature names (if using pandas DataFrames)
- Number of features
- Number of classes (for classifiers)

---

## Pipeline Integration

Monitor complete scikit-learn pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=5)),
    ('classifier', RandomForestClassifier())
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Monitor the entire pipeline
monitored_pipeline = SklearnMonitor(
    client=client,
    model=pipeline,
    model_name="iris-pipeline",
    track_preprocessing=True  # Log preprocessing steps
)

# Predictions logged automatically
predictions = monitored_pipeline.predict(X_test)
```

### ColumnTransformer Support

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Complete pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Monitor
monitored = SklearnMonitor(
    client=client,
    model=pipeline,
    model_name="feature-engineering-pipeline",
    log_transformed_features=True  # Log both raw and transformed features
)
```

---

## Cross-Validation Monitoring

Monitor models during cross-validation:

```python
from sklearn.model_selection import cross_val_score, cross_validate
from whiteboxai.integrations.sklearn import monitor_cv

# Simple cross-validation with monitoring
model = RandomForestClassifier()

scores, cv_results = monitor_cv(
    client=client,
    model=model,
    X=X,
    y=y,
    cv=5,
    model_name="rf-cv-experiment",
    scoring=['accuracy', 'precision', 'recall', 'f1'],
    log_fold_results=True  # Log each fold's results
)

print(f"Mean Accuracy: {scores['test_accuracy'].mean():.4f}")
print(f"Std Accuracy: {scores['test_accuracy'].std():.4f}")

# Access detailed CV results
for fold, result in enumerate(cv_results['folds']):
    print(f"Fold {fold}: {result}")
```

### Nested Cross-Validation

```python
from sklearn.model_selection import GridSearchCV

# Inner CV for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}

inner_cv = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=3
)

# Outer CV for model evaluation
outer_scores = monitor_cv(
    client=client,
    model=inner_cv,
    X=X,
    y=y,
    cv=5,
    model_name="nested-cv-experiment",
    log_best_params=True  # Log best params from each outer fold
)
```

---

## GridSearch Monitoring

Track hyperparameter search experiments:

```python
from whiteboxai.integrations.sklearn import MonitoredGridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Monitored GridSearch
grid_search = MonitoredGridSearchCV(
    client=client,
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    model_name="rf-grid-search",
    experiment_name="hyperparameter-tuning-v1",
    log_all_candidates=True,  # Log all parameter combinations
    scoring=['accuracy', 'f1_weighted']
)

# Fit and automatically log all results
grid_search.fit(X_train, y_train)

# Best model is automatically registered
best_model_id = grid_search.best_model_id_

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
print(f"Model ID: {best_model_id}")

# View all trials
trials = grid_search.get_trial_results()
for trial in trials:
    print(f"Params: {trial['params']}, Score: {trial['score']:.4f}")
```

---

## Best Practices

### 1. Feature Names

Always use pandas DataFrames to preserve feature names:

```python
import pandas as pd

# Good: Feature names preserved
X_df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width',
                                  'petal_length', 'petal_width'])
monitored_model.fit(X_df, y)

# Avoid: Generic feature names (f0, f1, f2...)
monitored_model.fit(X_array, y)  # NumPy array
```

### 2. Baseline Data

Set appropriate baseline data for drift detection:

```python
# Use training data as baseline
monitored_model = SklearnMonitor(
    client=client,
    model=model,
    model_name="my-model",
    baseline_data=X_train,
    baseline_predictions=y_train
)

# Or set later
monitored_model.set_baseline(X_train, y_train)
```

### 3. Sampling for High Volume

For high-throughput applications, use sampling:

```python
monitored_model = SklearnMonitor(
    client=client,
    model=model,
    model_name="high-volume-model",
    sampling_rate=0.1,  # Log only 10% of predictions
    batch_size=100,     # Batch predictions for efficiency
    async_logging=True  # Non-blocking logging
)
```

### 4. Explanations

Request explanations selectively:

```python
# Auto-generate explanations for a sample
monitored_model = SklearnMonitor(
    client=client,
    model=model,
    model_name="my-model",
    explain_samples=True,
    explain_sample_rate=0.01,  # Explain 1% of predictions
    explanation_method="shap"   # or "lime"
)

# Or request on-demand
prediction = monitored_model.predict(X_sample)
explanation = monitored_model.explain(X_sample, method="shap")
```

### 5. Model Versioning

Use clear version naming:

```python
# Version in model name
monitored_v1 = SklearnMonitor(
    client=client,
    model=model_v1,
    model_name="fraud-detector",
    version="1.0.0",
    metadata={"training_date": "2024-01-15", "dataset": "v1"}
)

# Later, register new version
monitored_v2 = SklearnMonitor(
    client=client,
    model=model_v2,
    model_name="fraud-detector",
    version="2.0.0",
    metadata={"training_date": "2024-02-15", "dataset": "v2"}
)
```

---

## Complete Examples

### Example 1: Binary Classification

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from whiteboxai import WhiteBoxAI
from whiteboxai.integrations.sklearn import SklearnMonitor
import pandas as pd

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15)
feature_names = [f'feature_{i}' for i in range(20)]
X_df = pd.DataFrame(X, columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2)

# Initialize client
client = WhiteBoxAI(api_key='your-api-key')

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create monitored model
monitored_model = SklearnMonitor(
    client=client,
    model=model,
    model_name="binary-classifier",
    model_type="classification",
    version="1.0.0",
    baseline_data=X_train,
    baseline_predictions=y_train,
    sampling_rate=1.0,  # Log all predictions for testing
    explain_samples=True,
    explain_sample_rate=0.1
)

# Make predictions (automatically logged)
predictions = monitored_model.predict(X_test)
proba = monitored_model.predict_proba(X_test)

# Get feature importance
importance = monitored_model.feature_importance()
print("\nTop Features:")
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {feature}: {score:.4f}")

# Check for drift
drift = monitored_model.check_drift(X_test)
print(f"\nDrift detected: {drift['drift_detected']}")
print(f"Drift score: {drift['drift_score']:.4f}")
```

### Example 2: Multi-Class Classification with Pipeline

```python
from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Load data
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=30)),
    ('svc', SVC(kernel='rbf', probability=True))
])

# Train
pipeline.fit(X_train, y_train)

# Monitor
monitored_pipeline = SklearnMonitor(
    client=client,
    model=pipeline,
    model_name="digits-classifier-pipeline",
    track_preprocessing=True,
    metadata={
        "classes": 10,
        "pca_components": 30,
        "kernel": "rbf"
    }
)

# Predict
predictions = monitored_pipeline.predict(X_test)
print(f"Accuracy: {(predictions == y_test).mean():.4f}")
```

### Example 3: Regression with Feature Engineering

```python
from sklearn.datasets import fetch_california_housing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor

# Load data
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature engineering
preprocessor = ColumnTransformer([
    ('poly', PolynomialFeatures(degree=2), ['MedInc', 'AveRooms']),
    ('std', StandardScaler(), ['HouseAge', 'AveOccup', 'Population'])
], remainder='passthrough')

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100))
])

# Train
pipeline.fit(X_train, y_train)

# Monitor
monitored_regression = SklearnMonitor(
    client=client,
    model=pipeline,
    model_name="housing-price-predictor",
    model_type="regression",
    log_transformed_features=True,
    metadata={
        "target": "median_house_value",
        "polynomial_features": ["MedInc", "AveRooms"]
    }
)

# Predict
predictions = monitored_regression.predict(X_test)
```

---

## Troubleshooting

### Issue: Model not detected correctly

**Solution**: Explicitly specify model type:
```python
monitored_model = SklearnMonitor(
    client=client,
    model=model,
    model_name="my-model",
    model_type="classification"  # or "regression"
)
```

### Issue: Feature names not preserved

**Solution**: Use pandas DataFrame:
```python
X_df = pd.DataFrame(X, columns=feature_names)
monitored_model.predict(X_df)
```

### Issue: Slow prediction logging

**Solution**: Enable async logging and batching:
```python
monitored_model = SklearnMonitor(
    client=client,
    model=model,
    model_name="my-model",
    async_logging=True,
    batch_size=100
)
```

---

## Resources

- [WhiteBoxAI SDK Documentation](https://docs.whiteboxai.com/sdk)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [API Reference](https://docs.whiteboxai.com/api)
- [Support](mailto:whiteboxai-support@kpmg.com)

---

*Last Updated: December 29, 2024*
