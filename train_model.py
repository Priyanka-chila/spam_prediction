import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Dataset
# -----------------------------
DATA_PATH = "data/train.csv"

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# -----------------------------
# 2. Basic Data Checks
# -----------------------------
print("\nTarget distribution:")
print(df['label'].value_counts())

# Drop user_id
df.drop(columns=['user_id'], inplace=True)

# Separate features and target
X = df.drop(columns=['label'])
y = df['label']

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------
# 4. Handle Class Imbalance (SMOTE)
# -----------------------------
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# -----------------------------
# 5. Build Training Pipeline
# -----------------------------
pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    ))
])

# -----------------------------
# 6. Train Model
# -----------------------------
pipeline.fit(X_train_resampled, y_train_resampled)

print("\nModel training completed.")

# -----------------------------
# 7. Model Evaluation
# -----------------------------
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC-AUC Score:", roc_auc)

# -----------------------------
# 8. Confusion Matrix Visualization
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# -----------------------------
# 9. Save Model
# -----------------------------
MODEL_PATH = "models/spammer_model.pkl"
joblib.dump(pipeline, MODEL_PATH)

print(f"\nModel saved at: {MODEL_PATH}")