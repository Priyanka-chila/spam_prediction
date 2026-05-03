import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# -----------------------
# Load data
# -----------------------
df = pd.read_csv("data/train.csv")

# Drop user_id
df = df.drop(columns=["user_id"])

# Ensure ALL features are numeric
for col in df.columns:
    if col != "label":
        df[col] = pd.to_numeric(df[col], errors="coerce")

X = df.drop("label", axis=1)
y = df["label"]

# -----------------------
# Train-test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------
# Pipeline (NO imblearn)
# -----------------------
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=-1
    ))
])

# -----------------------
# Train
# -----------------------
pipeline.fit(X_train, y_train)

# -----------------------
# Evaluate
# -----------------------
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.5).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# -----------------------
# Save model
# -----------------------
joblib.dump(pipeline, "models/spammer_model.pkl")

print("✅ Model saved successfully")