import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. Load CSV
# ============================================================

CSV_FILE = "training_data.csv"

df = pd.read_csv(CSV_FILE)

print("Dataset loaded!")
print("Number of samples:", len(df))
print()


# ============================================================
# 2. Select features
# ============================================================

FEATURES = [
    "nose_x",
    "nose_y",

    "left_shoulder_x",
    "left_shoulder_y",

    "right_shoulder_x",
    "right_shoulder_y",

    "left_elbow_angle",
    "right_elbow_angle",

    "left_knee_angle",
    "right_knee_angle",

    "shoulder_angle",
    "torso_angle",

    "normalized_head_distance"
]

X = df[FEATURES]
y = df["posture"]


# ============================================================
# 3. Show dataset information
# ============================================================

print("Posture classes:")
print(y.value_counts())
print()


# ============================================================
# 4. Split training / testing data
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print()


# ============================================================
# 5. Create Random Forest model
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 6. Train model
# ============================================================

print("Training Random Forest...")

model.fit(X_train, y_train)

print("Training completed!")
print()


# ============================================================
# 7. Test model
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("===================================")
print("Model Accuracy:", accuracy)
print("===================================")
print()

print("Classification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# 8. Save trained model
# ============================================================

MODEL_FILE = "posture_model.pkl"

joblib.dump(
    {
        "model": model,
        "features": FEATURES
    },
    MODEL_FILE
)

print()
print("Model saved as:", MODEL_FILE)