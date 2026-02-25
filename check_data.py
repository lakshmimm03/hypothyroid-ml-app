import pandas as pd

df = pd.read_csv("data/Thyroid-Dataset.csv")

print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget value counts:\n", df["class"].value_counts())
# Convert class to binary (1 = Hypothyroid, 0 = Others)

df["class"] = df["class"].apply(
    lambda x: 1 if "hypothyroid" in x.lower() else 0
)

print("\nUpdated Target Counts:\n", df["class"].value_counts())
# Fill missing numeric values with median
df["TSH"].fillna(df["TSH"].median(), inplace=True)
df["T3"].fillna(df["T3"].median(), inplace=True)
df["TT4"].fillna(df["TT4"].median(), inplace=True)
df["T4U"].fillna(df["T4U"].median(), inplace=True)
df["FTI"].fillna(df["FTI"].median(), inplace=True)

# Fill missing categorical values
df["sex"].fillna("Unknown", inplace=True)

print("\nMissing values after cleaning:\n", df.isnull().sum())

from sklearn.preprocessing import LabelEncoder

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = LabelEncoder().fit_transform(df[col])

print("\nData Types After Encoding:\n", df.dtypes)
from sklearn.model_selection import train_test_split

# ---- SELECT IMPORTANT FEATURES ONLY ----
selected_features = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]

X = df[selected_features]
y = df["class"]
print("Feature shape:", X.shape)
# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)
from sklearn.ensemble import RandomForestClassifier

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"   # VERY IMPORTANT for imbalanced data
)

# Train model
model.fit(X_train, y_train)

print("Model trained successfully!")
import joblib

joblib.dump(model, "model.pkl")
print("Model saved successfully!")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Make predictions
y_pred = model.predict(X_test)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get prediction probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("AUC Score:", auc_score)
import shap

# Create explainer
explainer = shap.TreeExplainer(model)

# Get SHAP explanation object
shap_values = explainer(X_test)

# Plot normal summary (NOT interaction)
shap.summary_plot(shap_values[:, :, 1], X_test)
# ---- CROSS VALIDATION ----
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="f1"
)

print("\nCross Validation F1 Scores:", cv_scores)
print("Average CV F1 Score:", cv_scores.mean())
