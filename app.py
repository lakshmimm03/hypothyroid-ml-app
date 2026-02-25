from flask import redirect, session
from flask import Flask, render_template, request
import joblib
import numpy as np
import shap
import os

# IMPORTANT: Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load model
model = joblib.load("model.pkl")


# ---------------- LOGIN ROUTE ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "doctor" and password == "1234":
            session["user"] = username
            return redirect("/")
        else:
            return "Invalid Credentials"

    return render_template("login.html")


# ---------------- HOME ROUTE ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # --------------------------
        # 1️⃣ Get form input
        # --------------------------
        age = float(request.form["age"])
        TSH = float(request.form["TSH"])
        T3 = float(request.form["T3"])
        TT4 = float(request.form["TT4"])
        T4U = float(request.form["T4U"])
        FTI = float(request.form["FTI"])

        # Create feature array
        features = np.array([[age, TSH, T3, TT4, T4U, FTI]])

        # --------------------------
        # 2️⃣ Model Prediction
        # --------------------------
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        # --------------------------
        # 3️⃣ SHAP Explanation
        # --------------------------
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(features)

        # For binary classification → class index 1
        values = shap_values.values[0, :, 1]

        feature_names = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
        explanations = []

        for i in range(len(feature_names)):
            val = float(values[i])
            if val > 0:
                explanations.append(f"{feature_names[i]} increased risk")
            elif val < 0:
                explanations.append(f"{feature_names[i]} reduced risk")

        # --------------------------
        # 4️⃣ Save SHAP Waterfall Plot
        # --------------------------
        if not os.path.exists("static"):
            os.makedirs("static")

        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(shap_values[0, :, 1], show=False)
        plt.tight_layout()
        fig.savefig("static/shap.png", bbox_inches="tight")
        plt.close(fig)

        # --------------------------
        # 5️⃣ Result Text
        # --------------------------
        result = "Hypothyroid Detected" if prediction == 1 else "No Hypothyroid"

        return render_template(
            "result.html",
            result=result,
            probability=round(prob * 100, 2),
            explanations=explanations
        )

    except Exception as e:
        return render_template(
            "result.html",
            result=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
