import os
from flask import Flask, render_template, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np
import shap

# IMPORTANT: Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- APP INITIALIZATION ----------------
app = Flask(__name__)
app.secret_key = "your_secret_key"

# ---------------- DATABASE CONFIG ----------------
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------------- DATABASE MODEL ----------------
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Float)
    tsh = db.Column(db.Float)
    t3 = db.Column(db.Float)
    tt4 = db.Column(db.Float)
    t4u = db.Column(db.Float)
    fti = db.Column(db.Float)
    result = db.Column(db.String(50))
    probability = db.Column(db.Float)

# Create tables
with app.app_context():
    db.create_all()

# ---------------- LOAD MODEL ----------------
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

# ---------------- PREDICTION ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        TSH = float(request.form["TSH"])
        T3 = float(request.form["T3"])
        TT4 = float(request.form["TT4"])
        T4U = float(request.form["T4U"])
        FTI = float(request.form["FTI"])

        features = np.array([[age, TSH, T3, TT4, T4U, FTI]])

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(features)

        values = shap_values.values[0, :, 1]
        feature_names = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]

        explanations = []
        for i in range(len(feature_names)):
            val = float(values[i])
            if val > 0:
                explanations.append(f"{feature_names[i]} increased risk")
            elif val < 0:
                explanations.append(f"{feature_names[i]} reduced risk")

        if not os.path.exists("static"):
            os.makedirs("static")

        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(shap_values[0, :, 1], show=False)
        plt.tight_layout()
        fig.savefig("static/shap.png", bbox_inches="tight")
        plt.close(fig)

        result_text = "Hypothyroid Detected" if prediction == 1 else "No Hypothyroid"

        # SAVE TO DATABASE
        new_prediction = Prediction(
            age=age,
            tsh=TSH,
            t3=T3,
            tt4=TT4,
            t4u=T4U,
            fti=FTI,
            result=result_text,
            probability=round(prob * 100, 2)
        )

        db.session.add(new_prediction)
        db.session.commit()

        return render_template(
            "result.html",
            result=result_text,
            probability=round(prob * 100, 2),
            explanations=explanations
        )

    except Exception as e:
        return render_template(
            "result.html",
            result=f"Error: {str(e)}"
        )

# ---------------- HISTORY ROUTE (CORRECT PLACE) ----------------
@app.route("/history")
def history():
    all_predictions = Prediction.query.order_by(Prediction.id.desc()).all()
    return render_template("history.html", predictions=all_predictions)


# ---------------- TEST ROUTE ----------------
@app.route("/test")
def test():
    return "Test route working"

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
