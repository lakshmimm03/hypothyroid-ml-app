from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from flask import send_file
import io
from werkzeug.security import generate_password_hash, check_password_hash
import os
import csv
from flask import Flask, render_template, request, redirect, session, Response
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np

# ---------------- APP INITIALIZATION ----------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")

# ---------------- DATABASE CONFIG ----------------
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------------- DATABASE MODELS ----------------
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
with app.app_context():
    db.create_all()

class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
with app.app_context():
    db.create_all()

# ✅ CREATE TABLES (CORRECT PLACE)
with app.app_context():
    db.create_all()


# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")


# ---------------- REGISTER ROUTE ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        new_doctor = Doctor(username=username, password=password)
        db.session.add(new_doctor)
        db.session.commit()

        return redirect("/login")

    return render_template("register.html")


# ---------------- LOGIN ROUTE ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        doctor = Doctor.query.filter_by(username=username).first()

        if doctor and check_password_hash(doctor.password, password):
            session["user"] = doctor.id
            return redirect("/")

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
            probability=float(round(prob * 100, 2))
        )

        db.session.add(new_prediction)
        db.session.commit()

        return render_template(
            "result.html",
            result=result_text,
            probability=float(round(prob * 100, 2))
        )

    except Exception as e:
        return render_template("result.html", result=f"Error: {str(e)}")


# ---------------- HISTORY ROUTE ----------------
@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/login")

    page = request.args.get('page', 1, type=int)
    per_page = 5

    pagination = Prediction.query.order_by(
        Prediction.id.desc()
    ).paginate(page=page, per_page=per_page)

    total = Prediction.query.count()
    positive = Prediction.query.filter(
        Prediction.result.contains("Detected")
    ).count()
    negative = total - positive

    return render_template(
        "history.html",
        predictions=pagination.items,
        pagination=pagination,
        total=total,
        positive=positive,
        negative=negative
    )


# ---------------- EXPORT CSV ROUTE ----------------
@app.route("/export")
def export():
    if "user" not in session:
        return redirect("/login")

    predictions = Prediction.query.all()

    def generate():
        yield "ID,Age,TSH,T3,TT4,T4U,FTI,Result,Probability\n"
        for p in predictions:
            yield f"{p.id},{p.age},{p.tsh},{p.t3},{p.tt4},{p.t4u},{p.fti},{p.result},{p.probability}\n"

    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=predictions.csv"}
    )
@app.route("/download/<int:id>")
def download_pdf(id):
    if "user" not in session:
        return redirect("/login")

    prediction = Prediction.query.get_or_404(id)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("ThyroPredict Medical Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    data = [
        ["Age", prediction.age],
        ["TSH", prediction.tsh],
        ["T3", prediction.t3],
        ["TT4", prediction.tt4],
        ["T4U", prediction.t4u],
        ["FTI", prediction.fti],
        ["Result", prediction.result],
        ["Probability", f"{prediction.probability}%"]
    ]

    table = Table(data)
    elements.append(table)

    doc.build(elements)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"thyropredict_report_{id}.pdf",
        mimetype="application/pdf"
    )

# ---------------- TEST ROUTE ----------------
@app.route("/test")
def test():
    return "Test route working"


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
