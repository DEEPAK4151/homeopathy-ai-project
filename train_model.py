import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# ---------- Load data ----------
df = pd.read_csv("homeopathy_remedy_dataset_1000_rows.csv")

df['symptoms'] = df['symptoms'].str.lower()
df['remedy_potency'] = df['remedy'] + " | " + df['potency']

# ---------- Vectorize ----------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['symptoms'])

# ---------- Disease model ----------
y_disease = df['disease']
X_train, X_test, y_train, y_test = train_test_split(
    X, y_disease, test_size=0.2, random_state=42
)

disease_model = RandomForestClassifier(n_estimators=200)
disease_model.fit(X_train, y_train)

# ---------- Remedy model ----------
mlb = MultiLabelBinarizer()
y_remedy = mlb.fit_transform(df['remedy_potency'].apply(lambda x: [x]))

remedy_model = OneVsRestClassifier(
    RandomForestClassifier(n_estimators=200)
)
remedy_model.fit(X, y_remedy)

# ---------- Save ----------
joblib.dump(disease_model, "disease_model.pkl")
joblib.dump(remedy_model, "remedy_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(mlb, "mlb.pkl")

print("✅ Models saved successfully!")