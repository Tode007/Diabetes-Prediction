# Diabetes-Prediction
The project aims to improve early diagnosis, achieving high accuracy (often over 90%) in identifying high-risk individuals using datasets 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 


# Load Dataset (Download diabetes.csv and keep in same folder)
data = pd.read_csv("diabetes.csv")

# Features and Target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Model
model = LogisticRegression(max_iter=1000)

# Train Model
model.fit(X_train, y_train)

# Model Accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# 🔹 User Input for Prediction
print("\nEnter Patient Details:")

preg = float(input("Pregnancies: "))
glucose = float(input("Glucose Level: "))
bp = float(input("Blood Pressure: "))
skin = float(input("Skin Thickness: "))
insulin = float(input("Insulin Level: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

user_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

prediction = model.predict(user_data)

if prediction[0] == 1:
    print("\nPrediction: Diabetic (Sugar Positive) ⚠️")
else:
    print("\nPrediction: Non-Diabetic (Sugar Negative) ✅")
    
    
plt.figure()
plt.scatter(data["Glucose"], data["Outcome"])
plt.xlabel("Glucose Level")
plt.ylabel("Diabetes Outcome (0=No, 1=Yes)")
plt.title("Glucose vs Diabetes\n Created by Dr Omika Bhati.")
plt.show()

