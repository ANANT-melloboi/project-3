import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

# Load Dataset
data = {
    "Brand": ["Philips", "Osram", "Syska", "Havells", "Wipro"],
    "Wattage": [40, 60, 75, 100, 150],
    "Voltage": [220, 220, 110, 110, 220],
    "Material": ["LED", "Halogen", "LED", "Fluorescent", "Tungsten"],
    "Price": [200, 150, 180, 250, 300],
    "Efficiency": ["5 Star", "4 Star", "4 Star", "3 Star", "2 Star"],
    "Failure Rate": [2, 5, 3, 8, 10],
    "Lifespan": [10000, 8000, 9000, 6000, 4000]
}
df = pd.DataFrame(data)

# One-Hot Encoding
df = pd.get_dummies(df, columns=["Brand", "Material", "Efficiency"])

# Regression Model
X = df.drop(columns=["Lifespan"])
y = df["Lifespan"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Classification Model
def classify_lifespan(lifespan):
    if lifespan > 8000:
        return "Best"
    elif lifespan > 5000:
        return "Average"
    else:
        return "Worst"

df["Lifespan Class"] = df["Lifespan"].apply(classify_lifespan)
y_class = df["Lifespan Class"]
X_class = df.drop(columns=["Lifespan", "Lifespan Class"])
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_cls, y_train_cls)

# Streamlit UI
st.title("LUMINA -Best Bulb Predictor")

# User Inputs
wattage = st.selectbox("Select Wattage", [40, 60, 75, 100, 150])
voltage = st.selectbox("Select Voltage", [110, 220])
price = st.slider("Price (₹)", 100, 500, 200)
failure_rate = st.slider("Failure Rate (%)", 0, 20, 5)
brand = st.selectbox("Select Brand", ["Philips", "Osram", "Syska", "Havells", "Wipro"])
material = st.selectbox("Select Material", ["LED", "Halogen", "Fluorescent", "Tungsten"])
efficiency = st.selectbox("Select Efficiency", ["5 Star", "4 Star", "3 Star", "2 Star"])

# Input Dictionary for One-Hot Encoding
input_dict = {
    "Wattage": wattage,
    "Voltage": voltage,
    "Price": price,
    "Failure Rate": failure_rate,
    "Brand_Havells": 0,
    "Brand_Osram": 0,
    "Brand_Philips": 0,
    "Brand_Syska": 0,
    "Brand_Wipro": 0,
    "Material_Fluorescent": 0,
    "Material_Halogen": 0,
    "Material_LED": 0,
    "Material_Tungsten": 0,
    "Efficiency_2 Star": 0,
    "Efficiency_3 Star": 0,
    "Efficiency_4 Star": 0,
    "Efficiency_5 Star": 0
}
input_dict[f"Brand_{brand}"] = 1
input_dict[f"Material_{material}"] = 1
input_dict[f"Efficiency_{efficiency}"] = 1

# DataFrame for Model Input
user_df = pd.DataFrame([input_dict])[X.columns]

# Predict
pred_lifespan = regressor.predict(user_df)[0]
pred_class = classifier.predict(user_df)[0]

# Output Text
st.write(f"Predicted Lifespan: {int(pred_lifespan)} hours")
st.write(f"Suggested Bulb Category: {pred_class}")

# Gauge Chart
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=pred_lifespan,
    title={'text': "Predicted Lifespan (hours)"},
    gauge={
        'axis': {'range': [0, 12000]},
        'bar': {'color': "green"},
        'steps': [
            {'range': [0, 5000], 'color': "red"},
            {'range': [5000, 8000], 'color': "yellow"},
            {'range': [8000, 12000], 'color': "lightgreen"}
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': pred_lifespan
        }
    }
))
st.plotly_chart(fig, use_container_width=True)
