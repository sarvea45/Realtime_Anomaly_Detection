import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------
# 📥 Load and Prepare the Dataset
# ---------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("cleaned_dataset.csv")  # file already preprocessed
    return data

data = load_data()

# ✅ Fix: Drop or fill NaN values
if data.isnull().sum().sum() > 0:
    st.warning("⚠️ Dataset contains missing values. Filling with 0.")
    data = data.fillna(0)

# Convert attack_type to string
data['attack_type'] = data['attack_type'].astype(str)

# Select features for model training
selected_features = [
    'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent',
    'count', 'srv_count', 'serror_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate'
]

X = data[selected_features]
y = data['attack_type']  # Multi-class classification

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------
# 🎯 Train and Evaluate the Model
# ---------------------------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate performance
y_pred = model.predict(X_test)
st.write("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
st.write("📊 Classification Report:")
st.text(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'nsl_model.pkl')

# ---------------------------------------------
# 🔍 Real-Time Attack Type Prediction UI
# ---------------------------------------------
# Load the saved model
model = joblib.load('nsl_model.pkl')

st.title("🚨 Real-Time Network Attack Detection")
st.subheader("🔧 Enter Network Traffic Features to Predict the Type of Attack")

# Define input features
input_features = selected_features

# Sidebar for user input
st.sidebar.header("🧾 Input Network Traffic Features")
user_input = []

for feature in input_features:
    min_val = float(data[feature].min())
    max_val = float(data[feature].max())
    mean_val = float(data[feature].mean())

    if data[feature].dtype == 'float64':
        value = st.sidebar.slider(f"{feature}", min_val, max_val, mean_val, step=0.01)
    else:
        value = st.sidebar.number_input(f"{feature}", int(min_val), int(max_val), int(mean_val))

    user_input.append(value)

# ✅ Convert user input to DataFrame (fixes warning)
input_df = pd.DataFrame([user_input], columns=input_features)

# Predict on user input
if st.button("🔍 Predict Attack Type"):
    try:
        prediction = model.predict(input_df)
        st.success(f"🚨 Predicted Attack Type: **{prediction[0]}**")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
