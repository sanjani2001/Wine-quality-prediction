import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Load your dataset and model
@st.cache_data
def load_data():
    return pd.read_csv("data/WineQT.csv")  # Ensure dataset file here

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")  # Ensure model.pkl is in root folder

df = load_data()
model = load_model()

# Features list (must match your training)
FEATURE_NAMES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                 "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                 "density", "pH", "sulphates", "alcohol"]

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Visualizations", "Prediction", "Model Performance"])

# --- Home ---
if page == "Home":
    st.title("🍷 Wine Quality Prediction App")
    st.write("""
    This app predicts whether a wine is **Good** or **Bad** based on its physicochemical properties.
    It was built using Python, Streamlit, and a trained Random Forest model.
    """)

# --- Data Exploration ---
elif page == "Data Exploration":
    st.title("📊 Data Exploration")
    st.write(f"Dataset shape: {df.shape}")
    st.write("Column data types:")
    st.write(df.dtypes)

    st.subheader("Sample Data")
    st.dataframe(df.head())

    st.subheader("Filter Data")
    feature = st.selectbox("Select feature to filter by", df.columns)
    min_val, max_val = st.slider("Select value range",
                                 float(df[feature].min()),
                                 float(df[feature].max()),
                                 (float(df[feature].min()), float(df[feature].max())))
    filtered = df[(df[feature] >= min_val) & (df[feature] <= max_val)]
    st.dataframe(filtered)

# --- Visualizations ---
elif page == "Visualizations":
    st.title("📈 Visualizations")

    st.subheader("Histogram of Feature")
    hist_feature = st.selectbox("Choose feature for histogram", df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[hist_feature], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Boxplot: Feature vs Quality")
    box_feature = st.selectbox("Select feature for boxplot", df.columns[:-1])
    fig, ax = plt.subplots()
    sns.boxplot(x="quality", y=box_feature, data=df, ax=ax)
    st.pyplot(fig)

# --- Prediction ---
elif page == "Prediction":
    st.title("🤖 Predict Wine Quality")

    # Input fields
    input_vals = []
    for feature in FEATURE_NAMES:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        step = 0.01 if df[feature].dtype != int else 1
        val = st.number_input(feature.capitalize(), min_value=min_val, max_value=max_val, step=step)
        input_vals.append(val)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_vals], columns=FEATURE_NAMES)
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][pred]

        if pred == 1:
            st.success(f"🍷 This wine is predicted to be **Good** ✅ (Confidence: {proba:.2%})")
        else:
            st.error(f"🍷 This wine is predicted to be **Bad** ❌ (Confidence: {proba:.2%})")

# --- Model Performance ---
elif page == "Model Performance":
    st.title("📌 Model Performance")

    # Convert quality to binary: Good if quality >= 6, else Bad
    X = df[FEATURE_NAMES]
    y = (df["quality"] >= 6).astype(int)

    y_pred = model.predict(X)

    st.subheader("Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)




# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np

# # Load the model
# model = joblib.load("model.pkl")

# st.title("🍷 Wine Quality Prediction App")
# st.write("Enter the wine characteristics below to predict if it’s good or bad.")

# # Sidebar input fields
# fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, step=0.1)
# volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, step=0.01)
# citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, step=0.01)
# residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=20.0, step=0.1)
# chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, step=0.001)
# free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, step=1.0)
# total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, step=1.0)
# density = st.number_input("Density", min_value=0.0, max_value=2.0, step=0.0001)
# pH = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.01)
# sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, step=0.01)
# alcohol = st.number_input("Alcohol", min_value=0.0, max_value=20.0, step=0.1)

# # Predict button
# if st.button("Predict Quality"):
#     input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
#                             chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
#                             density, pH, sulphates, alcohol]])
    
#     prediction = model.predict(input_data)[0]
    
#     if prediction == 1:
#         st.success("This wine is predicted to be **Good** 🍷✅")
#     else:
#         st.error("This wine is predicted to be **Bad** 🍷❌")
