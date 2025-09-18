import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---- Load dataset ----
uploaded_file = st.file_uploader("Upload Housing DB.csv", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Data Loaded Successfully")
    st.dataframe(df.head())
else:
    st.warning("Please upload the Housing DB.csv file to continue.")
    st.stop()
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# ---- Train Random Forest ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_scaled, y)

# ---- Sidebar Inputs for Features ----
st.sidebar.header("Input Features")
user_input = {}
for col in X.columns:
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    mean_val = float(X[col].mean())
    user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)

# ---- Convert to DataFrame and scale ----
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

# ---- Prediction for User Input ----
predicted_price = rf.predict(input_scaled)[0]
st.header("Predicted Median House Price")
st.write(f"${predicted_price*1000:.2f}")

# ---- Predictions for 3 Test Samples ----
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

sample_indices = [0, 1, 2]  # first 3 test samples
sample_preds = rf.predict(X_test_scaled.iloc[sample_indices])
sample_actuals = y_test.iloc[sample_indices]

# ---- Feature Importances ----
feat_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# ---- Side-by-side plots ----
fig, axes = plt.subplots(1, 2, figsize=(14,6))

# Left: Predictions vs Actuals for 3 samples
axes[0].bar(range(len(sample_preds)), sample_actuals, width=0.4, label="Actual", align="edge")
axes[0].bar([i+0.4 for i in range(len(sample_preds))], sample_preds, width=0.4, label="Predicted", align="edge")
axes[0].set_xticks([0.2, 1.2, 2.2])
axes[0].set_xticklabels([f"Sample {i}" for i in sample_indices])
axes[0].set_ylabel("Median Price")
axes[0].set_title("Predicted vs Actual Values (3 Samples)")
axes[0].legend()

# Right: Feature Importances
sns.barplot(x=feat_importances.values, y=feat_importances.index, ax=axes[1], palette="viridis")
axes[1].set_title("Feature Importances from Random Forest")
axes[1].set_xlabel("Importance Score")
axes[1].set_ylabel("Features")

st.pyplot(fig)
