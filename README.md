# 🧠 Stroke Risk Prediction System
## A Full-Stack Machine Learning Web Application

### 📌 Project Overview
This project is an end-to-end Machine Learning solution designed to estimate the probability of a stroke based on clinical health parameters. By leveraging XGBoost and Streamlit, I transformed a static healthcare dataset into an interactive diagnostic tool.

### 🚀 Live Demo
https://stroke-prediction-model-1.streamlit.app/

### 🛠️ Technical Stack
- Language: Python 3.10+
- Machine Learning: XGBoost, Scikit-Learn
- Data Manipulation: Pandas, NumPy
- Web Framework: Streamlit
- Deployment: GitHub & Streamlit Cloud

### 📊 Key Features & Workflow
1. Data Preprocessing: Handled missing BMI values and performed Label Encoding for categorical variables (Smoking Status, Work Type).
2. Feature Scaling: Utilized StandardScaler to normalize numerical inputs, ensuring the model treats Age and Glucose levels with the correct mathematical weight.
3. Model Architecture: Trained an XGBoost Classifier, optimized for high recall to minimize "False Negatives" in a medical context.
4. Interactive UI: Users can input real-time data to see an instant risk percentage and a color-coded risk assessment.

### 📈 Insights from EDA
During the analysis phase, I discovered:
- Glucose Impact: Patients with average glucose levels above 200 mg/dL showed a significantly higher correlation with stroke events.
- Age Factor: The risk factor accelerates sharply after age 60, particularly when combined with hypertension.

### 🤝 Connect with Me
Kosisochukwu Ukwandu macdonkcee@gmail.com
