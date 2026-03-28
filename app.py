import streamlit as st
import pandas as pd
import numpy as np
import pickle



# --- STEP 1: LOAD ASSETS SAFELY ---
@st.cache_resource
def load_assets():
    try:
        # We load them one by one to see where it breaks
        model = pickle.load(open("Stroke_prediction_model.pkl", "rb"))
        encoder = pickle.load(open("Stroke_prediction_encoder.pkl", "rb"))
        scaler = pickle.load(open("Stroke_prediction_scaler.pkl", "rb"))
        return model, encoder, scaler
    except FileNotFoundError as e:
        # This prevents the blank screen by showing the error
        st.error(f"Missing File: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Unexpected Error: {e}")
        return None, None, None

def main():
    st.title("Stroke Prediction App")
    
    model, encoder, scaler = load_assets()

    if model is not None:
        st.success("All systems go! Ready for inputs.")
        
        # 1. Collect User Inputs
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", 1, 120, 25)
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        with col2:
            ever_married = st.selectbox("Ever Married", ["Yes", "No"])
            work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            residence = st.selectbox("Residence Type", ["Urban", "Rural"])
            glucose = st.number_input("Avg Glucose Level", 50.0, 300.0, 90.0)
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
            smoking = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

        # 2. Prediction Button
        if st.button("Predict Stroke Risk"):
            try:
                # 1. Manual mapping for categorical columns
                gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
                married_map = {'No': 0, 'Yes': 1}
                residence_map = {'Rural': 0, 'Urban': 1}
                work_map = {'Govt_job': 0, 'Never_worked': 1, 'Private': 2, 'Self-employed': 3, 'children': 4}
                smoking_map = {'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3}

                # 2. Separate numerical features for scaling
                num_df = pd.DataFrame([[age, glucose, bmi]],columns=['age','avg_glucose_level','bmi'])
                scaled_num = scaler.transform(num_df)
                
                # 3. Combine everything in the EXACT order of training X
                # Order: gender, age, hypertension, heart_disease, ever_married, 
                # work_type, Residence_type, avg_glucose_level, bmi, smoking_status
                final_features = [
                    gender_map[gender], 
                    scaled_num[0][0], # scaled age
                    hypertension, 
                    heart_disease, 
                    married_map[ever_married], 
                    work_map[work_type], 
                    residence_map[residence], 
                    scaled_num[0][1], # scaled glucose
                    scaled_num[0][2], # scaled bmi
                    smoking_map[smoking]
                ]

                # 4. Make Prediction
                prob = model.predict_proba([final_features])[0][1]
                
                st.markdown("---")
                if prob > 0.5:
                    st.error(f"High Stroke Risk Detected: {prob:.2%}")
                else:
                    st.success(f"Low Stroke Risk: {prob:.2%}")
                st.balloons()

            except Exception as e:
                st.error(f"Error during prediction: {e}")
if __name__ == "__main__":
    main()