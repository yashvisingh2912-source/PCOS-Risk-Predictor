import streamlit as st
import joblib
import pandas as pd
import time
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title='PCOS Risk Predictor',page_icon="🧬" ,layout='wide')

sym_model = joblib.load('sModel.pkl')
cli_model = joblib.load('cModel.pkl')

def show_result_explanation(prob):
    st.markdown("---")
    
    st.subheader("What does this result mean?")

    if prob >= 0.7:
        st.markdown("""
        🔴 **High Risk**
        
        This means your inputs show a strong likelihood of PCOS-related patterns.
        It does not confirm a diagnosis, but medical consultation is strongly recommended.

        **What you should do:**
        - Consult a gynecologist or endocrinologist
        - Consider blood tests and ultrasound
        - Start monitoring lifestyle (diet, exercise, sleep)
        """)
    
    elif prob >= 0.45:
        st.markdown("""
        🟡 **Moderate Risk**
        
        Some indicators suggest possible hormonal imbalance or early PCOS signs.

        **What you should do:**
        - Track menstrual cycle and symptoms
        - Improve diet and physical activity
        - Consider medical advice if symptoms persist
        """)

    else:
        st.markdown("""
        🟢 **Low Risk**
        
        Your inputs currently show low likelihood of PCOS patterns.

        **What you should do:**
        - Maintain a healthy lifestyle
        - Keep monitoring any changes in symptoms
        - Stay consistent with exercise and nutrition
        """)

    st.info("⚠️ This tool is for awareness only and not a medical diagnosis.")

st.title('Know Your PCOS Risk Early')

st.markdown("""
<div style="padding:14px; border-left:4px solid #6c8cff; line-height:1.6;">
<b>🧬 PCOS (Polycystic Ovary Syndrome)</b>

<p>
PCOS is a common but often overlooked hormonal disorder that can affect menstrual cycles, 
cause weight gain, acne, and impact fertility.Many people remain undiagnosed in the early stages, which can lead to long-term health issues 
like diabetes and hormonal imbalance.
</p>

<p>
<b>Early detection and lifestyle changes</b> can significantly improve health and quality of life.
</p>

<p>
Use this tool to check your risk and take a step toward better health.
</p>

<p style="font-size:13px; color:gray; margin-top:8px;">
⚠️ This tool is for awareness purposes only and is not a medical diagnosis. 
Please consult a healthcare professional for proper evaluation.
</p>
</div>
""", unsafe_allow_html=True)

sym,cli = st.tabs(['Symptom Check','Clinical Analysis'])

with sym:
    st.header('Symptom Check')
    st.write("Select the symptoms and lifestyle factors that best match your condition. "
        "This section estimates PCOS risk based on visible signs and daily habits. "
        "It is designed for early awareness, not diagnosis.")

    col1,col2=st.columns(2)
    with col1:
        sym_age = st.number_input('Age', 10, 100, key="sym_age")
        sym_height = st.number_input('Height (cm)', 100, 220, key="sym_height")
        sym_cycleR = st.radio("Cycle Regularity", ["Regular", "Irregular"], key="sym_cycleR")
        sym_cycleL = st.number_input("Period Duration (Days)", 1, 15, key="sym_cycleL")

        sym_weightGain = st.radio("Weight Gain", ["Yes", "No"], key="sym_weightGain")
        sym_hairGrowth = st.radio("Excess Hair Growth", ["Yes", "No"], key="sym_hairGrowth")
        sym_skinDarkening = st.radio("Skin Darkening", ["Yes", "No"], key="sym_skinDarkening")

    with col2:
        sym_weight = st.number_input('Weight (kg)', 20, 150, key="sym_weight")
        sym_hairloss = st.radio("Hair Loss", ["Yes", "No"], key="sym_hairloss")
        sym_pimples = st.radio("Pimples / Acne", ["Yes", "No"], key="sym_pimples")

        sym_fastFood = st.radio("Frequent Fast Food", ["Yes", "No"], key="sym_fastFood")
        sym_exercise = st.radio("Exercise", ["Yes", "No"], key="sym_exercise")

        sym_pregnant = st.radio("Pregnant", ["Yes", "No"], key="sym_pregnant")
        sym_abortions = st.number_input("Abortions", 0, 10, key="sym_abortions")

    submit = st.button("Predict")

    if submit:
        with st.spinner("Analyzing your inputs... Please wait."):
            time.sleep(1.5)
            sym_data = {
                " Age (yrs)": sym_age,
                "Weight (Kg)": sym_weight,
                "Height(Cm) ": sym_height,
                "Cycle(R/I)": 1 if sym_cycleR == "Irregular" else 0,
                "Cycle length(days)": sym_cycleL,
                "Pregnant(Y/N)": 1 if sym_pregnant == "Yes" else 0,
                "No. of abortions": sym_abortions,
                "Weight gain(Y/N)": 1 if sym_weightGain == "Yes" else 0,
                "hair growth(Y/N)": 1 if sym_hairGrowth == "Yes" else 0,
                "Skin darkening (Y/N)": 1 if sym_skinDarkening == "Yes" else 0,
                "Hair loss(Y/N)": 1 if sym_hairloss == "Yes" else 0,
                "Pimples(Y/N)": 1 if sym_pimples == "Yes" else 0,
                "Fast food (Y/N)": 1 if sym_fastFood == "Yes" else 0,
                "Reg.Exercise(Y/N)": 1 if sym_exercise == "Yes" else 0,
                
            }
            input_df = pd.DataFrame([sym_data])
            prob = sym_model.predict_proba(input_df)[0][1]

            if prob >= 0.7:
                st.error(f"⚠️ High Risk ({prob*100:.1f}%) — Please consult a gynecologist")
            elif prob >= 0.45:
                st.warning(f"Moderate Risk ({prob*100:.1f}%) — Monitor your symptoms")
            else:
                st.success(f"Low Risk ({prob*100:.1f}%) — Continue healthy habits")
            show_result_explanation(prob)

            #SHAP
            sym_cols=[' Age (yrs)','Weight (Kg)','Height(Cm) ','Cycle(R/I)','Cycle length(days)',
                    'Pregnant(Y/N)','No. of abortions','Weight gain(Y/N)','hair growth(Y/N)',
                    'Skin darkening (Y/N)','Hair loss(Y/N)','Pimples(Y/N)','Fast food (Y/N)','Reg.Exercise(Y/N)',]
            
            explainer = shap.LinearExplainer(
                sym_model.named_steps['model'],
                shap.maskers.Independent(input_df)
            )
            fig, ax = plt.subplots(figsize=(6, 3.5))  # 👈 smaller size
            
            input_scaled = sym_model.named_steps['scaler'].transform(input_df)
            shap_vals = explainer.shap_values(input_scaled)[0]

            n_features = len(shap_vals)
            fig_height = max(3, n_features * 0.28)

            fig, ax = plt.subplots(figsize=(5, fig_height), dpi=90)

            sorted_idx = sorted(range(n_features), key=lambda i: abs(shap_vals[i]))
            sorted_vals = [shap_vals[i] for i in sorted_idx]
            sorted_features = [sym_cols[i] for i in sorted_idx]

            colors = ['#ef4444' if v > 0 else '#22c55e' for v in sorted_vals]

            ax.barh(sorted_features, sorted_vals, color=colors)
            ax.axvline(0, color='#94a3b8', linewidth=1)

            ax.set_title("🔍 Feature Impact", fontsize=9, color="#e2e8f0", pad=6)
            ax.set_xlabel("Impact", fontsize=8, color="#cbd5f5")

            ax.set_facecolor('#0f172a')
            fig.patch.set_facecolor('#0f172a')

            ax.tick_params(axis='x', colors='#cbd5f5', labelsize=7)
            ax.tick_params(axis='y', colors='#e2e8f0', labelsize=7)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

            ax.spines['left'].set_color('#475569')
            ax.spines['bottom'].set_color('#475569')

            plt.tight_layout(pad=0.5)
        
        with st.expander("🧠 Why this prediction?"):
            st.pyplot(fig, use_container_width=False)

with cli:
    st.header('Clinical Analysis')

    st.write(
        "Enter clinical measurements and test results. "
        "This section provides a more detailed PCOS risk assessment based on medical data."
    )

    col3,col4=st.columns(2)
    
    with col3:
        cli_age = st.number_input('Age (yrs)', 10, 100, key="cli_age")
        cli_weight = st.number_input('Weight (Kg)', 20, 150, key="cli_weight")
        cli_height = st.number_input('Height (Cm)', 100, 220, key="cli_height")
        cli_bmi = st.number_input('BMI', 10.0, 60.0, key="cli_bmi")

        cli_blood = st.selectbox(
            "Blood Group",
            ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
            key="cli_blood"
        )

        cli_pulse = st.number_input('Pulse rate (bpm)', 40, 150, key="cli_pulse")
        cli_rr = st.number_input('RR (breaths/min)', 10, 40, key="cli_rr")
        cli_hb = st.number_input('Hb (g/dl)', 0.0, 20.0, key="cli_hb")

        cli_cycleR = st.radio("Cycle Regularity", ["Regular", "Irregular"], key="cli_cycleR")
        cli_cycleL = st.number_input("Period Duration(days)", 1, 15, key="cli_cycleL")

        cli_marriage = st.number_input("Marriage Duration (years)", 0, 50, key="cli_marriage")
        cli_pregnant = st.radio("Pregnant", ["Yes", "No"], key="cli_pregnant")
        cli_abortions = st.number_input("No. of abortions", 0, 10, key="cli_abortions")

        cli_weightGain = st.radio("Weight Gain", ["Yes", "No"], key="cli_weightGain")
        cli_hairGrowth = st.radio("Hair Growth", ["Yes", "No"], key="cli_hairGrowth")
        cli_skinDarkening = st.radio("Skin Darkening", ["Yes", "No"], key="cli_skinDarkening")
        cli_hairloss = st.radio("Hair Loss", ["Yes", "No"], key="cli_hairloss")
        cli_pimples = st.radio("Pimples", ["Yes", "No"], key="cli_pimples")
        cli_fastFood = st.radio("Fast Food", ["Yes", "No"], key="cli_fastFood")
        cli_exercise = st.radio("Regular Exercise", ["Yes", "No"], key="cli_exercise")

    with col4:
        cli_hcg1 = st.number_input("I beta-HCG (mIU/mL)", 0.0, 100000.0, key="cli_hcg1")
        cli_hcg2 = st.number_input("II beta-HCG (mIU/mL)", 0.0, 100000.0, key="cli_hcg2")

        cli_fsh = st.number_input("FSH (mIU/mL)", 0.0, 50.0, key="cli_fsh")
        cli_lh = st.number_input("LH (mIU/mL)", 0.0, 50.0, key="cli_lh")

        cli_hip = st.number_input("Hip (inch)", 20.0, 80.0, key="cli_hip")
        cli_waist = st.number_input("Waist (inch)", 20.0, 80.0, key="cli_waist")

        cli_tsh = st.number_input("TSH (mIU/L)", 0.0, 10.0, key="cli_tsh")
        cli_amh = st.number_input("AMH (ng/mL)", 0.0, 2000.0, key="cli_amh")
        cli_prl = st.number_input("PRL (ng/mL)", 0.0, 100.0, key="cli_prl")
        cli_vitd = st.number_input("Vit D3 (ng/mL)", 0.0, 100.0, key="cli_vitd")
        cli_prg = st.number_input("PRG (ng/mL)", 0.0, 50.0, key="cli_prg")

        cli_rbs = st.number_input("RBS (mg/dl)", 50.0, 400.0, key="cli_rbs")

        cli_bp_sys = st.number_input("BP Systolic (mmHg)", 80, 200, key="cli_bp_sys")
        cli_bp_dia = st.number_input("BP Diastolic (mmHg)", 50, 130, key="cli_bp_dia")

        cli_fol_L = st.number_input("Follicle No. (L)", 0, 50, key="cli_fol_L")
        cli_fol_R = st.number_input("Follicle No. (R)", 0, 50, key="cli_fol_R")

        cli_fsize_L = st.number_input("Avg. F size (L) (mm)", 0.0, 30.0, key="cli_fsize_L")
        cli_fsize_R = st.number_input("Avg. F size (R) (mm)", 0.0, 30.0, key="cli_fsize_R")

        cli_endo = st.number_input("Endometrium (mm)", 0.0, 20.0, key="cli_endo")

    cli_fsh_lh = cli_fsh / cli_lh if cli_lh != 0 else 0
    cli_waist_hip = cli_waist / cli_hip if cli_hip != 0 else 0

    submit_cli = st.button("Analyze Clinical Data", key="cli_submit")

    if submit_cli:
        with st.spinner("Analyzing your inputs... Please wait."):
            time.sleep(1.5)
            blood_group_map = {
                "A+": 11, "A-": 12, 
                "B+": 13, "B-": 14, 
                "AB+": 15, "AB-": 16, 
                "O+": 17, "O-": 18
            }
            cli_data = {
                ' Age (yrs)': cli_age,
                'Weight (Kg)': cli_weight,
                'Height(Cm) ': cli_height,
                'BMI': cli_bmi,
                'Blood Group': blood_group_map[cli_blood],
                'Pulse rate(bpm) ': cli_pulse,
                'RR (breaths/min)': cli_rr,
                'Hb(g/dl)': cli_hb,
                'Cycle(R/I)': 1 if cli_cycleR == "Irregular" else 0,
                'Cycle length(days)': cli_cycleL,
                'Marraige Status (Yrs)': cli_marriage,
                'Pregnant(Y/N)': 1 if cli_pregnant == "Yes" else 0,
                'No. of abortions': cli_abortions,
                '  I   beta-HCG(mIU/mL)': cli_hcg1,
                'II    beta-HCG(mIU/mL)': cli_hcg2,
                'FSH(mIU/mL)': cli_fsh,
                'LH(mIU/mL)': cli_lh,
                'FSH/LH': cli_fsh_lh,
                'Hip(inch)': cli_hip,
                'Waist(inch)': cli_waist,
                'Waist:Hip Ratio': cli_waist_hip,
                'TSH (mIU/L)': cli_tsh,
                'AMH(ng/mL)': cli_amh,
                'PRL(ng/mL)': cli_prl,
                'Vit D3 (ng/mL)': cli_vitd,
                'PRG(ng/mL)': cli_prg,
                'RBS(mg/dl)': cli_rbs,
                'Weight gain(Y/N)': 1 if cli_weightGain == "Yes" else 0,
                'hair growth(Y/N)': 1 if cli_hairGrowth == "Yes" else 0,
                'Skin darkening (Y/N)': 1 if cli_skinDarkening == "Yes" else 0,
                'Hair loss(Y/N)': 1 if cli_hairloss == "Yes" else 0,
                'Pimples(Y/N)': 1 if cli_pimples == "Yes" else 0,
                'Fast food (Y/N)': 1 if cli_fastFood == "Yes" else 0,
                'Reg.Exercise(Y/N)': 1 if cli_exercise == "Yes" else 0,
                'BP _Systolic (mmHg)': cli_bp_sys,
                'BP _Diastolic (mmHg)': cli_bp_dia,
                'Follicle No. (L)': cli_fol_L,
                'Follicle No. (R)': cli_fol_R,
                'Avg. F size (L) (mm)': cli_fsize_L,
                'Avg. F size (R) (mm)': cli_fsize_R,
                'Endometrium (mm)': cli_endo
            }
        
            input_df = pd.DataFrame([cli_data])
            prob = cli_model.predict_proba(input_df)[0][1]

            if prob >= 0.7:
                st.error(f"⚠️ High Risk ({prob*100:.1f}%) — Please consult a gynecologist")
            elif prob >= 0.45:
                st.warning(f"Moderate Risk ({prob*100:.1f}%) — Monitor your symptoms")
            else:
                st.success(f"Low Risk ({prob*100:.1f}%) — Continue healthy habits")
            show_result_explanation(prob)

            cli_cols = [
                ' Age (yrs)','Weight (Kg)','Height(Cm) ','BMI','Blood Group',
                'Pulse rate(bpm) ','RR (breaths/min)','Hb(g/dl)','Cycle(R/I)',
                'Cycle length(days)','Marraige Status (Yrs)','Pregnant(Y/N)',
                'No. of abortions','  I   beta-HCG(mIU/mL)','II    beta-HCG(mIU/mL)',
                'FSH(mIU/mL)','LH(mIU/mL)','FSH/LH','Hip(inch)','Waist(inch)',
                'Waist:Hip Ratio','TSH (mIU/L)','AMH(ng/mL)','PRL(ng/mL)',
                'Vit D3 (ng/mL)','PRG(ng/mL)','RBS(mg/dl)','Weight gain(Y/N)',
                'hair growth(Y/N)','Skin darkening (Y/N)','Hair loss(Y/N)',
                'Pimples(Y/N)','Fast food (Y/N)','Reg.Exercise(Y/N)',
                'BP _Systolic (mmHg)','BP _Diastolic (mmHg)',
                'Follicle No. (L)','Follicle No. (R)',
                'Avg. F size (L) (mm)','Avg. F size (R) (mm)',
                'Endometrium (mm)'
            ]

            # explainer (same pipeline logic)
            explainer = shap.LinearExplainer(
                cli_model.named_steps['model'],
                shap.maskers.Independent(input_df)
            )

            input_scaled = cli_model.named_steps['scaler'].transform(input_df)

            shap_vals = explainer(input_scaled).values[0]

            n_features = len(shap_vals)
            fig_height = max(4, n_features * 0.22)   # slightly tighter than symptom

            fig, ax = plt.subplots(figsize=(5, fig_height), dpi=90)

            sorted_idx = sorted(range(n_features), key=lambda i: abs(shap_vals[i]))
            sorted_vals = [shap_vals[i] for i in sorted_idx]
            sorted_features = [cli_cols[i] for i in sorted_idx]

            colors = ['#ef4444' if v > 0 else '#22c55e' for v in sorted_vals]

            ax.barh(sorted_features, sorted_vals, color=colors)
            ax.axvline(0, color='#94a3b8', linewidth=1)

            ax.set_title("🧠 Clinical Feature Impact", fontsize=9, color="#e2e8f0", pad=6)
            ax.set_xlabel("Impact on Risk", fontsize=8, color="#cbd5f5")

            ax.set_facecolor('#0f172a')
            fig.patch.set_facecolor('#0f172a')

            ax.tick_params(axis='x', colors='#cbd5f5', labelsize=7)
            ax.tick_params(axis='y', colors='#e2e8f0', labelsize=6)  # 👈 smaller because many features

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

            ax.spines['left'].set_color('#475569')
            ax.spines['bottom'].set_color('#475569')

            plt.tight_layout(pad=0.4)

            with st.expander("🧠 Why this clinical prediction?"):
                st.pyplot(fig, use_container_width=False)

