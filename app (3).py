import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="EcoRank", layout="wide")

st.title("🌿 EcoRank: Big Data-Powered VIKOR System for Sustainable Stock Decision-Making")
st.markdown("Use the VIKOR method to rank stocks based on multiple financial sustainability indicators.")

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload your CSV file with 1st column = Stock Name, next 10 = Criteria", type="csv")

required_columns = ['EPS', 'DPS', 'NTA', 'DY', 'ROE', 'GPM', 'OPM', 'ROA', 'PE', 'PTBV']
benefit_criteria = ['EPS', 'DPS', 'NTA', 'DY', 'ROE', 'GPM', 'OPM', 'ROA']
cost_criteria = ['PE', 'PTBV']

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if df.shape[1] != 11 or any(col not in df.columns[1:] for col in required_columns):
        st.error("❌ Your file must contain 1 alternative column + these 10 criteria: " + ', '.join(required_columns))
    else:
        st.subheader("📁 Raw Data")
        st.dataframe(df)

        alternatives = df.iloc[:, 0].values
        criteria_data = df[required_columns]

        # Step 1: Normalize
        norm = pd.DataFrame()
        for col in required_columns:
            if col in benefit_criteria:
                norm[col] = (criteria_data[col] - criteria_data[col].min()) / (criteria_data[col].max() - criteria_data[col].min())
            else:
                norm[col] = (criteria_data[col].max() - criteria_data[col]) / (criteria_data[col].max() - criteria_data[col].min())

        st.markdown("### ✅ Step 1: Normalized Matrix")
        st.dataframe(norm)

        # Step 2: Best and Worst values
        f_star = norm.max()
        f_minus = norm.min()

        st.markdown("### ⭐ Step 2: Best (f*) and Worst (f-) Values")
        st.write("**Best (f*)**:", f_star.to_dict())
        st.write("**Worst (f-)**:", f_minus.to_dict())

        # Step 3: Compute S and R
        weights = np.ones(len(required_columns)) / len(required_columns)
        weights_series = pd.Series(weights, index=required_columns)

        S = ((weights_series * (f_star - norm) / (f_star - f_minus + 1e-9)).sum(axis=1))
        R = ((weights_series * (f_star - norm) / (f_star - f_minus + 1e-9)).max(axis=1))

        st.markdown("### 📉 Step 3: Utility (Sᵢ) and Regret (Rᵢ)")
        st.write("S (Group Utility):", S)
        st.write("R (Individual Regret):", R)

        # Step 4: Compute Q
        v = 0.5
        S_star, S_minus = S.min(), S.max()
        R_star, R_minus = R.min(), R.max()

        Q = v * (S - S_star) / (S_minus - S_star + 1e-9) + (1 - v) * (R - R_star) / (R_minus - R_star + 1e-9)

        result_df = pd.DataFrame({
            'Stock': alternatives,
            'S': S,
            'R': R,
            'Q': Q
        }).sort_values(by='Q').reset_index(drop=True)

        st.subheader("🏁 Final VIKOR Ranking")
        st.dataframe(result_df)

        st.success(f"🥇 Top Ranked Stock: **{result_df.iloc[0]['Stock']}**")

        # Step 5: Plot
        st.markdown("### 📊 Q Values Bar Chart")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(result_df['Stock'], result_df['Q'], color='green')
        ax.set_ylabel("Q Value")
        ax.set_title("VIKOR Q Rankings (Lower is Better)")
        ax.set_xticklabels(result_df['Stock'], rotation=45)
        st.pyplot(fig)

else:
    st.info("Upload a CSV file with company names + 10 criteria (EPS, DPS, NTA, DY, ROE, GPM, OPM, ROA, PE, PTBV)")
