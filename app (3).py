import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="EcoRank: VIKOR Stock Ranking", layout="wide")

st.title("🌱 EcoRank: Big Data-Powered VIKOR System for Sustainable Stock Decision-Making")
st.markdown("Upload a CSV file with **1 alternative column** (stock name) and **7 criteria columns** (EPS, DPS, NTA, PE, DY, ROE, PTBV).")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your stock CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Uploaded Data")
        st.dataframe(df)

        expected_cols = ['EPS', 'DPS', 'NTA', 'PE', 'DY', 'ROE', 'PTBV']
        if not all(col in df.columns for col in expected_cols):
            st.error(f"❌ Your file must contain these columns: {expected_cols}")
        else:
            # Extract alternatives and criteria
            alternatives = df.iloc[:, 0].values
            criteria = df[expected_cols]

            # Define benefit and cost criteria
            benefit = ['EPS', 'DPS', 'NTA', 'DY', 'ROE']
            cost = ['PE', 'PTBV']

            # Step 1: Normalize
            norm = pd.DataFrame()
            for col in criteria.columns:
                if col in benefit:
                    norm[col] = (criteria[col] - criteria[col].min()) / (criteria[col].max() - criteria[col].min())
                elif col in cost:
                    norm[col] = (criteria[col].max() - criteria[col]) / (criteria[col].max() - criteria[col].min())

            st.markdown("### Step 1️⃣: Normalized Matrix")
            st.dataframe(norm)

            # Step 2: Best and Worst
            f_star = norm.max()
            f_minus = norm.min()

            st.markdown("### Step 2️⃣: Best and Worst Values")
            st.write("⭐ Best (f*):", f_star)
            st.write("🔻 Worst (f-):", f_minus)

            # Step 3: S and R
            weights = np.ones(len(norm.columns)) / len(norm.columns)
            weighted_diff = weights * (f_star - norm) / (f_star - f_minus + 1e-9)
            S = weighted_diff.sum(axis=1)
            R = weighted_diff.max(axis=1)

            # Step 4: Q calculation
            v = 0.5
            S_star, S_minus = S.min(), S.max()
            R_star, R_minus = R.min(), R.max()
            Q = v * (S - S_star) / (S_minus - S_star + 1e-9) + (1 - v) * (R - R_star) / (R_minus - R_star + 1e-9)

            # Step 5: Results
            result_df = pd.DataFrame({
                'Stock': alternatives,
                'S': S,
                'R': R,
                'Q': Q
            }).sort_values(by='Q').reset_index(drop=True)

            st.markdown("### 🏁 Final VIKOR Ranking")
            st.dataframe(result_df)

            st.success(f"🎯 Top Ranked Stock: {result_df.iloc[0]['Stock']}")

            # Step 6: Bar Chart
            st.markdown("### 📊 Q Value Chart")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(result_df['Stock'], result_df['Q'], color='green')
            ax.set_xlabel("Stock")
            ax.set_ylabel("Q Value")
            ax.set_title("VIKOR Ranking (Lower Q is Better)")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("Upload a CSV file with your stock data to begin.")
