
import streamlit as st
import pandas as pd
import numpy as np

# Page setup
st.set_page_config(page_title="EcoRank", layout="wide")

st.title("📊 EcoRank")
st.subheader("Big Data-Powered VIKOR System for Sustainable Stock Decision-Making")

REQUIRED_CRITERIA = ['EPS', 'DPS', 'NTA', 'PE', 'DY', 'ROE', 'PTBV']

st.markdown("""
Upload a CSV file where:
- First column = Stock Name (Alternative)
- Next **7 columns** (in order) = EPS, DPS, NTA, PE, DY, ROE, PTBV

All criteria are assumed to be **beneficial**.
""")

# Upload CSV file
uploaded_file = st.file_uploader("📎 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validate columns
    if df.columns[1:].tolist() != REQUIRED_CRITERIA:
        st.error(f"❌ Column names do not match.\nExpected: {REQUIRED_CRITERIA}")
    else:
        st.success("✅ Data loaded successfully!")

        st.write("### 🧾 Raw Data")
        st.dataframe(df)

        # Sidebar: VIKOR parameters
        st.sidebar.title("⚙️ VIKOR Settings")
        v = st.sidebar.slider("V value (balance between group utility & regret)", 0.0, 1.0, 0.5)

        # Criteria weights
        st.sidebar.markdown("### Criteria Weights")
        weights = []
        for crit in REQUIRED_CRITERIA:
            w = st.sidebar.slider(f"Weight for {crit}", 0.0, 1.0, 1.0/len(REQUIRED_CRITERIA), step=0.05)
            weights.append(w)

        weights = np.array(weights)
        weights /= weights.sum()

        # Normalize data (benefit criteria)
        data = df[REQUIRED_CRITERIA].astype(float)
        f_star = data.max()
        f_minus = data.min()
        norm = (f_star - data) / (f_star - f_minus + 1e-9)

        st.write("### 📊 Normalized Data")
        st.dataframe(norm)

        # Calculate S, R
        S = (weights * norm).sum(axis=1)
        R = (weights * norm).max(axis=1)

        S_star, S_minus = S.min(), S.max()
        R_star, R_minus = R.min(), R.max()

        Q = v * (S - S_star) / (S_minus - S_star + 1e-9) + \
            (1 - v) * (R - R_star) / (R_minus - R_star + 1e-9)

        # Compile result
        results = pd.DataFrame({
            'Stock': df.iloc[:, 0],
            'S': S.round(4),
            'R': R.round(4),
            'Q': Q.round(4)
        })

        results = results.sort_values(by='Q').reset_index(drop=True)

        st.write("### 🏆 VIKOR Ranking")
        st.dataframe(results)

        st.download_button("📥 Download Ranking CSV", results.to_csv(index=False), "ecorank_vikor_results.csv")
else:
    st.info("Upload your properly formatted CSV file to begin.")

st.markdown("---")
st.caption("🔬 EcoRank | Developed for INoDEx 2025")
