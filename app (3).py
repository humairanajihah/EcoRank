import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="EcoRank", layout="wide")
st.title("📈 EcoRank: Big Data-Powered VIKOR System")
st.markdown("A decision support tool for sustainable stock ranking using the VIKOR MCDM method.")

# File upload
uploaded_file = st.file_uploader("📂 Upload your CSV file (First column: Stock Name, Next 10: EPS, DPS, NTA, DY, ROE, GPM, OPM, ROA, PE, PTBV)", type="csv")

# Expected column order
expected_columns = ['EPS', 'DPS', 'NTA', 'DY', 'ROE', 'GPM', 'OPM', 'ROA', 'PE', 'PTBV']
benefit_criteria = ['EPS', 'DPS', 'NTA', 'DY', 'ROE', 'GPM', 'OPM', 'ROA']
cost_criteria = ['PE', 'PTBV']

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    if df.shape[1] != 11:
        st.error("❌ Your file must contain 1 alternative column + 10 criteria columns.")
        st.stop()

    stock_names = df.iloc[:, 0]
    data = df.iloc[:, 1:]
    data.columns = expected_columns

    st.subheader("🧾 Raw Data")
    st.dataframe(df)

    # Step 1: Normalize
    st.subheader("✅ Step 1: Normalize Decision Matrix")
    norm = pd.DataFrame()
    for col in expected_columns:
        if col in benefit_criteria:
            norm[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        elif col in cost_criteria:
            norm[col] = (data[col].max() - data[col]) / (data[col].max() - data[col].min())
    st.dataframe(norm)

    # Step 2: Determine best and worst values
    st.subheader("⭐ Step 2: Best (f*) and Worst (f-) Values")
    f_star = norm.max()
    f_minus = norm.min()
    st.write("Best (f*):", f_star.to_dict())
    st.write("Worst (f-):", f_minus.to_dict())

    # Step 3: Calculate S and R
    st.subheader("📉 Step 3: Group Utility (S) and Regret (R)")
    weights = np.ones(len(expected_columns)) / len(expected_columns)
    weights = pd.Series(weights, index=expected_columns)

    diff = (f_star - norm) / (f_star - f_minus + 1e-9)
    S = (weights * diff).sum(axis=1)
    R = (weights * diff).max(axis=1)
    st.write("S (Group Utility):", S.round(4))
    st.write("R (Individual Regret):", R.round(4))

    # Step 4: Calculate Q
    st.subheader("📊 Step 4: Compute Q Index")
    v = st.slider("V value (Weight of Strategy of Majority)", 0.0, 1.0, 0.5)
    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()

    Q = v * (S - S_star) / (S_minus - S_star + 1e-9) + (1 - v) * (R - R_star) / (R_minus - R_star + 1e-9)

    # Compile results
    result = pd.DataFrame({
        'Stock': stock_names,
        'S': S.round(4),
        'R': R.round(4),
        'Q': Q.round(4)
    }).sort_values(by='Q').reset_index(drop=True)

    st.subheader("🏆 Final VIKOR Ranking")
    st.dataframe(result)

    st.success(f"🎯 Top Ranked Stock: {result.iloc[0]['Stock']}")

    # Step 5: Plot Q values
    st.subheader("📈 Step 5: Visualize Q Rankings")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(result['Stock'], result['Q'], color='teal')
    ax.set_title("Q Values (Lower is Better)")
    ax.set_xlabel("Stock")
    ax.set_ylabel("Q Index")
    ax.tick_params(axis='x', rotation=90)
    st.pyplot(fig)

    # Download result
    st.download_button("📥 Download Ranking CSV", result.to_csv(index=False), "ecorank_results.csv")

else:
    st.info("Upload a properly formatted CSV file to begin.")
