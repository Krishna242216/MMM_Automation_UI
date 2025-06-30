import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import io


def detect_outliers_ui(df):
    st.title("📊 MMM - Step 3: Outlier Detection")

    # Access the target variable from session state
    target_var = st.session_state.get("target_var")

    if not target_var or target_var not in df.columns:
        st.error("❌ Target variable not found in the cleaned dataset.")
        return df

    st.subheader(f"Outlier Detection for Target Variable: `{target_var}`")

    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.5)

    with col2:
        treatment = st.selectbox("Outlier Treatment", ["None", "Cap at Threshold", "Remove Outliers"])

    # Detect outliers using Z-score
    z_scores = np.abs(stats.zscore(df[target_var].dropna()))
    outliers = z_scores > threshold
    n_outliers = sum(outliers)

    # Get min/max values
    min_val = df[target_var].min()
    max_val = df[target_var].max()

    treated_df = df.copy()

    # Apply treatment
    if treatment == "Cap at Threshold" and n_outliers > 0:
        mean = df[target_var].mean()
        std = df[target_var].std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        treated_df[target_var] = df[target_var].clip(lower_bound, upper_bound)

    elif treatment == "Remove Outliers" and n_outliers > 0:
        treated_df = df.loc[~outliers]

    # Show report
    outlier_summary = pd.DataFrame([{
        "Variable": target_var,
        "Min Value": f"{min_val:.2f}",
        "Max Value": f"{max_val:.2f}",
        "Outliers Count": n_outliers,
        "% Outliers": f"{(n_outliers / len(df)) * 100:.1f}%"
    }])
    st.dataframe(outlier_summary)

    # Show treatment preview
    if treatment != "None":
        st.subheader("Treatment Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Before Treatment**")
            st.write(df[[target_var]].describe())
        with col2:
            st.write("**After Treatment**")
            st.write(treated_df[[target_var]].describe())

    # Download option
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        treated_df.to_excel(writer, index=False, sheet_name='Outliers_Treated')
    st.download_button(
        label="📥 Download Outlier-Treated Data",
        data=output.getvalue(),
        file_name="MMM_Outliers_Treated.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    return treated_df
