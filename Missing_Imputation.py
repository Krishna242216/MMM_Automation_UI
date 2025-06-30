import io
import streamlit as st
import pandas as pd



def handle_missing_values_ui(final_df):
    st.subheader("🧹 Step 3: Handle Missing Values")
    missing_cols = final_df.columns[final_df.isnull().any()]

    if len(missing_cols) == 0:
        st.success("✅ No missing values found in the dataset!")
        return final_df

    st.write("The following columns have missing values:")
    strategy_dict = {}

    col1, col2, col3 = st.columns([1, 1, 1.5])
    with col1:
        st.write("**Missing Columns**")
    with col2:
        st.write("**Missing Percentage**")
    with col3:
        st.write("**Fill Strategy**")

    for col in missing_cols:
        missing_percent = round(final_df[col].isnull().mean() * 100, 2)

        col1, col2, col3 = st.columns([1, 1, 1.5])
        with col1:
            st.write(col)
        with col2:
            st.write(f"{missing_percent}%")
        with col3:
            selected_strategy = st.selectbox(
                f"Strategy for {col}", ["NONE", "Median", "Mode", "Mean"], key=f"strategy_{col}"
            )
            strategy_dict[col] = selected_strategy

    # Ensure the button logic is using current session_state strategies
    if st.button("🧪 Apply Missing Value Treatments"):
        cleaned_df = final_df.copy()
        for col in missing_cols:
            strategy = st.session_state.get(f"strategy_{col}", "NONE")
            if strategy == "Mean":
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif strategy == "Median":
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            elif strategy == "Mode":
                cleaned_df[col].fillna(cleaned_df[col].mode().iloc[0], inplace=True)
            elif strategy == "NONE":
                cleaned_df[col].fillna(0, inplace=True)

        st.success("✅ Missing values handled successfully!")
        st.dataframe(cleaned_df.head())

        output_cleaned = io.BytesIO()
        with pd.ExcelWriter(output_cleaned, engine='xlsxwriter') as writer:
            cleaned_df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
        output_cleaned.seek(0)

        st.download_button(
            label="📥 Download Cleaned Data",
            data=output_cleaned.getvalue(),
            file_name="MMM_Cleaned_Data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        return cleaned_df

