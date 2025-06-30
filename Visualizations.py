import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def show_visualizations(df, target_var=None):
    """Display various data visualizations"""
    st.header("📈 Trend & Distribution Analysis")

    # 1. Time Series Trend
    with st.expander("Time Series Trend", expanded=True):
        if 'Date' in df.columns:
            date_col = 'Date'
        else:
            date_cols = [col for col in df.columns
                         if pd.api.types.is_datetime64_any_dtype(df[col])]
            date_col = date_cols[0] if date_cols else None

        if date_col:
            if target_var and target_var in df.columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.lineplot(data=df, x=date_col, y=target_var)
                plt.title(f'{target_var} Trend Over Time')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.warning("Could not find target variable for trend analysis")
        else:
            st.warning("No date column found for trend analysis")

    # 2. Distribution Plots
    with st.expander("Distribution Analysis", expanded=True):
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if num_cols:
            selected_num_col = st.selectbox("Select numerical column for distribution", num_cols)
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))

            # Histogram
            sns.histplot(df[selected_num_col], kde=True, ax=ax[0])
            ax[0].set_title(f'Distribution of {selected_num_col}')

            # Boxplot
            sns.boxplot(x=df[selected_num_col], ax=ax[1])
            ax[1].set_title(f'Boxplot of {selected_num_col}')

            st.pyplot(fig)
        else:
            st.warning("No numerical columns found for distribution analysis")

    # 3. Correlation Heatmap
    with st.expander("Correlation Analysis", expanded=True):
        if len(num_cols) > 1:
            # Compute the correlation matrix
            corr_matrix = df[num_cols].corr()

            # Plot the heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            plt.title('Correlation Matrix')
            st.pyplot(fig)

            # Extract upper triangle of the correlation matrix without the diagonal
            mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            upper_corr = corr_matrix.where(mask)

            # Flatten to pairwise format
            corr_pairs = (
                upper_corr.stack()
                .reset_index()
                .rename(columns={0: 'Correlation', 'level_0': 'Feature 1', 'level_1': 'Feature 2'})
            )
            corr_pairs['Abs Correlation'] = corr_pairs['Correlation'].abs()
            # corr_pair['Abs Correlation'] = corr_pairs['Correlation'].abs()

            # Get top 3 most and least correlated pairs
            most_corr = corr_pairs.sort_values(by='Abs Correlation', ascending=False).head(3)
            least_corr = corr_pairs.sort_values(by='Abs Correlation', ascending=True).head(3)

            st.subheader("Most Highly Correlated Feature Pairs")
            st.dataframe(most_corr[['Feature 1', 'Feature 2', 'Correlation']], hide_index=True)

            st.subheader("Least Correlated Feature Pairs")
            st.dataframe(least_corr[['Feature 1', 'Feature 2', 'Correlation']], hide_index=True)
        else:
            st.warning("Need at least 2 numerical columns for correlation analysis")
