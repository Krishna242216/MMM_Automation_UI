import streamlit as st
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt


# Define statsmodels availability at module level
STATSMODELS_AVAILABLE = False
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except ImportError:
    st.warning("Statsmodels not available - will skip VIF calculations. Install with: pip install statsmodels")

with open("config.yaml",'r') as file:
    config_file=yaml.safe_load(file)
def handle_zero_missing(df, target_var):
    """Handle zero/missing values in key features with automatic helper selection"""
    with st.expander("🔍 Zero/Missing Value Analysis", expanded=True):
        st.subheader("Missing/Zero Value Report")

        # Make a copy of the dataframe to avoid SettingWithCopyWarning
        df = df.copy()

        # Convert Date column to datetime if it exists
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except Exception as e:
                st.error(f"Failed to convert Date column: {str(e)}")

        # Identify problematic columns (only numeric columns)
        numeric_co = df.select_dtypes(include=[np.number]).columns.tolist()
        missing_or_zero_pct = (df[numeric_co].isnull().sum() + (df[numeric_co] == 0).sum()) / len(df) * 100
        MP = config_file["thresholds"]["Missing_Percentage"]

        high_missing_cols = missing_or_zero_pct[missing_or_zero_pct > MP].index.tolist()
        st.info("High Missing Columns are dropped")
        st.dataframe(high_missing_cols)

        # Drop high-missing columns IN-PLACE
        df.drop(columns=high_missing_cols, inplace=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        zero_or_missing = (df[numeric_cols].isnull().sum() + (df[numeric_cols] == 0).sum()).sort_values(ascending=True)

        problem_cols = zero_or_missing[zero_or_missing > 0].index.tolist()




        if not problem_cols:
            st.success("✅ No missing/zero values found in key features")
            return df

        # Show top 5 most problematic columns
        st.warning(f"⚠️ Top problematic columns:")
        top_5_problems = zero_or_missing
        st.dataframe(top_5_problems.rename("Missing/Zero Count"))

        # Show time periods with issues for top 5 columns
        if 'Date' in df.columns:
            st.write("Sample problematic time periods (showing first 5 occurrences per column):")
            for col in top_5_problems.index.tolist():
                problem_dates = df[df[col].isnull() | (df[col] == 0)]['Date']
                # if not problem_dates.empty:
                    # st.write(f"{col} ({top_5_problems[col]} issues):")
                    # st.dataframe(problem_dates.head(5).to_frame())

        # Treatment options
        st.subheader("Treatment Options")
        option = st.radio("Choose treatment method:", [
            "Continue as is",
            "Auto-impute using correlated variable",
            "Aggregate over time period"
        ])



        # compare with the Percentage comparision
        if option == "Auto-impute using correlated variable":
            imputation_details = st.container()

            # Reset index to ensure safe integer-based access
            df = df.reset_index(drop=True)

            for col in problem_cols:
                if col not in numeric_cols:
                    continue

                corr_matrix = df[numeric_cols].corr()
                potential_helpers = corr_matrix[col].drop(col, errors='ignore')
                CP = config_file["thresholds"]["Correlation_Percentage"]

                strong_helpers = potential_helpers[potential_helpers >= CP].sort_values(ascending=False)

                with imputation_details:
                    st.markdown(f"**Imputation details for '{col}'**")

                    if len(strong_helpers) > 0:
                        helper = strong_helpers.index[0]
                        corr_value = strong_helpers.iloc[0]
                        st.success(f"Found strong correlation: '{helper}' (r = {corr_value:.2f})")

                        # Track last valid value and its helper value
                        last_valid = None
                        last_helper = None

                        # Use .iloc for position-based access
                        for i in range(len(df)):
                            current_target = df[col].iloc[i]
                            current_helper_val = df[helper].iloc[i]

                            if pd.isna(current_target) or (current_target == 0):
                                if last_valid is not None:
                                    if not pd.isna(current_helper_val) and current_helper_val != 0 and last_helper != 0:
                                        pct_change = (current_helper_val - last_helper) / last_helper
                                        imputed_val = last_valid * (1 + pct_change)
                                        df.at[df.index[i], col] = imputed_val  # Use actual index label
                                        st.write(
                                            f"Row {i + 1}: Imputed {imputed_val:.2f} using {helper}'s % change ({pct_change:.2%})"
                                        )

                                        # Update tracking variables
                                        last_valid = imputed_val
                                        last_helper = current_helper_val
                                        continue

                            else:
                                # Update last valid values
                                last_valid = current_target
                                last_helper = current_helper_val

                        # Fallback for remaining zeros/missing
                        remaining = (df[col].isna() | (df[col] == 0)).sum()
                        if remaining > 0:
                            ratio = (df[col] / df[helper]).median()
                            mask = (df[col].isna() | (df[col] == 0)) & (~df[helper].isna())
                            df.loc[mask, col] = df.loc[mask, helper] * ratio
                            st.warning(f"Used median ratio for {remaining} remaining values")

                    else:
                        st.error(f"No strongly correlated variables found for '{col}'")

            st.markdown("---")

            df.to_excel(r"C:\Users\MM3815\Downloads\ALL MMM Auto Outputs\file.xlsx")



        elif option == "Aggregate over time period":

            if 'Date' in df.columns:

                if pd.api.types.is_datetime64_any_dtype(df['Date']):

                    period = st.selectbox("Aggregation period", ["Weekly", "Monthly"])

                    freq_map = {"Weekly": "W", "Monthly": "M"}

                    try:
                        # Set Date as index for resampling

                        temp_df = df.set_index('Date')

                        # Resample and aggregate

                        resampled = temp_df.resample(freq_map[period]).mean()

                        # Reset index to bring Date back as a column

                        df = resampled.reset_index()

                        st.success(f"Successfully aggregated data by {period} periods")

                    except Exception as e:

                        st.error(f"Failed to aggregate data: {str(e)}")

                else:

                    st.error("Date column is not in datetime format - cannot aggregate by time period")

            else:

                st.warning("Date column not found - cannot aggregate by time period")

        return df


def time_lag_analysis(df, target_var, reach_var, time_base):
    with st.expander("📈 Sequential Peak Lag Analysis", expanded=True):
        # Sort data and reset index
        df = df.sort_values('Date').reset_index(drop=True).copy()
        if not reach_var or reach_var not in df.columns:
            st.warning("Reach variable missing")
            return

        RW = config_file["thresholds"]["Rolling_Window"]
        # Find peaks using 7-day rolling window
        df['reach_peak'] = df[reach_var].rolling(RW, center=True).max() == df[reach_var]
        df['conversion_peak'] = df[target_var].rolling(RW, center=True).max() == df[target_var]

        reach_peaks = df[df['reach_peak']].reset_index(drop=True)
        conversion_peaks = df[df['conversion_peak']].reset_index(drop=True)

        # Initialize tracking
        pairs = []
        reach_ptr = 0
        conv_ptr = 0

        while reach_ptr < len(reach_peaks) and conv_ptr < len(conversion_peaks):
            current_reach = reach_peaks.iloc[reach_ptr]

            # Find next conversion after current reach peak
            subsequent_conversions = conversion_peaks[conversion_peaks['Date'] > current_reach['Date']]

            if not subsequent_conversions.empty:
                next_conv = subsequent_conversions.iloc[0]

                # Calculate lag
                lag_days = (next_conv['Date'] - current_reach['Date']).days
                if time_base == "Weekly":
                    lag = lag_days / 7
                elif time_base == "Monthly":
                    lag = lag_days / 30
                else:
                    lag = lag_days

                pairs.append({
                    'Reach Date': current_reach['Date'],
                    'Reach Value': current_reach[reach_var],  # Use actual variable name
                    'Conversion Date': next_conv['Date'],
                    'Conversion Value': next_conv[target_var],  # Use actual variable name
                    'Lag (days)': lag_days,
                    f'Lag ({time_base})': lag
                })

                # Move to next reach peak AFTER this conversion
                reach_ptr = reach_peaks[reach_peaks['Date'] > next_conv['Date']].index.min()
                if pd.isna(reach_ptr):
                    break

                # Move conversion pointer
                conv_ptr = conversion_peaks.index.get_loc(next_conv.name) + 1

            else:
                break

        # Visualization
        if pairs:
            fig, ax1 = plt.subplots(figsize=(14, 7))

            # Plot base lines
            ax1.plot(df['Date'], df[reach_var], 'b-', label=reach_var)
            ax2 = ax1.twinx()
            ax2.plot(df['Date'], df[target_var], 'g-', label=target_var)

            # Plot peaks and connections
            # Plot peaks and connections
            for i, pair in enumerate(pairs):
                # Add label only once for legend
                ax1.scatter(pair['Reach Date'], pair['Reach Value'], color='red', s=100,
                            label='Reach Peak' if i == 0 else "")
                ax2.scatter(pair['Conversion Date'], pair['Conversion Value'], color='purple', s=100,
                            label='Conversion Peak' if i == 0 else "")
                ax1.plot([pair['Reach Date'], pair['Conversion Date']],
                         [pair['Reach Value'], pair['Conversion Value']],
                         'k--', alpha=0.4, label='Lag Line' if i == 0 else "")

                lag_label = f"{pair['Lag (days)']}d" if time_base == "Daily" else f"{pair[f'Lag ({time_base})']:.1f}{time_base[0].lower()}"
                ax1.text(pair['Reach Date'] + (pair['Conversion Date'] - pair['Reach Date']) / 2,
                         (pair['Reach Value'] + pair['Conversion Value']) / 2,
                         lag_label,
                         bbox=dict(facecolor='white', alpha=0.8))

            # Combine legends from both axes
            lines_labels_1 = ax1.get_legend_handles_labels()
            lines_labels_2 = ax2.get_legend_handles_labels()
            lines = lines_labels_1[0] + lines_labels_2[0]
            labels = lines_labels_1[1] + lines_labels_2[1]
            ax1.legend(lines, labels, loc='upper left')

            ax1.set_ylabel(reach_var, color='b')
            ax2.set_ylabel(target_var, color='g')
            ax1.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='g')
            plt.title(f"Sequential {reach_var} → {target_var} Peak Mapping")
            st.pyplot(fig)

            # Display results
            st.subheader("Peak-to-Peak Lag Analysis")
            results = pd.DataFrame(pairs)

            # Format display columns
            display_cols = ['Reach Date', 'Reach Value', 'Conversion Date', 'Conversion Value',
                            f'Lag ({time_base})', 'Lag (days)']

            st.dataframe(results[display_cols].style.format({
                'Reach Date': lambda x: x.strftime('%Y-%m-%d'),
                'Conversion Date': lambda x: x.strftime('%Y-%m-%d'),
                'Reach Value': '{:,.0f}',
                'Conversion Value': '{:,.0f}',
                f'Lag ({time_base})': '{:.1f}',
                'Lag (days)': '{:.0f}'
            }))

            # Calculate statistics
            avg_lag = results[f'Lag ({time_base})'].mean()
            min_lag = results[f'Lag ({time_base})'].min()
            max_lag = results[f'Lag ({time_base})'].max()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Lag", f"{avg_lag:.1f} {time_base.lower()}")
            with col2:
                st.metric("Minimum Lag", f"{min_lag:.1f} {time_base.lower()}")
            with col3:
                st.metric("Maximum Lag", f"{max_lag:.1f} {time_base.lower()}")

            st.write(
                f"**Pattern**: {reach_var} peaks typically lead to {target_var} peaks in {avg_lag:.1f} {time_base.lower()} on average")
        else:
            st.warning("No valid peak pairs found for analysis")


def conversion_peak_analysis(df, target_var, spend_var, revenue_var):
    """Analyze conversion peak duration, ad spend, and ROI"""
    with st.expander("📊 Conversion Peak Analysis", expanded=True):
        # Input validation
        missing_vars = [var for var in spend_var if var not in df.columns]
        if missing_vars:
            st.warning(f"Missing spend variables: {missing_vars}")
            return
        if not revenue_var or revenue_var not in df.columns:
            st.warning("Revenue variable missing")
            return
        if 'Date' not in df.columns:
            st.error("Date column required")
            return

        df['Date'] = pd.to_datetime(df['Date'])

        # Detect peaks (7-day rolling window)
        RW = config_file["thresholds"]["Rolling_Window"]
        df['conversion_peak'] = (
            (df[target_var].rolling(RW, center=True).max() == df[target_var]) &
            (df[target_var] > df[target_var].quantile(0.75))  # Only significant peaks
        )
        peaks = df[df['conversion_peak']].copy()

        if peaks.empty:
            st.warning("No significant conversion peaks detected")
            return

        # Analyze each peak
        results = []
        for _, peak in peaks.iterrows():
            # Dynamic window sizing based on when conversions fall below threshold
            post_peak = df[df['Date'] >= peak['Date']]
            below_threshold = post_peak[post_peak[target_var] < (peak[target_var] * 0.5)]
            if not below_threshold.empty:
                end_date = below_threshold['Date'].iloc[0]
                duration = (end_date - peak['Date']).days
            else:
                duration = (df['Date'].iloc[-1] - peak['Date']).days

            # Calculate metrics for the full spike duration window
            window = df[
                (df['Date'] >= peak['Date']) &
                (df['Date'] <= peak['Date'] + pd.Timedelta(days=duration))
            ]

            total_spend = window[spend_var].sum().sum()
            total_revenue = window[revenue_var].sum()
            roi = (total_revenue - total_spend) / total_spend if total_spend else float('inf')

            results.append({
                'Peak Date': peak['Date'].strftime('%Y-%m-%d'),
                'Peak Value': peak[target_var],
                'Spike Duration (days)': duration,
                'Ad Spend': total_spend,
                'Revenue Generated': total_revenue,
                'ROI': roi,
                'Profit': total_revenue - total_spend
            })

        # Display results
        results_df = pd.DataFrame(results)

        # Summary Metrics
        st.subheader("💰 Performance Summary")
        cols = st.columns(4)
        cols[0].metric("Avg Spike Duration", f"{results_df['Spike Duration (days)'].mean():.1f} days")
        cols[1].metric("Avg ROI", f"{results_df['ROI'].mean():.1%}")
        cols[2].metric("Total Spend", f"${results_df['Ad Spend'].sum():,.0f}")
        cols[3].metric("Total Profit", f"${results_df['Profit'].sum():,.0f}")

        # Detailed Results
        st.subheader("📈 Conversion Peak Analysis")
        st.dataframe(
            results_df.style.format({
                'Ad Spend': '${:,.0f}',
                'Revenue Generated': '${:,.0f}',
                'ROI': '{:.1%}',
                'Profit': '${:,.0f}'
            }),
            height=400
        )

        # Visualizations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Duration vs ROI scatter
        ax1.scatter(
            results_df['Spike Duration (days)'],
            results_df['ROI'],
            c=results_df['Profit'],
            cmap='viridis',
            s=100
        )
        ax1.set_xlabel('Spike Duration (days)')
        ax1.set_ylabel('ROI')
        ax1.set_title('Duration vs ROI (Color = Profit)')
        plt.colorbar(ax1.collections[0], ax=ax1, label='Profit')

        # Timeline plot
        for _, row in results_df.iterrows():
            peak_date = pd.to_datetime(row['Peak Date'])
            ax2.plot(
                [peak_date, peak_date],
                [0, row['Peak Value']],
                'r-',
                alpha=0.3
            )
            ax2.text(
                peak_date,
                row['Peak Value'] * 1.05,
                f"{row['Spike Duration (days)']}d\nROI:{row['ROI']:.0%}",
                ha='center'
            )
        ax2.plot(df['Date'], df[target_var], 'b-', label=target_var)
        ax2.set_title('Conversion Peaks with Duration')
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)



def multicollinearity_check(df, spend_var):
    """Check for multicollinearity issues"""
    with st.expander("🔄 Multicollinearity Check", expanded=False):
        if not STATSMODELS_AVAILABLE:
            st.warning("Cannot check multicollinearity - statsmodels package not installed")
            st.info("Please install statsmodels with: pip install statsmodels")
            return df

        # Select numeric columns from the specified spend variables
        num_cols = df[spend_var].select_dtypes(include=[np.number]).columns.tolist()

        if len(num_cols) < 2:
            st.warning("Need at least 2 numerical variables for analysis")
            return df

        # Calculate VIF
        vif_data = pd.DataFrame()
        vif_data["Variable"] = num_cols
        vif_data["VIF"] = [variance_inflation_factor(df[num_cols].values, i)
                           for i in range(len(num_cols))]

        st.write("Variance Inflation Factors (VIF > 5 indicates multicollinearity):")
        st.dataframe(vif_data.sort_values("VIF", ascending=False))

        problematic = vif_data[vif_data["VIF"] > 5]
        if not problematic.empty:
            st.warning(f"Multicollinearity detected in: {', '.join(problematic['Variable'])}")

            action = st.radio("Multicollinearity treatment:", [
                "Warn but proceed",
                "Automatically drop variables (VIF > 5)",
                "Let me choose which to keep"
            ])

            if action == "Automatically drop variables (VIF > 5)":
                df = df.drop(columns=problematic['Variable'].tolist())
            elif action == "Let me choose which to keep":
                keep = st.multiselect("Select variables to retain", num_cols)
                drop = [v for v in num_cols if v not in keep]
                df = df.drop(columns=drop)

        return df



def frequency_capping_impact(df, reach_var, frequency_var, target_var):
    with st.expander("📊 Impact of Frequency Capping", expanded=True):
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        if not reach_var or reach_var not in df.columns:
            st.warning("Reach variable missing")
            return

        # Determine high reach threshold (80th percentile)
        reach_threshold = df[reach_var].quantile(0.8)
        high_reach_weeks = df[df[reach_var] >= reach_threshold].copy()

        if high_reach_weeks.empty:
            st.warning("No weeks found above 80th percentile of Reach.")
            return

        # Analyze trend in Frequency vs Conversions
        high_reach_weeks['Prev Frequency'] = high_reach_weeks[frequency_var].shift(1)
        high_reach_weeks['Prev Conversions'] = high_reach_weeks[target_var].shift(1)

        high_reach_weeks['Frequency Change'] = high_reach_weeks[frequency_var] - high_reach_weeks['Prev Frequency']
        high_reach_weeks['Conversion Change'] = high_reach_weeks[target_var] - high_reach_weeks['Prev Conversions']

        # Interpretation Column
        def interpret(row):
            if pd.isna(row['Prev Frequency']) or pd.isna(row['Prev Conversions']):
                return "-"
            if row['Frequency Change'] > 0 and row['Conversion Change'] < 0:
                return "Possible Saturation"
            elif row['Frequency Change'] > 0 and row['Conversion Change'] > 0:
                return "Effective Frequency"
            else:
                return "No Clear Impact"

        high_reach_weeks['Impact'] = high_reach_weeks.apply(interpret, axis=1)

        # Scorecards
        total_high_reach_weeks = len(high_reach_weeks)
        saturation_weeks = (high_reach_weeks['Impact'] == "Possible Saturation").sum()
        percent_saturation = (saturation_weeks / total_high_reach_weeks) * 100 if total_high_reach_weeks > 0 else 0
        avg_freq_saturation_weeks = high_reach_weeks.loc[high_reach_weeks['Impact'] == "Possible Saturation", frequency_var].mean()
        median_freq_overall = df[frequency_var].median()

        st.subheader("📊 Summary Metrics")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Saturation Incidents (%)", value=f"{percent_saturation:.1f}%")
        with col2:
            st.metric(label="Avg Frequency (Saturation Weeks)", value=f"{avg_freq_saturation_weeks:.2f}")
        with col3:
            st.metric(label="Median Frequency (Overall)", value=f"{median_freq_overall:.2f}")

        st.caption("If Saturation % is high and Avg Frequency during saturation weeks is much higher than Median, Frequency Capping might be necessary.")

        # Show Detailed Table
        st.subheader("📋 High Reach Weeks Analysis")
        st.dataframe(
            high_reach_weeks[['Date', reach_var, frequency_var, target_var, 'Frequency Change', 'Conversion Change', 'Impact']]
                .style.format({
                    reach_var: '{:,.0f}',
                    frequency_var: '{:,.2f}',
                    target_var: '{:,.0f}',
                    'Frequency Change': '{:+.2f}',
                    'Conversion Change': '{:+.0f}'
                }),
            use_container_width=True
        )

        # Main Graph: Frequency vs Conversions
        st.subheader("📈 Frequency vs Conversions Over Time")
        fig1, ax1 = plt.subplots(figsize=(14, 6))

        color_freq = 'tab:blue'
        color_conv = 'tab:green'

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Frequency', color=color_freq)
        ax1.plot(df['Date'], df[frequency_var], color=color_freq, label='Frequency')
        ax1.tick_params(axis='y', labelcolor=color_freq)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Conversions', color=color_conv)
        ax2.plot(df['Date'], df[target_var], color=color_conv, label='Conversions')
        ax2.tick_params(axis='y', labelcolor=color_conv)

        fig1.suptitle('Frequency vs Conversions Over Time', fontsize=16)
        fig1.tight_layout()
        st.pyplot(fig1)

        # Second Graph: Reach vs Frequency
        st.subheader("📈 Reach vs Frequency Over Time")
        fig2, ax3 = plt.subplots(figsize=(14, 6))

        color_reach = 'tab:purple'
        color_freq2 = 'tab:blue'

        ax3.set_xlabel('Date')
        ax3.set_ylabel('Reach', color=color_reach)
        ax3.plot(df['Date'], df[reach_var], color=color_reach, label='Reach')
        ax3.tick_params(axis='y', labelcolor=color_reach)

        ax4 = ax3.twinx()
        ax4.set_ylabel('Frequency', color=color_freq2)
        ax4.plot(df['Date'], df[frequency_var], color=color_freq2, label='Frequency')
        ax4.tick_params(axis='y', labelcolor=color_freq2)

        fig2.suptitle('Reach vs Frequency Over Time', fontsize=16)
        fig2.tight_layout()
        st.pyplot(fig2)

        # st.success("Impact of Frequency Capping Analysis Complete ✅")





def show_data_insights(df, target_var, time_base, spend_var, reach_var=None, revenue_var=None, frequency_var=None):
    """Main function to display all data insights"""
    st.header("🔬 Advanced Data Insights")

    # print("ffsdfsffaf:",time_base)
    # print("ffsdfsffaf:",target_var)

    # 1. Zero/missing handling
    df = handle_zero_missing(df, target_var)

    # 2. Time lag analysis
    time_lag_analysis(df, target_var, reach_var, time_base )

    # 3. Conversion peak analysis
    conversion_peak_analysis(df, target_var, spend_var, revenue_var)

    # 4. Multicollinearity check
    df = multicollinearity_check(df, spend_var)

    frequency_capping_impact(df, reach_var, frequency_var, target_var)

    return df