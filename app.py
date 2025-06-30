from datetime import datetime
import io
import zipfile
import streamlit as st
import glob
import pandas as pd
import numpy as np
import subprocess
import tempfile
import os
import json
from src.Missing_Imputation import handle_missing_values_ui
from src.Outliers import detect_outliers_ui
from src.Visualizations import show_visualizations
from src.Data_Insights import show_data_insights
from config.Authenticator import Authenticator
from tab_login import handle_login
from mmlogger import setup_logger
import yaml
import re


with open("config.yaml",'r') as file:
    config_file = yaml.safe_load(file)

def pivot_data(df, date_col, pivot_var, paid_organic_vars, target_var, time_granularity, control_vars):
    df[date_col] = pd.to_datetime(df[date_col])
    if time_granularity == "Weekly":
        df['Time_Period'] = df[date_col] - pd.to_timedelta(df[date_col].dt.weekday, unit='D')
    elif time_granularity == "Monthly":
        df['Time_Period'] = df[date_col].dt.to_period("M").apply(lambda r: r.start_time)
    else:
        df['Time_Period'] = df[date_col].dt.floor('D')
    df['Time_Period'] = pd.to_datetime(df['Time_Period']).dt.date

    numeric_cols = paid_organic_vars + [target_var] + control_vars
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    pivot_grouped = df.groupby(['Time_Period', pivot_var])[paid_organic_vars].sum().reset_index()
    pivoted = pivot_grouped.pivot(index='Time_Period', columns=pivot_var, values=paid_organic_vars)
    pivoted.columns = [f"{metric}_{cat}" for metric, cat in pivoted.columns]
    pivoted = pivoted.reset_index()

    exclude_cols = [date_col, pivot_var] + paid_organic_vars
    control_vars_df = df[['Time_Period'] + control_vars].groupby('Time_Period').sum().reset_index()
    target_var_df = df[['Time_Period', target_var]].groupby('Time_Period').sum().reset_index()

    final_df = pd.merge(pivoted, control_vars_df, on='Time_Period', how='left')
    final_df = pd.merge(final_df, target_var_df, on='Time_Period', how='left')

    return final_df.rename(columns={"Time_Period": "Date"})

def clean_column_names(df):
    def make_valid_name(name):
        name = str(name).strip()
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        name = re.sub(r'_+', '_', name)
        # name = re.sub(r'_+', '_', name)
        if name and name[0].isdigit():
            name = f"var_{name}"
        return name
    df.columns = [make_valid_name(col) for col in df.columns]
    return df


@st.cache_data(show_spinner=False)

def create_zip_buffer(folder_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname)
    buffer.seek(0)
    return buffer

logger = setup_logger('app')
authenticator = Authenticator()

tab1, tab2= st.tabs(["Login","MMM-AUTOMATION"])

try:
    # Open and read the log file
    with open('src/app.log', 'r') as f:
        # Read the log file content
        lines = f.readlines()
        lines_list = [line.strip() for line in lines]  # Strip whitespace and newlines

        # Convert the list of lines into a JSON array
        json_lines_array = json.dumps({"log": lines_list[:100]})

        # Inject JavaScript to print the log in the console
        st.markdown(f"""
        <script type="text/javascript">
            function showLog() {{
                let json_log = JSON.parse({json.dumps(json_lines_array)});
                console.log(json_log);
            }}
        </script>
        """, unsafe_allow_html=True)
except Exception as e:
    try:
        os.write(1, f'{e}.\n'.encode())
    except:
        pass


if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'final_df' not in st.session_state:
    st.session_state.final_df = None
if 'transformed' not in st.session_state:
    st.session_state.transformed = False
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'outliers_treated_df' not in st.session_state:
    st.session_state.outliers_treated_df = None
if 'insights_df' not in st.session_state:
    st.session_state.insights_df = None

with tab1:
        handle_login(authenticator)

with tab2:
    if not st.session_state.get("authenticated", False):
        st.info("🔒 Please login with your Gmail account in the 'Login' tab.")
        logger.info("User attempted to access the app without authentication.")
        st.stop()
    st.markdown("<h1 style='text-align: center;'>📊 MMM Automation Interface</h1>", unsafe_allow_html=True)
    # st.header("Step 1: Upload and Transform Your Data")
    #
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])

    if st.session_state.current_step == 1 and uploaded_file is not None:
        st.header("Step 1: Upload and Transform Your Data")

        # uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)

            st.success("✅ File uploaded successfully!")
            st.subheader("Preview of Uploaded Data")
            st.dataframe(df.head())
            st.subheader("Enter Project Details")

            # 1️⃣ Input for Username
            username = st.text_input("Enter your name")

            # 2️⃣ Auto-filled Project Name (from file name)
            default_project_name = os.path.splitext(uploaded_file.name)[0]
            project_name = st.text_input("Edit project name if needed", value=default_project_name)

            # 3️⃣ Create a DataFrame with these two columns
            if username:
                preview_df = pd.DataFrame({
                    "User Name": [username],
                    "Project Name": [project_name]
                })
                st.session_state["username"] = username
                st.session_state["project_name"] = project_name
                st.subheader("Project Details Preview")
                st.dataframe(preview_df)
            else:
                st.info("Please enter your name to see the project details preview.")

            st.subheader("Step 2: Select Variables")

            datetime_columns = [col for col in df.columns if pd.to_datetime(df[col], errors='coerce').notna().sum() > 0]
            if not datetime_columns:
                st.error("❌ No valid date columns found!")
                st.stop()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                date_col = st.selectbox("Date Column", datetime_columns)
                try:
                    sample_date = pd.to_datetime(df[date_col].dropna().astype(str).iloc[0], errors='coerce')
                    if pd.notna(sample_date):
                        valid_date = True
                    else:
                        st.error("❌ Date column cannot be parsed")
                        valid_date = False
                except Exception:
                    st.error("❌ Unable to parse date column")
                    valid_date = False

            with col2:
                target_var = st.selectbox("Target Variable", df.columns)
                valid_target = pd.api.types.is_numeric_dtype(df[target_var])
                if not valid_target:
                    st.error("❌ Target must be numeric")

            with col3:
                pivot_var = st.selectbox("Pivot Variable", df.columns)
                valid_pivot = df[pivot_var].dtype in ['object', 'category']
                if not valid_pivot:
                    st.error("❌ Pivot variable must be categorical")

            with col4:
                time_base = st.selectbox("Time Granularity", ["Daily", "Weekly", "Monthly"])

            paid_organic_vars = st.multiselect(
                "Paid & Organic Variable(s)",
                [col for col in df.columns if col not in [target_var, pivot_var, date_col]]
            )

            invalid_paid_cols = [col for col in paid_organic_vars if not pd.api.types.is_numeric_dtype(df[col])]
            if invalid_paid_cols:
                st.error(f"❌ Non-numeric variables: {', '.join(invalid_paid_cols)}")
                valid_paid = False
            else:
                valid_paid = True

            control_vars = st.multiselect(
                "Control Variable(s)",
                [col for col in df.columns if col not in [target_var, pivot_var, date_col] + paid_organic_vars]
            )

            invalid_control_cols = [col for col in control_vars if not pd.api.types.is_numeric_dtype(df[col])]
            if invalid_control_cols:
                st.error(f"❌ Non-numeric variables: {', '.join(invalid_control_cols)}")
                valid_control = False
            else:
                valid_control = True

            if valid_date and valid_target and valid_pivot and valid_paid:
                if st.button("🔄 Transform Data"):
                    st.session_state["target_var"] = target_var
                    st.session_state["time_base"] = time_base
                    final_df = pivot_data(df, date_col, pivot_var, paid_organic_vars, target_var, time_base,
                                          control_vars)
                    st.session_state.final_df = final_df
                    st.session_state.transformed = True

                    st.success("✅ Data pivoted successfully!")
                    st.write(f"The dataset contains **{final_df.shape[0]}** rows and **{final_df.shape[1]}** columns.")
                    st.dataframe(final_df.head())
                    row_count = len(final_df)
                    if time_base == "Weekly" and not (104 <= row_count <= 156):
                        st.error(
                            "⚠️ Insufficient Data: Weekly MMM requires 2 to 2.5 years of data, equating to 104–156 weeks.")
                    elif time_base == "Monthly" and not (36 <= row_count <= 48):
                        st.error(
                            "⚠️ Insufficient Data: Monthly MMM requires 3 to 4 years of data (36–48 monthly records).")
                    elif time_base == "Daily" and not (365 <= row_count <= 450):
                        st.error(
                            "⚠️ Insufficient Data: Daily MMM requires 1 to 1.25 years of data (365–450 daily records).")

            if st.session_state.transformed:
                if st.button("➡️ Continue to Missing Value Treatment"):
                    st.session_state.current_step = 2
                    st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    elif st.session_state.current_step == 2 and st.session_state.final_df is not None:
        cleaned = handle_missing_values_ui(st.session_state.final_df)
        if cleaned is not None:
            st.session_state.cleaned_df = cleaned

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Data Transformation"):
                st.session_state.current_step -= 1
                st.rerun()
        with col2:
            if st.session_state.cleaned_df is not None and st.button("➡️ Proceed to Outlier Detection"):
                st.session_state.current_step = 3
                st.rerun()

    elif st.session_state.current_step == 3 and st.session_state.cleaned_df is not None:
        st.session_state.outliers_treated_df = detect_outliers_ui(st.session_state.cleaned_df)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Missing Values"):
                st.session_state.current_step -= 1
                st.rerun()
        with col2:
            if st.session_state.outliers_treated_df is not None and st.button("➡️ Continue to Visualizations"):
                st.session_state.current_step = 4
                st.rerun()

    elif st.session_state.current_step == 4:
        st.write(
            f"The dataset contains **{st.session_state.outliers_treated_df.shape[0]}** rows and **{st.session_state.outliers_treated_df.shape[1]}** columns.")
        st.header("Final Processed Data")
        st.dataframe(st.session_state.outliers_treated_df.head())

        show_visualizations(
            df=st.session_state.outliers_treated_df,
            target_var=st.session_state.get("target_var", None)
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Outlier Treatment"):
                st.session_state.current_step -= 1
                st.rerun()
        with col2:
            if st.button("➡️ Proceed to Data Insights"):
                st.session_state.current_step = 5
                st.rerun()

    elif st.session_state.current_step == 5:
        st.header("Advanced Data Insights")
        st.dataframe(st.session_state.outliers_treated_df.head())

        num_cols = st.session_state.outliers_treated_df.select_dtypes(include=[np.number]).columns.tolist()
        spend_var = st.multiselect("Select spend variables (for ROI analysis)", [None] + num_cols)
        reach_var = st.selectbox("Select reach variable (for time-lag analysis)", [None] + num_cols)
        revenue_var = st.selectbox("Select revenue variable (for ROI analysis)", [None] + num_cols)
        frequency_var = st.selectbox("Select frequency variable (for frequency cap analysis)", [None] + num_cols)

        st.session_state.insights_df = show_data_insights(
            df=st.session_state.outliers_treated_df,
            target_var=st.session_state.get("target_var"),
            time_base=st.session_state.get("time_base"),
            spend_var=spend_var,
            reach_var=reach_var,
            revenue_var=revenue_var,
            frequency_var=frequency_var
        )

        if st.button("⬅️ Back to Visualizations"):
            st.session_state.current_step -= 1
            st.rerun()
        if st.button("➡️ Proceed to Modeling"):
            st.session_state.current_step = 6
            st.rerun()

    elif st.session_state.current_step == 6:
        st.header("📈 Run Marketing Mix Model")

        if st.session_state.insights_df is not None:
            st.subheader("Configure Modeling Parameters")

            if 'rscript_path' not in st.session_state:
                st.session_state.rscript_path = r"C:\Program Files\R\R-4.4.2\bin\Rscript.exe"
            if 'robyn_script_path' not in st.session_state:
                st.session_state.robyn_script_path = config_file["scripts"]["run_robyn"]

            if not os.path.exists(st.session_state.rscript_path):
                st.error(f"Rscript not found at: {st.session_state.rscript_path}")
                st.stop()
            if not os.path.exists(st.session_state.robyn_script_path):
                st.error(f"Robyn script not found at: {st.session_state.robyn_script_path}")
                st.stop()

            with st.expander("Advanced Configuration"):
                st.session_state.rscript_path = st.text_input(
                    "Rscript Path", value=st.session_state.rscript_path, help="Path to Rscript executable"
                )
                st.session_state.robyn_script_path = st.text_input(
                    "Robyn Script Path", value=st.session_state.robyn_script_path, help="Path to run_robyn.R script"
                )

            all_vars = st.session_state.insights_df.select_dtypes(include=[np.number]).columns.tolist()
            date_var = "Date" if "Date" in st.session_state.insights_df.columns else None
            target_var = st.session_state.get("target_var", "Conversions")
            media_vars_options = [v for v in all_vars if v != target_var and v != date_var]

            col1, col2 = st.columns(2)
            with col1:
                media_vars = st.multiselect(
                    "Media Variables (spend/impressions)",
                    media_vars_options,
                    default=media_vars_options[:min(5, len(media_vars_options))],
                    help="Select variables representing media spend or impressions"
                )
                adstock = st.selectbox(
                    "Adstock Transformation",
                    ["geometric", "weibull"],
                    index=0,
                    help="Choose adstock decay type for media effects"
                )
                iterations = st.number_input("Number of Iterations", min_value=100, max_value=2000, value=500, step=100)
                trials = st.number_input("Number of Trials", min_value=1, max_value=5, value=2, step=1)
            with col2:
                control_vars = st.multiselect(
                    "Control Variables (optional)",
                    [v for v in all_vars if v not in media_vars and v != target_var and v != date_var],
                    help="Select variables to control for external factors"
                )

            n_rows = len(st.session_state.insights_df)
            n_ind_vars = len(media_vars) + len(control_vars)
            recommended_rows = n_ind_vars * 10

            st.write(
                f"Dataset: {n_rows} rows, {n_ind_vars} independent variables (Media: {len(media_vars)}, Control: {len(control_vars)})"
            )

            if n_ind_vars > 0 and n_rows < recommended_rows:
                st.error(
                    f"❌ Insufficient data: {n_rows} rows with {n_ind_vars} independent variables. "
                    f"Robyn requires at least {recommended_rows} rows (10:1 ratio)."
                )
                max_vars = n_rows // 10
                if max_vars == 0:
                    st.error("Dataset is too small to model with any variables.")
                    st.button("📂 Re-upload Data", on_click=lambda: st.session_state.update(current_step=1))
                    st.stop()
                reduce_vars = st.checkbox(f"Reduce to {max_vars} variables", value=True)
                if reduce_vars:
                    st.info(f"Limiting to {max_vars} variables.")
                    total_vars = media_vars + control_vars
                    total_vars = total_vars[:max_vars]
                    media_vars = [v for v in total_vars if v in media_vars]
                    control_vars = [v for v in total_vars if v in control_vars]
                    st.write("Selected variables:", media_vars + control_vars)
                else:
                    st.button("📂 Re-upload Data", on_click=lambda: st.session_state.update(current_step=1))
                    st.stop()

            st.markdown("**System Requirements:**\n- Expected runtime: 5–15 minutes")

            if not media_vars:
                st.warning("Please select at least one media variable.")
            elif st.button("🚀 Run Robyn Model"):
                with st.spinner("Running MMM modeling (may take 5–15 minutes)..."):
                    try:
                        output_dir = "robyn_output"
                        os.makedirs(output_dir, exist_ok=True)
                        cleaned_df = clean_column_names(st.session_state.insights_df.copy())
                        old_to_new_names = dict(zip(st.session_state.insights_df.columns, cleaned_df.columns))
                        target_var = old_to_new_names.get(target_var, target_var)
                        media_vars = [old_to_new_names.get(var, var) for var in media_vars]
                        control_vars = [old_to_new_names.get(var, var) for var in control_vars]
                        date_var = old_to_new_names.get(date_var, date_var) if date_var else None

                        config = {
                            "data_path": "",
                            "dep_var": target_var,
                            "media_vars": ",".join(media_vars),
                            "control_vars": ",".join(control_vars) if control_vars else "none",
                            "date_var": date_var if date_var else "none",
                            "adstock_type": adstock,
                            "iterations": str(int(iterations)),
                            "trials": str(int(trials)),
                            "User": st.session_state.get("username", "Unknown"),
                            "Project_Name": st.session_state.get("project_name", "Unknown")
                        }

                        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
                            cleaned_df.to_csv(tmp_file.name, index=False)
                            data_path = tmp_file.name
                            config["data_path"] = data_path

                        # # Save JSON
                        # config_path = os.path.join(output_dir, "config.json")
                        # with open(config_path, "w") as f:
                        #     json.dump(config, f)

                        all_experiments_path = os.path.join(output_dir, "all_experiments.json")
                        if os.path.exists(all_experiments_path):
                            with open(all_experiments_path, "r") as f:
                                experiments = json.load(f)
                        else:
                            experiments = {}

                        # exp_numbers = [int(k.split("_")[1]) for k in experiments.keys() if k.startswith("experiment_")]
                        # next_exp_num = max(exp_numbers) + 1 if exp_numbers else 1
                        # exp_key = f"{next_exp_num}"
                        # # exp_k = f"{next_exp_num}"
                        # config["ID"] = exp_key
                        exp_numbers = [int(k) for k in experiments.keys() if k.isdigit()]
                        next_exp_num = max(exp_numbers) + 1 if exp_numbers else 1
                        exp_key = str(next_exp_num)
                        config["ID"] = exp_key

                        exp = {
                            "User": st.session_state.get("username", "Unknown"),
                            "Project_Name": st.session_state.get("project_name", "Unknown"),
                            "Timestamp": datetime.now().isoformat(),
                            "Spend_variables": ",".join(media_vars),
                            "Control_variables": ",".join(control_vars) if control_vars else "none",
                            "Adstock": adstock,
                            "Iterations": str(int(iterations)),
                            "Trials": str(int(trials))

                        }

                        experiments[exp_key] = exp
                        with open(all_experiments_path, "w") as f:
                            json.dump(experiments, f, indent=4)
                        # experiments[exp_key] = config
                        # with open(config_path, "w") as f:
                        #     json.dump(experiments, f, indent=4)
                        config_path = os.path.join(output_dir, "config.json")
                        with open(config_path, "w") as f:
                            json.dump(config, f)

                        result = subprocess.run(
                            ["Rscript", st.session_state.robyn_script_path],
                            capture_output=True, text=True
                        )


                        plot_path = os.path.join(output_dir, "plots")

                        best_model_path = os.path.join(output_dir, "Best_Models")

                        if result.returncode == 0:
                            st.success("🎉 Modeling completed successfully!")
                            st.session_state.modeling_done = True  # ✅ Set this flag

                            if os.path.exists(best_model_path):
                                png_files = [os.path.join(best_model_path, f) for f in os.listdir(best_model_path) if
                                             f.endswith(".png")]
                                if png_files:
                                    latest_best_model_img = max(png_files, key=os.path.getmtime)
                                    st.subheader("🏆 Best Model")
                                    st.image(latest_best_model_img)
                            if os.path.exists(plot_path):
                                subdirs = [os.path.join(plot_path, d) for d in os.listdir(plot_path) if
                                           os.path.isdir(os.path.join(plot_path, d))]
                                if subdirs:
                                    latest_plot_folder = max(subdirs, key=os.path.getmtime)
                                    st.info("📊 Download all the model output")

                                    # 🔽 Create a ZIP archive of the folder in memory
                                    zip_buffer = io.BytesIO()
                                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                                        for root, _, files in os.walk(latest_plot_folder):
                                            for file in files:
                                                file_path = os.path.join(root, file)
                                                arcname = os.path.relpath(file_path,
                                                                          latest_plot_folder)  # Store relative path
                                                zip_file.write(file_path, arcname)

                                    zip_buffer.seek(0)  # Move to start of the buffer

                                    # 💾 Provide download button
                                    st.download_button(
                                        label="📥 Download Model Plots (ZIP)",
                                        data=zip_buffer,
                                        file_name="model_plots.zip",
                                        mime="application/zip"
                                    )

                            # Show model summary
                        else:

                            st.error("❌ Modeling failed.")

                            # Display actual stderr output from R

                            if result.stderr:
                                st.code(result.stderr, language='bash')  # Styled error output

                            # Add common error-specific messages

                            if "not converged" in result.stderr.lower():

                                st.warning(
                                    "⚠️ Model did not converge. Try increasing iterations or adjusting variables.")

                            elif "coefficient = 0" in result.stderr.lower():

                                st.warning(
                                    "⚠️ Zero coefficients detected. Some predictors might be weak or irrelevant.")

                            elif "argument is of length zero" in result.stderr.lower():

                                st.warning("⚠️ Generic error encountered. Check `robyn_log.txt` for more insights.")


                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")
                    finally:
                        if os.path.exists(data_path):
                            os.unlink(data_path)

            output_dir = "robyn_output"
            os.makedirs(output_dir, exist_ok=True)
            budget_opt_dir = os.path.join(output_dir, "budget_optimization")

            # 📌 Input box to capture the best model ID
            best_model_id = st.text_input(
                "Enter the Best Model ID to Save:",
                placeholder="e.g., 4_190_9",
                key="best_model_input"
            )

            # Step 1: Budget Allocation Type
            option = st.radio("Choose your budget type:", ["Open Allocation", "Fixed Budget"])
            allocation_type = option
            fixed_budget = None

            if option == "Open Allocation":
                st.success("Proceeding with Open Allocation...")

            elif option == "Fixed Budget":
                fixed_budget = st.number_input("Enter your fixed budget value:", min_value=0.0, step=100.0)
                st.success(f"Fixed budget selected: {fixed_budget}")

            constraints = {}
            with st.expander("➕ Optional: Customized Constraints Budget"):
                st.write("Set lower and upper constraints for each media variable:")

                for var in media_vars:  # Assuming media_variables is your list of channel/variable names
                    col1, col2 = st.columns(2)
                    with col1:
                        lower = st.number_input(
                            f"Lower bound for {var}:", min_value=0.0, step=0.1, value=0.7, key=f"lower_{var}"
                        )
                    with col2:
                        upper = st.number_input(
                            f"Upper bound for {var}:", min_value=0.0, step=0.1, value=1.5, key=f"upper_{var}"
                        )

                    if lower > upper:
                        st.error(f"❌ Lower bound cannot be greater than upper for {var}")
                    elif lower > 0.0 or upper > 0.0:
                        constraints[var] = {"lower": lower, "upper": upper}
                        st.success(f"✅ {var} → Constraint Range: {lower} to {upper}")

            # If no constraints were added, set to None
            if not constraints:
                constraints = None

            # Step 3: Optional - Time-based Budget
            time_range_options = ["last_5", "last_10", "last_15", "last_20"]
            selected_value = st.selectbox("Please select the range (optional)", ["None"] + time_range_options)

            # Set time_range to None if "None" is selected
            time_range = None if selected_value == "None" else selected_value

            # Step 4: Save Model Info to JSON
            if best_model_id:
                st.session_state.best_model_id = best_model_id
                model_json_path = os.path.join(output_dir, "model.json")

                model_data = {
                    "model_id": best_model_id,
                    "allocation_type": allocation_type,
                    "fixed_budget": fixed_budget if allocation_type == "Fixed Budget" else None,
                    "constraints": constraints,
                    "time_range": time_range
                }

            # Step 4: Save Model Info to JSON
            if best_model_id:
                st.session_state.best_model_id = best_model_id
                model_json_path = os.path.join(output_dir, "model.json")

                model_data = {
                    "model_id": best_model_id,
                    "allocation_type": allocation_type,
                    "fixed_budget": fixed_budget if allocation_type == "Fixed Budget" else None,
                    "constraints": constraints,
                    "time_range": time_range
                }

                try:
                    with open(model_json_path, "w") as f:
                        json.dump(model_data, f, indent=4)
                    st.success(f"✅ Saved selected model info to {model_json_path}")
                    st.session_state.model_id_saved = True
                except Exception as e:
                    st.error(f"❌ Failed to save model info: {e}")

            # ✅ Show budget optimization button if model ID saved
        if st.session_state.get("model_id_saved"):
            if st.button("🚀 Run Budget Optimization"):
                try:
                    r_script_path = config_file["scripts"]["budgetOptimizerPath"]
                    results = subprocess.run(
                        ["Rscript", r_script_path],
                        capture_output=True,
                        text=True
                    )
                    if results.returncode == 0:
                        st.success("✅ Budget optimization completed successfully.")

                        # Get all subfolders in the budget optimization directory
                        folders = [os.path.join(budget_opt_dir, f) for f in os.listdir(budget_opt_dir) if
                                   os.path.isdir(os.path.join(budget_opt_dir, f))]

                        if not folders:
                            st.warning("⚠️ No folders found in budget optimization directory.")
                        else:
                            # Find the most recently modified folder
                            latest_folder = max(folders, key=os.path.getmtime)

                            st.write("### Displaying budget optimization plots ")

                            # Find all PNG files in the latest folder
                            png_files = sorted(glob.glob(os.path.join(latest_folder, "*.png")))

                            if not png_files:
                                st.warning("⚠️ No PNG files found in the latest budget optimization folder.")
                            else:
                                for img_path in png_files:
                                    st.image(img_path, use_column_width=True)
                    else:
                        st.error("❌ Budget optimization failed.")
                        st.text_area("R Error Output", results.stderr, height=300)
                except Exception as e:
                    st.error(f"❌ Failed to run R script: {e}")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("⬅️ Back to Data Insights"):
                    st.session_state.current_step -= 1
                    st.rerun()
            with col2:
                if st.button("🔄 Restart Modeling"):
                    st.session_state.current_step = 6
                    st.rerun()
            with col3:
                if st.button("📜 History"):
                    st.session_state.current_step = 7
                    st.rerun()



    elif st.session_state.current_step == 7:
        st.header("📜 Modeling History")

        output_dir = os.path.join(os.path.dirname(st.session_state.robyn_script_path), "robyn_output")
        history_path = os.path.join(output_dir, "all_experiments.json")
        plot_path = os.path.join(output_dir, "plots")

        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                experiments = json.load(f)

            if experiments:
                df_history = pd.DataFrame.from_dict(experiments, orient="index").reset_index()

                desired_order = [
                    "index", "User", "Project_Name", "Timestamp", "Spend_variables",
                    "Control_variables", "Adstock", "Iterations", "Trials"
                ]
                df_history = df_history[desired_order]

                st.dataframe(df_history, use_container_width=True)

                # 🔽 Select and download experiment by folder name
                st.subheader("📁 Select Experiment ID to Download")

                # Build folder prefix list and check for matches in plots
                experiment_options = {}
                for i, row in df_history.iterrows():
                    user = row['User']
                    project = row['Project_Name']
                    exp_id = row['index']
                    folder_prefix = f"{user}~{project}~{exp_id}"

                    matching_folders = glob.glob(os.path.join(plot_path, f"{folder_prefix}*"))
                    if matching_folders:
                        experiment_options[exp_id] = matching_folders[0]  # Use experiment ID as dropdown label

                if experiment_options:
                    selected_exp = st.selectbox("Select an Experiment ID", options=list(experiment_options.keys()))
                    if selected_exp:
                        zip_buffer = create_zip_buffer(experiment_options[selected_exp])
                        st.download_button(
                            label=f"📥 Download {selected_exp}.zip",
                            data=zip_buffer,
                            file_name=f"{selected_exp}.zip",
                            mime="application/zip"
                        )
                else:
                    st.info("No matching experiment folders found in the 'plots' directory.")
            else:
                st.info("No experiments found in history.")
        else:
            st.warning("History file (all_experiments.json) not found.")

        # Navigation Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Modeling"):
                st.session_state.current_step = 6
                st.rerun()
        with col2:
            if st.button("🏠 Back to Home"):
                st.session_state.current_step = 1
                st.rerun()






