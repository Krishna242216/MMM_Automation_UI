# 📈 Marketing Mix Modeling (MMM) Automation Platform

An end-to-end MMM solution using **Meta’s Robyn (R)** integrated with a **Streamlit app** for data preprocessing, modeling, diagnostics, and campaign optimization. Includes secure access via **Google Authenticator** and full support for data validation, insights, and model interpretation — all through an interactive UI.

---

## 🔧 Key Features

- 🧠 **Automated Robyn MMM Modeling**  
  - Supports hyperparameter tuning, calibration, and budget allocation
  - Integrates directly with Robyn R scripts triggered via Python

- 🖥️ **Streamlit Frontend with Google Authenticator**  
  - Secure login and access control
  - Upload dataset and configure model settings
  - View plots, diagnostics, and budget outputs

- 🔍 **Advanced Data Preprocessing in UI**  
  - **Missing Value Handling**: Visual and automatic imputation
  - **Outlier Detection**: Visual flagging, filtering, and adjustment
  - **Pivot Transformation**: Weekly, monthly, or daily grouping
  - **Insight Generation**: Summary stats and visual exploration

- 📊 **Interactive Visualizations & Diagnostics**  
  - Media spend vs conversions
  - Trend lines, outlier patterns, and variable importance
  - Robyn-generated plots and budget recommendations

---

## 📁 Folder Structure

```bash
├── app/
│   ├── MMM_Streamlit_App.py           # Main Streamlit app (UI + logic)
│   ├── auth/
│   │   └── Google_Auth.py             # Google Authenticator integration
│   ├── utils/
│   │   ├── Missing_Imputation.py      # Handles missing values
│   │   ├── Outlier_Handler.py         # Detects/removes outliers
│   │   ├── Data_Insights.py           # Data summary/EDA
│   │   └── App.py          # Pivoting and formatting
│   │   └── Visualizations.py          # Analyze the trends
├── robyn/
│   └── run_robyn.R                    # Robyn modeling script (invoked by Python)
├── data/
│   └── sample_input.csv               # Sample input format
├── output/
│   ├── plots/                         # Robyn model plots
│   ├── model_metrics.json             # JSON summary of model
│   └── budget_allocation.csv          # Recommended spend allocation
├── README.md
