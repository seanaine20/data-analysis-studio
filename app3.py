"""
Advanced Data Analysis Studio
=============================
A professional-grade Streamlit application for exploratory data analysis,
statistical testing, regression modeling, and interactive visualization.

Run with:
    streamlit run advanced_data_analysis_app.py

Required packages:
    pip install streamlit pandas numpy scipy statsmodels plotly openpyxl scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced Data Analysis Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Overall page */
    .stApp {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef2f7 100%);
    }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1f4e79 0%, #2e75b6 60%, #4aa3df 100%);
        padding: 28px 36px;
        border-radius: 16px;
        color: white;
        box-shadow: 0 8px 24px rgba(31, 78, 121, 0.25);
        margin-bottom: 6px;
    }
    .header-banner h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .header-banner p {
        font-size: 1.05rem;
        margin: 6px 0 0 0;
        opacity: 0.92;
    }
    .header-badges span {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 20px;
        padding: 4px 14px;
        margin: 10px 6px 0 0;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e3e8ef;
        padding: 16px 12px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #5a6b7d;
    }
    [data-testid="stMetricValue"] {
        color: #1f4e79;
        font-weight: 800;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f4e79 0%, #173a5e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #f0f4f8 !important;
    }
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #d8e3ee !important;
    }

    /* Section headers */
    h2, h3 {
        color: #1f4e79 !important;
        font-weight: 700;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    /* Buttons */
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, #2e75b6, #1f4e79);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5em 1.2em;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(31, 78, 121, 0.35);
    }

    /* Footer */
    .footer-box {
        text-align: center;
        padding: 18px;
        margin-top: 24px;
        border-radius: 12px;
        background: linear-gradient(135deg, #1f4e79, #2e75b6);
        color: #eaf2fb;
        font-size: 0.85rem;
        box-shadow: 0 4px 14px rgba(31,78,121,0.25);
    }
    .footer-box b { color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)

COLOR_THEMES = {
    "Corporate Blue": {"seq": px.colors.sequential.Blues, "qual": px.colors.qualitative.Prism, "template": "plotly_white"},
    "Vibrant": {"seq": px.colors.sequential.Plasma, "qual": px.colors.qualitative.Bold, "template": "plotly_white"},
    "Ocean": {"seq": px.colors.sequential.Teal, "qual": px.colors.qualitative.Set2, "template": "plotly_white"},
    "Sunset": {"seq": px.colors.sequential.Oranges, "qual": px.colors.qualitative.Vivid, "template": "plotly_white"},
    "Forest": {"seq": px.colors.sequential.Greens, "qual": px.colors.qualitative.Dark2, "template": "plotly_white"},
    "Dark Mode": {"seq": px.colors.sequential.Inferno, "qual": px.colors.qualitative.D3, "template": "plotly_dark"},
    "Monochrome": {"seq": px.colors.sequential.Greys, "qual": px.colors.qualitative.Pastel, "template": "ggplot2"},
}

if "theme" not in st.session_state:
    st.session_state.theme = "Corporate Blue"

with st.sidebar:
    st.header("🎨 Appearance")
    st.session_state.theme = st.selectbox("Color Theme", list(COLOR_THEMES.keys()),
                                           index=list(COLOR_THEMES.keys()).index(st.session_state.theme))

theme = COLOR_THEMES[st.session_state.theme]
px.defaults.template = theme["template"]
px.defaults.color_continuous_scale = theme["seq"]
px.defaults.color_discrete_sequence = theme["qual"]

def explain(text):
    """Render an auto-generated explanation box."""
    st.markdown(
        f"""<div style="background:#eef5fc;border-left:5px solid #2e75b6;
        padding:12px 16px;border-radius:8px;margin:10px 0;font-size:0.95rem;color:#1f4e79;">
        🧠 <b>Interpretation:</b> {text}</div>""",
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="header-banner">
        <h1>📊 Advanced Data Analysis Studio</h1>
        <p>Enterprise-grade exploratory analysis, statistical testing, regression, clustering and visualization — all in one workspace.</p>
        <div class="header-badges">
            <span>📈 13+ Chart Types</span>
            <span>🧪 8 Statistical Tests</span>
            <span>📉 Regression & VIF</span>
            <span>🧩 K-Means & PCA</span>
            <span>🧹 Data Cleaning</span>
            <span>⬇️ Export Ready</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "df_original" not in st.session_state:
    st.session_state.df_original = None

# ------------------------------------------------------------------
# DATA UPLOAD
# ------------------------------------------------------------------
with st.sidebar:
    st.header("1️⃣ Data Source")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                xls = pd.ExcelFile(uploaded_file)
                sheet = st.selectbox("Select sheet", xls.sheet_names)
                df = pd.read_excel(uploaded_file, sheet_name=sheet)

            st.session_state.df = df
            st.session_state.df_original = df.copy()
            st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error loading file: {e}")

if st.session_state.df is None:
    st.info("👈 Upload a CSV or Excel file from the sidebar to begin.")
    st.stop()

df = st.session_state.df

# ------------------------------------------------------------------
# SIDEBAR: DATA CLEANING TOOLS
# ------------------------------------------------------------------
with st.sidebar:
    st.header("2️⃣ Data Cleaning")

    with st.expander("Handle Missing Values"):
        missing_strategy = st.selectbox(
            "Strategy",
            ["None", "Drop rows with any NA", "Fill numeric with mean",
             "Fill numeric with median", "Fill with zero", "Forward fill", "Backward fill"]
        )
        if st.button("Apply Missing Value Strategy"):
            if missing_strategy == "Drop rows with any NA":
                df = df.dropna()
            elif missing_strategy == "Fill numeric with mean":
                num_cols = df.select_dtypes(include=np.number).columns
                df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
            elif missing_strategy == "Fill numeric with median":
                num_cols = df.select_dtypes(include=np.number).columns
                df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            elif missing_strategy == "Fill with zero":
                df = df.fillna(0)
            elif missing_strategy == "Forward fill":
                df = df.fillna(method="ffill")
            elif missing_strategy == "Backward fill":
                df = df.fillna(method="bfill")
            st.session_state.df = df
            st.success("Applied.")

    with st.expander("Remove Duplicates"):
        if st.button("Drop Duplicate Rows"):
            before = df.shape[0]
            df = df.drop_duplicates()
            st.session_state.df = df
            st.success(f"Removed {before - df.shape[0]} duplicate rows.")

    with st.expander("Column Type Conversion"):
        col_to_convert = st.selectbox("Column", df.columns, key="convert_col")
        new_type = st.selectbox("Convert to", ["numeric", "datetime", "category", "string"])
        if st.button("Convert"):
            try:
                if new_type == "numeric":
                    df[col_to_convert] = pd.to_numeric(df[col_to_convert], errors="coerce")
                elif new_type == "datetime":
                    df[col_to_convert] = pd.to_datetime(df[col_to_convert], errors="coerce")
                elif new_type == "category":
                    df[col_to_convert] = df[col_to_convert].astype("category")
                else:
                    df[col_to_convert] = df[col_to_convert].astype(str)
                st.session_state.df = df
                st.success(f"Converted {col_to_convert} to {new_type}.")
            except Exception as e:
                st.error(str(e))

    with st.expander("Reset Data"):
        if st.button("Reset to Original Upload"):
            st.session_state.df = st.session_state.df_original.copy()
            st.success("Data reset.")

df = st.session_state.df
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
all_cols = df.columns.tolist()

# ------------------------------------------------------------------
# MAIN NAVIGATION
# ------------------------------------------------------------------
with st.sidebar:
    st.header("3️⃣ Navigate")
    section = st.radio(
        "Go to",
        [
            "Data Overview",
            "Descriptive Statistics",
            "Visualizations",
            "Hypothesis Testing",
            "Correlation & Relationships",
            "Regression Analysis",
            "Clustering & PCA",
            "Smart Insights & Presentation Tips",
            "Export Results",
        ],
    )

# ====================================================================
# SECTION 1: DATA OVERVIEW
# ====================================================================
if section == "Data Overview":
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.markdown("#### Preview")
    n_rows = st.slider("Rows to display", 5, 100, 10)
    st.dataframe(df.head(n_rows), use_container_width=True)

    st.markdown("#### Column Information")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str),
        "Non-Null Count": df.notnull().sum(),
        "Null Count": df.isnull().sum(),
        "Unique Values": df.nunique(),
    })
    st.dataframe(info_df, use_container_width=True)

    st.markdown("#### Missing Values Map")
    if df.isnull().sum().sum() > 0:
        fig = px.imshow(
            df.isnull().T,
            color_continuous_scale=["#1f4e79", "#e74c3c"],
            aspect="auto",
            labels=dict(color="Missing"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values detected.")

# ====================================================================
# SECTION 2: DESCRIPTIVE STATISTICS
# ====================================================================
elif section == "Descriptive Statistics":
    st.subheader("Descriptive Statistics")

    if numeric_cols:
        st.markdown("#### Numeric Summary")
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

        st.markdown("#### Additional Statistics")
        adv_stats = pd.DataFrame({
            "Skewness": df[numeric_cols].skew(),
            "Kurtosis": df[numeric_cols].kurt(),
            "Variance": df[numeric_cols].var(),
            "Std Dev": df[numeric_cols].std(),
            "Range": df[numeric_cols].max() - df[numeric_cols].min(),
            "IQR": df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25),
            "Coefficient of Variation": df[numeric_cols].std() / df[numeric_cols].mean(),
        })
        st.dataframe(adv_stats, use_container_width=True)

        most_skewed = adv_stats["Skewness"].abs().idxmax()
        skew_val = adv_stats.loc[most_skewed, "Skewness"]
        skew_desc = "highly skewed" if abs(skew_val) > 1 else "moderately skewed" if abs(skew_val) > 0.5 else "roughly symmetric"
        direction = "right (positive)" if skew_val > 0 else "left (negative)"
        most_variable = adv_stats["Coefficient of Variation"].abs().idxmax()

        explain(
            f"<b>{most_skewed}</b> is the most {skew_desc} variable, skewed to the {direction} "
            f"(skewness = {skew_val:.2f}). <b>{most_variable}</b> shows the highest relative variability "
            f"(coefficient of variation = {adv_stats.loc[most_variable, 'Coefficient of Variation']:.2f}), "
            "meaning it varies the most relative to its own mean — worth investigating for outliers or "
            "natural subgroups."
        )

    if categorical_cols:
        st.markdown("#### Categorical Summary")
        cat_col = st.selectbox("Select categorical column", categorical_cols)
        freq = df[cat_col].value_counts().reset_index()
        freq.columns = [cat_col, "Count"]
        freq["Percentage"] = (freq["Count"] / freq["Count"].sum() * 100).round(2)
        st.dataframe(freq, use_container_width=True)

    st.markdown("#### Normality Tests (Shapiro-Wilk)")
    if numeric_cols:
        norm_col = st.selectbox("Column to test", numeric_cols, key="norm_test")
        data = df[norm_col].dropna()
        if 3 <= len(data) <= 5000:
            stat, p = stats.shapiro(data)
            st.write(f"**Statistic:** {stat:.4f} | **p-value:** {p:.4f}")
            if p < 0.05:
                st.warning("Data significantly deviates from a normal distribution (p < 0.05).")
            else:
                st.success("Data does not significantly deviate from normality (p ≥ 0.05).")
        else:
            st.info("Shapiro-Wilk test requires between 3 and 5000 observations.")

# ====================================================================
# SECTION 3: VISUALIZATIONS
# ====================================================================
elif section == "Visualizations":
    st.subheader("Interactive Visualizations")

    chart_type = st.selectbox(
        "Chart Type",
        [
            "Histogram", "Box Plot", "Violin Plot", "Scatter Plot", "Line Chart",
            "Bar Chart", "Pie Chart", "Heatmap (Correlation)", "Pair Plot",
            "Density Contour", "Area Chart", "Sunburst", "Treemap"
        ],
    )

    if chart_type == "Histogram":
        col = st.selectbox("Column", numeric_cols)
        bins = st.slider("Number of bins", 5, 100, 30)
        color_by = st.selectbox("Color by (optional)", [None] + categorical_cols)
        fig = px.histogram(df, x=col, nbins=bins, color=color_by, marginal="box")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot":
        y_col = st.selectbox("Numeric column (Y)", numeric_cols)
        x_col = st.selectbox("Group by (X, optional)", [None] + categorical_cols)
        fig = px.box(df, x=x_col, y=y_col, points="all")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Violin Plot":
        y_col = st.selectbox("Numeric column (Y)", numeric_cols, key="violin_y")
        x_col = st.selectbox("Group by (X, optional)", [None] + categorical_cols, key="violin_x")
        fig = px.violin(df, x=x_col, y=y_col, box=True, points="all")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":
        x_col = st.selectbox("X axis", numeric_cols, key="scatter_x")
        y_col = st.selectbox("Y axis", numeric_cols, key="scatter_y")
        color_col = st.selectbox("Color by (optional)", [None] + all_cols, key="scatter_color")
        size_col = st.selectbox("Size by (optional)", [None] + numeric_cols, key="scatter_size")
        trendline = st.checkbox("Add trendline (OLS)")
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_col, size=size_col,
            trendline="ols" if trendline else None,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Line Chart":
        x_col = st.selectbox("X axis", all_cols, key="line_x")
        y_cols = st.multiselect("Y axis (one or more)", numeric_cols, key="line_y")
        if y_cols:
            fig = px.line(df, x=x_col, y=y_cols, markers=True)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bar Chart":
        x_col = st.selectbox("Category (X)", categorical_cols + numeric_cols, key="bar_x")
        y_col = st.selectbox("Value (Y)", numeric_cols, key="bar_y")
        agg_func = st.selectbox("Aggregation", ["sum", "mean", "median", "count", "max", "min"])
        grouped = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
        fig = px.bar(grouped, x=x_col, y=y_col, color=x_col)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie Chart":
        cat_col = st.selectbox("Category column", categorical_cols, key="pie_cat")
        counts = df[cat_col].value_counts().reset_index()
        counts.columns = [cat_col, "count"]
        fig = px.pie(counts, names=cat_col, values="count", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Heatmap (Correlation)":
        if len(numeric_cols) >= 2:
            method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
            corr = df[numeric_cols].corr(method=method)
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns.")

    elif chart_type == "Pair Plot":
        cols_for_pair = st.multiselect("Select columns (2-5 recommended)", numeric_cols, default=numeric_cols[:3])
        color_col = st.selectbox("Color by", [None] + categorical_cols, key="pair_color")
        if len(cols_for_pair) >= 2:
            fig = px.scatter_matrix(df, dimensions=cols_for_pair, color=color_col)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Density Contour":
        x_col = st.selectbox("X axis", numeric_cols, key="dens_x")
        y_col = st.selectbox("Y axis", numeric_cols, key="dens_y")
        fig = px.density_contour(df, x=x_col, y=y_col)
        fig.update_traces(contours_coloring="fill", contours_showlabels=True)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Area Chart":
        x_col = st.selectbox("X axis", all_cols, key="area_x")
        y_cols = st.multiselect("Y axis", numeric_cols, key="area_y")
        if y_cols:
            fig = px.area(df, x=x_col, y=y_cols)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Sunburst":
        path_cols = st.multiselect("Hierarchy (order matters)", categorical_cols, key="sun_path")
        value_col = st.selectbox("Value", [None] + numeric_cols, key="sun_val")
        if len(path_cols) >= 1:
            fig = px.sunburst(df, path=path_cols, values=value_col)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Treemap":
        path_cols = st.multiselect("Hierarchy (order matters)", categorical_cols, key="tree_path")
        value_col = st.selectbox("Value", [None] + numeric_cols, key="tree_val")
        if len(path_cols) >= 1:
            fig = px.treemap(df, path=path_cols, values=value_col)
            st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# SECTION 4: HYPOTHESIS TESTING
# ====================================================================
elif section == "Hypothesis Testing":
    st.subheader("Hypothesis Testing Toolkit")

    test = st.selectbox(
        "Select a Test",
        [
            "One-Sample T-Test",
            "Independent Samples T-Test",
            "Paired Samples T-Test",
            "One-Way ANOVA",
            "Chi-Square Test of Independence",
            "Mann-Whitney U Test",
            "Wilcoxon Signed-Rank Test",
            "Kruskal-Wallis Test",
        ],
    )

    alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)

    if test == "One-Sample T-Test":
        col = st.selectbox("Numeric variable", numeric_cols)
        pop_mean = st.number_input("Hypothesized population mean", value=0.0)
        data = df[col].dropna()
        t_stat, p = stats.ttest_1samp(data, pop_mean)
        st.write(f"**t-statistic:** {t_stat:.4f} | **p-value:** {p:.4f}")
        st.write(f"Sample mean: {data.mean():.4f} | Sample size: {len(data)}")
        st.success("Reject H0: mean differs significantly." if p < alpha else "Fail to reject H0.")
        explain(
            f"The sample mean of <b>{col}</b> ({data.mean():.3f}) "
            + (f"differs significantly from the hypothesized value of {pop_mean} (p = {p:.4f} &lt; {alpha})."
               if p < alpha else
               f"is not significantly different from the hypothesized value of {pop_mean} (p = {p:.4f} ≥ {alpha}).")
        )

    elif test == "Independent Samples T-Test":
        group_col = st.selectbox("Grouping variable", categorical_cols + all_cols, key="ind_t_group")
        value_col = st.selectbox("Numeric variable", numeric_cols, key="ind_t_val")
        groups = df[group_col].dropna().unique()
        if len(groups) >= 2:
            g1_name = st.selectbox("Group 1", groups, key="g1")
            g2_name = st.selectbox("Group 2", [g for g in groups if g != g1_name], key="g2")
            g1 = df[df[group_col] == g1_name][value_col].dropna()
            g2 = df[df[group_col] == g2_name][value_col].dropna()
            equal_var = st.checkbox("Assume equal variances", value=False)
            t_stat, p = stats.ttest_ind(g1, g2, equal_var=equal_var)
            c1, c2 = st.columns(2)
            c1.metric(f"Mean ({g1_name})", f"{g1.mean():.3f}")
            c2.metric(f"Mean ({g2_name})", f"{g2.mean():.3f}")
            st.write(f"**t-statistic:** {t_stat:.4f} | **p-value:** {p:.4f}")
            st.success("Significant difference between groups." if p < alpha else "No significant difference.")
            higher = g1_name if g1.mean() > g2.mean() else g2_name
            explain(
                (f"There is a statistically significant difference in <b>{value_col}</b> between "
                 f"<b>{g1_name}</b> and <b>{g2_name}</b> (p = {p:.4f}). On average, <b>{higher}</b> has the higher value. "
                 "This difference is unlikely due to chance."
                 if p < alpha else
                 f"There is no statistically significant difference in <b>{value_col}</b> between "
                 f"<b>{g1_name}</b> and <b>{g2_name}</b> (p = {p:.4f}). Observed differences could be due to random variation.")
            )
        else:
            st.error("Grouping variable needs at least 2 categories.")

    elif test == "Paired Samples T-Test":
        col1 = st.selectbox("Variable 1 (e.g., before)", numeric_cols, key="paired1")
        col2 = st.selectbox("Variable 2 (e.g., after)", numeric_cols, key="paired2")
        valid = df[[col1, col2]].dropna()
        t_stat, p = stats.ttest_rel(valid[col1], valid[col2])
        st.write(f"**t-statistic:** {t_stat:.4f} | **p-value:** {p:.4f}")
        st.write(f"Mean difference: {(valid[col1] - valid[col2]).mean():.4f}")
        st.success("Significant difference between paired measurements." if p < alpha else "No significant difference.")

    elif test == "One-Way ANOVA":
        group_col = st.selectbox("Grouping variable", categorical_cols + all_cols, key="anova_group")
        value_col = st.selectbox("Numeric variable", numeric_cols, key="anova_val")
        groups = [df[df[group_col] == g][value_col].dropna() for g in df[group_col].dropna().unique()]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            f_stat, p = stats.f_oneway(*groups)
            st.write(f"**F-statistic:** {f_stat:.4f} | **p-value:** {p:.4f}")
            st.success("At least one group mean differs significantly." if p < alpha else "No significant difference among groups.")
            explain(
                (f"The average <b>{value_col}</b> differs significantly across levels of <b>{group_col}</b> "
                 f"(F = {f_stat:.2f}, p = {p:.4f}). At least one group stands out — check the Tukey HSD "
                 "post-hoc test below to see which specific pairs differ."
                 if p < alpha else
                 f"The average <b>{value_col}</b> does not differ significantly across levels of <b>{group_col}</b> "
                 f"(F = {f_stat:.2f}, p = {p:.4f}). Group means appear statistically similar.")
            )

            st.markdown("#### Group Means")
            means = df.groupby(group_col)[value_col].agg(["mean", "std", "count"])
            st.dataframe(means, use_container_width=True)

            with st.expander("Post-hoc: Tukey HSD"):
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                clean = df[[group_col, value_col]].dropna()
                tukey = pairwise_tukeyhsd(clean[value_col], clean[group_col], alpha=alpha)
                st.text(str(tukey))
        else:
            st.error("Need at least 2 groups.")

    elif test == "Chi-Square Test of Independence":
        col1 = st.selectbox("Variable 1", categorical_cols, key="chi1")
        col2 = st.selectbox("Variable 2", [c for c in categorical_cols if c != col1], key="chi2")
        contingency = pd.crosstab(df[col1], df[col2])
        st.markdown("#### Contingency Table")
        st.dataframe(contingency, use_container_width=True)
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        st.write(f"**Chi-square statistic:** {chi2:.4f} | **p-value:** {p:.4f} | **Degrees of freedom:** {dof}")
        st.success("Significant association between variables." if p < alpha else "No significant association.")
        explain(
            (f"<b>{col1}</b> and <b>{col2}</b> appear to be related (chi-square = {chi2:.2f}, p = {p:.4f}). "
             "Knowing one variable's category gives information about the other — useful for segmentation or targeting."
             if p < alpha else
             f"<b>{col1}</b> and <b>{col2}</b> appear to be independent (chi-square = {chi2:.2f}, p = {p:.4f}). "
             "Knowing one variable's category does not help predict the other.")
        )

    elif test == "Mann-Whitney U Test":
        group_col = st.selectbox("Grouping variable", categorical_cols + all_cols, key="mw_group")
        value_col = st.selectbox("Numeric variable", numeric_cols, key="mw_val")
        groups = df[group_col].dropna().unique()
        if len(groups) >= 2:
            g1_name = st.selectbox("Group 1", groups, key="mw_g1")
            g2_name = st.selectbox("Group 2", [g for g in groups if g != g1_name], key="mw_g2")
            g1 = df[df[group_col] == g1_name][value_col].dropna()
            g2 = df[df[group_col] == g2_name][value_col].dropna()
            stat, p = stats.mannwhitneyu(g1, g2)
            st.write(f"**U statistic:** {stat:.4f} | **p-value:** {p:.4f}")
            st.success("Significant difference in distributions." if p < alpha else "No significant difference.")

    elif test == "Wilcoxon Signed-Rank Test":
        col1 = st.selectbox("Variable 1", numeric_cols, key="wil1")
        col2 = st.selectbox("Variable 2", numeric_cols, key="wil2")
        valid = df[[col1, col2]].dropna()
        stat, p = stats.wilcoxon(valid[col1], valid[col2])
        st.write(f"**Statistic:** {stat:.4f} | **p-value:** {p:.4f}")
        st.success("Significant difference between paired samples." if p < alpha else "No significant difference.")

    elif test == "Kruskal-Wallis Test":
        group_col = st.selectbox("Grouping variable", categorical_cols + all_cols, key="kw_group")
        value_col = st.selectbox("Numeric variable", numeric_cols, key="kw_val")
        groups = [df[df[group_col] == g][value_col].dropna() for g in df[group_col].dropna().unique()]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            stat, p = stats.kruskal(*groups)
            st.write(f"**H statistic:** {stat:.4f} | **p-value:** {p:.4f}")
            st.success("Significant difference among groups." if p < alpha else "No significant difference.")

# ====================================================================
# SECTION 5: CORRELATION & RELATIONSHIPS
# ====================================================================
elif section == "Correlation & Relationships":
    st.subheader("Correlation & Relationships")

    if len(numeric_cols) >= 2:
        method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
        corr = df[numeric_cols].corr(method=method)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Pairwise Correlation Test")
        col1 = st.selectbox("Variable X", numeric_cols, key="corr_x")
        col2 = st.selectbox("Variable Y", [c for c in numeric_cols if c != col1], key="corr_y")
        valid = df[[col1, col2]].dropna()

        if method == "pearson":
            r, p = stats.pearsonr(valid[col1], valid[col2])
        elif method == "spearman":
            r, p = stats.spearmanr(valid[col1], valid[col2])
        else:
            r, p = stats.kendalltau(valid[col1], valid[col2])

        c1, c2 = st.columns(2)
        c1.metric("Correlation coefficient", f"{r:.4f}")
        c2.metric("p-value", f"{p:.4f}")

        fig2 = px.scatter(df, x=col1, y=col2, trendline="ols")
        st.plotly_chart(fig2, use_container_width=True)

        strength = (
            "very strong" if abs(r) >= 0.8 else
            "strong" if abs(r) >= 0.6 else
            "moderate" if abs(r) >= 0.4 else
            "weak" if abs(r) >= 0.2 else
            "negligible"
        )
        direction = "positive" if r > 0 else "negative"
        sig_text = "statistically significant" if p < 0.05 else "not statistically significant"
        explain(
            f"There is a <b>{strength} {direction}</b> relationship between <b>{col1}</b> and <b>{col2}</b> "
            f"(r = {r:.2f}, p = {p:.4f}), which is {sig_text}. "
            + ("As one increases, the other tends to increase as well."
               if r > 0 else "As one increases, the other tends to decrease."
               if r < 0 else "There is little to no linear relationship.")
            + " Remember: correlation does not imply causation."
        )

        # Top correlated pairs across the dataset
        with st.expander("📌 Strongest relationships in the dataset"):
            corr_unstacked = corr.where(~np.eye(len(corr), dtype=bool)).unstack().dropna()
            corr_unstacked = corr_unstacked[corr_unstacked.index.get_level_values(0) < corr_unstacked.index.get_level_values(1)]
            top_corr = corr_unstacked.abs().sort_values(ascending=False).head(5)
            for (a, b), val in top_corr.items():
                actual = corr.loc[a, b]
                st.write(f"• **{a}** ↔ **{b}**: r = {actual:.2f}")
    else:
        st.warning("Need at least 2 numeric columns for correlation analysis.")

# ====================================================================
# SECTION 6: REGRESSION ANALYSIS
# ====================================================================
elif section == "Regression Analysis":
    st.subheader("Regression Analysis")

    reg_type = st.radio("Regression Type", ["Simple Linear Regression", "Multiple Linear Regression"])

    if reg_type == "Simple Linear Regression":
        x_col = st.selectbox("Independent variable (X)", numeric_cols, key="slr_x")
        y_col = st.selectbox("Dependent variable (Y)", [c for c in numeric_cols if c != x_col], key="slr_y")

        valid = df[[x_col, y_col]].dropna()
        X = sm.add_constant(valid[x_col])
        model = sm.OLS(valid[y_col], X).fit()

        st.text(model.summary())

        fig = px.scatter(valid, x=x_col, y=y_col, trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

        coef = model.params[x_col]
        r2 = model.rsquared
        p_val = model.pvalues[x_col]
        sig = "a statistically significant" if p_val < 0.05 else "not a statistically significant"
        explain(
            f"For every 1-unit increase in <b>{x_col}</b>, <b>{y_col}</b> changes by "
            f"<b>{coef:.3f}</b> on average. This relationship is {sig} predictor (p = {p_val:.4f}). "
            f"The model explains <b>{r2*100:.1f}%</b> of the variation in {y_col} (R² = {r2:.3f})."
            + (" This is a fairly strong fit." if r2 > 0.6 else " Other factors likely also influence " + y_col + ".")
        )

    else:
        y_col = st.selectbox("Dependent variable (Y)", numeric_cols, key="mlr_y")
        x_cols = st.multiselect("Independent variables (X)", [c for c in numeric_cols if c != y_col], key="mlr_x")

        if len(x_cols) >= 1:
            valid = df[[y_col] + x_cols].dropna()
            X = sm.add_constant(valid[x_cols])
            model = sm.OLS(valid[y_col], X).fit()

            st.text(model.summary())

            st.markdown("#### Predicted vs Actual")
            preds = model.predict(X)
            fig = px.scatter(x=valid[y_col], y=preds, labels={"x": "Actual", "y": "Predicted"})
            fig.add_shape(
                type="line",
                x0=valid[y_col].min(), y0=valid[y_col].min(),
                x1=valid[y_col].max(), y1=valid[y_col].max(),
                line=dict(color="red", dash="dash"),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Residual Plot")
            residuals = valid[y_col] - preds
            fig2 = px.scatter(x=preds, y=residuals, labels={"x": "Predicted", "y": "Residuals"})
            fig2.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig2, use_container_width=True)

            r2 = model.rsquared
            adj_r2 = model.rsquared_adj
            sig_vars = [v for v in x_cols if model.pvalues[v] < 0.05]
            best_var = model.params[x_cols].abs().idxmax()
            explain(
                f"This model explains <b>{r2*100:.1f}%</b> of the variation in <b>{y_col}</b> "
                f"(R² = {r2:.3f}, Adjusted R² = {adj_r2:.3f}). "
                + (f"Statistically significant predictors (p &lt; 0.05): <b>{', '.join(sig_vars)}</b>. "
                   if sig_vars else "None of the predictors were statistically significant at the 0.05 level. ")
                + f"<b>{best_var}</b> has the largest estimated effect per unit change. "
                + ("The model fits the data reasonably well." if r2 > 0.5 else
                   "Consider adding more relevant variables — the current predictors explain only a portion of the variation.")
            )

            if len(x_cols) >= 2:
                st.markdown("#### Multicollinearity (VIF)")
                vif_data = pd.DataFrame()
                vif_data["Feature"] = x_cols
                vif_data["VIF"] = [variance_inflation_factor(valid[x_cols].values, i) for i in range(len(x_cols))]
                st.dataframe(vif_data, use_container_width=True)
                st.caption("VIF > 10 indicates potential multicollinearity concerns.")
        else:
            st.info("Select at least one independent variable.")

# ====================================================================
# SECTION 7: CLUSTERING & PCA
# ====================================================================
elif section == "Clustering & PCA":
    st.subheader("Clustering & Dimensionality Reduction")

    analysis_choice = st.radio("Choose Analysis", ["K-Means Clustering", "Principal Component Analysis (PCA)"])

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
    else:
        cols_used = st.multiselect("Select numeric columns to use", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])

        if len(cols_used) >= 2:
            valid = df[cols_used].dropna()
            scaler = StandardScaler()
            scaled = scaler.fit_transform(valid)

            if analysis_choice == "K-Means Clustering":
                k = st.slider("Number of clusters (k)", 2, 10, 3)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(scaled)

                result = valid.copy()
                result["Cluster"] = clusters.astype(str)

                x_axis = st.selectbox("X axis for plot", cols_used, key="km_x")
                y_axis = st.selectbox("Y axis for plot", [c for c in cols_used if c != x_axis], key="km_y")

                fig = px.scatter(result, x=x_axis, y=y_axis, color="Cluster", title="K-Means Clustering Result")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Cluster Centers (original scale)")
                centers = scaler.inverse_transform(kmeans.cluster_centers_)
                centers_df = pd.DataFrame(centers, columns=cols_used)
                st.dataframe(centers_df, use_container_width=True)

                # Elbow method
                with st.expander("Elbow Method (optimal k)"):
                    inertias = []
                    k_range = range(1, 11)
                    for k_val in k_range:
                        km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
                        km.fit(scaled)
                        inertias.append(km.inertia_)
                    fig_elbow = px.line(x=list(k_range), y=inertias, markers=True,
                                         labels={"x": "Number of clusters", "y": "Inertia"})
                    st.plotly_chart(fig_elbow, use_container_width=True)

            else:  # PCA
                n_components = st.slider("Number of components", 2, min(len(cols_used), 5), 2)
                pca = PCA(n_components=n_components)
                components = pca.fit_transform(scaled)

                explained = pca.explained_variance_ratio_
                st.markdown("#### Explained Variance Ratio")
                exp_df = pd.DataFrame({
                    "Component": [f"PC{i+1}" for i in range(n_components)],
                    "Explained Variance Ratio": explained,
                    "Cumulative": np.cumsum(explained),
                })
                st.dataframe(exp_df, use_container_width=True)

                pc_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])
                fig = px.scatter(pc_df, x="PC1", y="PC2", title="PCA: First Two Components")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Component Loadings")
                loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(n_components)], index=cols_used)
                st.dataframe(loadings, use_container_width=True)

# ====================================================================
# SECTION: SMART INSIGHTS & PRESENTATION TIPS
# ====================================================================
elif section == "Smart Insights & Presentation Tips":
    st.subheader("🧠 Smart Insights & Presentation Recommendations")
    st.caption("An automated overview of your dataset with suggestions on how to present it effectively.")

    n_rows, n_cols = df.shape
    pct_missing = df.isnull().sum().sum() / (n_rows * n_cols) * 100

    st.markdown("### 1. Dataset Summary")
    explain(
        f"Your dataset contains <b>{n_rows:,} rows</b> and <b>{n_cols} columns</b>, including "
        f"<b>{len(numeric_cols)} numeric</b>, <b>{len(categorical_cols)} categorical</b>, and "
        f"<b>{len(datetime_cols)} date/time</b> columns. Overall missingness is "
        f"<b>{pct_missing:.1f}%</b>."
        + (" Consider addressing missing data before deeper analysis." if pct_missing > 5 else
           " Data completeness looks good.")
    )

    if numeric_cols:
        st.markdown("### 2. Notable Numeric Patterns")
        desc = df[numeric_cols].describe().T
        skewed = df[numeric_cols].skew().abs().sort_values(ascending=False)

        bullets = []
        for col in numeric_cols[:6]:
            bullets.append(
                f"<b>{col}</b>: ranges from {desc.loc[col,'min']:.2f} to {desc.loc[col,'max']:.2f}, "
                f"averaging {desc.loc[col,'mean']:.2f}."
            )
        explain("<br>".join(bullets))

        if skewed.iloc[0] > 1:
            top_skew_col = skewed.index[0]
            explain(
                f"<b>{top_skew_col}</b> is highly skewed — a <b>box plot</b> or <b>log-scaled histogram</b> "
                "will represent it better than a simple bar chart of averages."
            )

    if categorical_cols:
        st.markdown("### 3. Categorical Breakdown")
        for col in categorical_cols[:4]:
            n_unique = df[col].nunique()
            top_val = df[col].value_counts().idxmax()
            top_pct = df[col].value_counts(normalize=True).max() * 100
            explain(
                f"<b>{col}</b> has <b>{n_unique}</b> unique categories. The most common is "
                f"<b>'{top_val}'</b>, representing <b>{top_pct:.1f}%</b> of records."
                + (" With many categories, a treemap or sorted bar chart works better than a pie chart."
                   if n_unique > 6 else " A pie chart or donut chart would summarize this well.")
            )

    if datetime_cols:
        st.markdown("### 4. Time-Based Trends")
        explain(
            f"Your dataset includes date/time column(s): <b>{', '.join(datetime_cols)}</b>. "
            "Consider a <b>line chart</b> or <b>area chart</b> over time to reveal trends, seasonality, "
            "or growth patterns."
        )

    st.markdown("### 5. Recommended Chart Types for Your Data")
    recs = []
    if numeric_cols:
        recs.append(("Distribution of a single numeric variable", "Histogram or Box Plot"))
        recs.append(("Comparing a numeric variable across categories", "Box Plot, Violin Plot, or Bar Chart"))
    if len(numeric_cols) >= 2:
        recs.append(("Relationship between two numeric variables", "Scatter Plot with trendline"))
        recs.append(("Overview of all numeric relationships", "Correlation Heatmap"))
    if categorical_cols:
        recs.append(("Composition of a categorical variable", "Pie Chart (few categories) or Treemap (many categories)"))
    if categorical_cols and numeric_cols:
        recs.append(("Comparing groups", "Grouped Bar Chart"))
    if datetime_cols and numeric_cols:
        recs.append(("Trends over time", "Line Chart or Area Chart"))
    if len(numeric_cols) >= 2:
        recs.append(("Identifying natural groupings", "K-Means Clustering scatter plot"))

    rec_df = pd.DataFrame(recs, columns=["When you want to show...", "Use this chart"])
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    st.markdown("### 6. Presenting to Different Audiences")
    st.markdown(
        """
        - **Executives / Management:** Lead with 2–3 key metrics (use Metric cards), one headline chart,
          and a one-sentence takeaway. Avoid statistical jargon — focus on business implications.
        - **Analysts / Technical Teams:** Include correlation heatmaps, regression summaries, p-values,
          and confidence intervals. Show methodology alongside results.
        - **General / Public Reports:** Use simple, labeled charts (bar, line, pie) with clear titles and
          minimal numbers on screen. Highlight one main story per chart.
        """
    )

    st.markdown("### 7. Overall Take")
    conclusion_points = []
    if pct_missing > 5:
        conclusion_points.append("address missing data")
    if numeric_cols and df[numeric_cols].skew().abs().max() > 1:
        conclusion_points.append("account for skewed variables when choosing summary statistics (prefer median over mean)")
    if len(numeric_cols) >= 2:
        conclusion_points.append("explore correlations before building predictive models")
    if categorical_cols:
        conclusion_points.append("review category balance, especially for any group-based comparisons")

    if conclusion_points:
        explain("Before finalizing your analysis, consider: " + "; ".join(conclusion_points) + ".")
    else:
        explain("Your data looks clean and well-structured — ready for deeper statistical modeling and presentation.")

# ====================================================================
# SECTION 8: EXPORT RESULTS
# ====================================================================
elif section == "Export Results":
    st.subheader("Export Cleaned Data")

    st.markdown("#### Current Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Cleaned Data as CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv",
    )

    st.markdown("#### Download Summary Statistics")
    if numeric_cols:
        summary_csv = df[numeric_cols].describe().T.to_csv().encode("utf-8")
        st.download_button(
            label="⬇️ Download Summary Statistics as CSV",
            data=summary_csv,
            file_name="summary_statistics.csv",
            mime="text/csv",
        )

st.divider()
st.markdown(
    """
    <div class="footer-box">
        <b>Advanced Data Analysis Studio</b><br>
        Built with Streamlit · Pandas · SciPy · Statsmodels · Scikit-learn · Plotly<br>
        For internal analytics use — verify statistical assumptions before reporting results.
    </div>
    """,
    unsafe_allow_html=True,
)
