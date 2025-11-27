import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns

# Page config
st.set_page_config(page_title="Divorce Analysis", page_icon="💔")

# Title
st.title("Divorce Prediction Analysis")
st.write("Analysing factors that influence divorce")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/tishsrisasi/my-first-project/refs/heads/main/divorce_df.csv")
    return df

df = load_data()

# Basic info
st.header("Dataset Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Couples", len(df))
with col2:
    st.metric("Features", len(df.columns))
with col3:
    st.metric("Divorce Rate", f"{df['divorced'].mean():.1%}")

# Show data
st.subheader("Sample Data")
st.dataframe(df.head())

# Basic statistics
st.subheader("Basic Statistics")
st.write(df.describe())

# Simple visualizations
st.header("Visualizations")

# 1. Divorce distribution
st.subheader("Divorce Distribution")
fig1, ax1 = plt.subplots()
divorce_counts = df["divorced"].value_counts()
colors = ["#2ecc71", "#e74c3c"]
ax1.pie(
    divorce_counts,
    labels=["Not Divorced", "Divorced"],
    autopct="%1.1f%%",
    colors=colors,
)
st.pyplot(fig1)

# 2. Age at marriage distribution
st.subheader("Age at Marriage Distribution")
fig2, ax2 = plt.subplots()
ax2.hist(df["age_at_marriage"], bins=30, color="skyblue", edgecolor="black")
ax2.set_xlabel("Age")
ax2.set_ylabel("Count")
st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("**App 1:** Basic data overview and simple charts")

# NEW: Additional visualizations
st.header("Detailed Analysis")

# 3. Divorce by key factors
col3, col4 = st.columns(2)

with col3:
    st.subheader("Divorce Rate by Infidelity")
    fig3, ax3 = plt.subplots()
    divorce_by_infidelity = df.groupby('infidelity_occurred')['divorced'].mean()
    ax3.bar(['No Infidelity', 'Infidelity'], divorce_by_infidelity.values, 
            color=['green', 'red'])
    ax3.set_ylabel('Divorce Rate')
    ax3.set_ylim(0, 1)
    st.pyplot(fig3)

with col4:
    st.subheader("Divorce Rate by Children")
    fig4, ax4 = plt.subplots()
    divorce_by_children = df.groupby('num_children')['divorced'].mean()
    ax4.bar(divorce_by_children.index, divorce_by_children.values, color='coral')
    ax4.set_xlabel('Number of Children')
    ax4.set_ylabel('Divorce Rate')
    st.pyplot(fig4)

# Correlation heatmap
st.header("Correlation Analysis")
numerical_cols = df.select_dtypes(include='number').columns.tolist()
correlation_matrix = df[numerical_cols].corr()

fig5, ax5 = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=ax5)
ax5.set_title('Feature Correlation Matrix')
st.pyplot(fig5)

# Top correlations with divorce
st.subheader("Top Factors Correlated with Divorce")
correlations = correlation_matrix['divorced'].sort_values(ascending=False)[1:11]
fig6, ax6 = plt.subplots()
colors = ['red' if x > 0 else 'green' for x in correlations.values]
ax6.barh(correlations.index, correlations.values, color=colors)
ax6.set_xlabel('Correlation with Divorce')
st.pyplot(fig6)

# Footer
st.markdown("---")
st.markdown("**App 2:** Enhanced with correlation analysis and multiple visualizations")

# NEW: Sidebar filters
st.sidebar.header("Filters")

# Age filter
age_range = st.sidebar.slider(
    "Age at Marriage",
    min_value=int(df['age_at_marriage'].min()),
    max_value=int(df['age_at_marriage'].max()),
    value=(20, 40)
)

# Duration filter
duration_range = st.sidebar.slider(
    "Marriage Duration (years)",
    min_value=int(df['marriage_duration_years'].min()),
    max_value=int(df['marriage_duration_years'].max()),
    value=(0, 10)
)

# Children filter
children_filter = st.sidebar.multiselect(
    "Number of Children",
    options=sorted(df['num_children'].unique()),
    default=sorted(df['num_children'].unique())
)

# Infidelity filter
infidelity_filter = st.sidebar.radio(
    "Infidelity Status",
    options=['All', 'No Infidelity', 'Infidelity Occurred']
)

# Apply filters
filtered_df = df[
    (df['age_at_marriage'] >= age_range[0]) & 
    (df['age_at_marriage'] <= age_range[1]) &
    (df['marriage_duration_years'] >= duration_range[0]) & 
    (df['marriage_duration_years'] <= duration_range[1]) &
    (df['num_children'].isin(children_filter))
]

if infidelity_filter == 'No Infidelity':
    filtered_df = filtered_df[filtered_df['infidelity_occurred'] == 0]
elif infidelity_filter == 'Infidelity Occurred':
    filtered_df = filtered_df[filtered_df['infidelity_occurred'] == 1]

# Basic info with filtered data
st.header("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Filtered Couples", len(filtered_df))
with col2:
    st.metric("Total Features", len(df.columns))
with col3:
    st.metric("Filtered Divorce Rate", f"{filtered_df['divorced'].mean():.1%}")
with col4:
    pct_shown = len(filtered_df) / len(df) * 100
    st.metric("Data Shown", f"{pct_shown:.1f}%")

# Show filtered data
st.subheader("Filtered Data")
st.dataframe(filtered_df.head())

# Basic statistics
with st.expander("Statistics for Filtered Data"):
    st.write(filtered_df.describe())

# Visualizations with filtered data
st.header("Visualizations (Filtered Data)")

# Create two columns for charts
col1, col2 = st.columns(2)

with col1:
    # 1. Divorce distribution
    st.subheader("Divorce Distribution")
    fig1, ax1 = plt.subplots()
    divorce_counts = filtered_df['divorced'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    if len(divorce_counts) > 0:
        ax1.pie(divorce_counts, labels=['Not Divorced', 'Divorced'], 
                autopct='%1.1f%%', colors=colors)
    st.pyplot(fig1)

with col2:
    # 2. Age at marriage distribution
    st.subheader("Age at Marriage Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(filtered_df['age_at_marriage'], bins=20, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Count')
    st.pyplot(fig2)

# Detailed Analysis
st.header("Detailed Analysis (Filtered)")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Divorce Rate by Infidelity")
    fig3, ax3 = plt.subplots()
    if len(filtered_df) > 0:
        divorce_by_infidelity = filtered_df.groupby('infidelity_occurred')['divorced'].mean()
        if len(divorce_by_infidelity) > 0:
            labels = ['No Infidelity', 'Infidelity'][:len(divorce_by_infidelity)]
            ax3.bar(labels, divorce_by_infidelity.values, color=['green', 'red'][:len(divorce_by_infidelity)])
            ax3.set_ylabel('Divorce Rate')
            ax3.set_ylim(0, 1)
    st.pyplot(fig3)

with col4:
    st.subheader("Divorce Rate by Children")
    fig4, ax4 = plt.subplots()
    if len(filtered_df) > 0:
        divorce_by_children = filtered_df.groupby('num_children')['divorced'].mean()


