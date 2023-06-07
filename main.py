import pandas as pd
import streamlit as st
import plotly.express as px

# Load the data
data_path = r'D:\Hopcharge\Session_1_7_May23 - Session_1_7_May23.xlsx'
df = pd.read_excel(data_path)

# Convert 'Actual Date' column to datetime
df['Actual Date'] = pd.to_datetime(df['Actual Date'], errors='coerce')

# Filter data by date range
st.sidebar.title('Date Range Selection')
min_date = df['Actual Date'].min().date()
max_date = df['Actual Date'].max().date()
start_date = st.sidebar.date_input('Start Date', min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input('End Date', min_value=min_date, max_value=max_date, value=max_date)

# Convert start and end dates to Pandas datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

filtered_df = df[(df['Actual Date'] >= start_date) & (df['Actual Date'] <= end_date)]

# Create the Streamlit application
st.title("Hopcharge Executive Dashboard")

# Create the pie chart
fig = px.pie(filtered_df, values='T-15 min KPI Flag', names='Actual Date', title='T-15 KPI (Overall)')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(showlegend=False)

st.plotly_chart(fig)
