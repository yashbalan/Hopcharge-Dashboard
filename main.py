pip install streamlit
pip install plotly.express
pip install go
pip install plotly

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set the page config to wide layout
st.set_page_config(layout="wide")

def hop():


    st.markdown(
        """
        <style>
        .stApp {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .logo-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-bottom: 10px;
        }
        .logo-container img {
            width: 250px;
            height: 150px;
            margin-bottom: 1px;
        }
        .dashboard-title {
            text-align: center;
            font-size: 42px;
            color: blue;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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

    # Calculate record count by EPOD Name and T-15 min KPI Flag
    record_count_df = filtered_df.groupby(['EPOD Name', 'T-15 min KPI Flag']).size().reset_index(name='Record Count')

    # Calculate record count by Customer Location City and T-15 min KPI Flag
    city_count_df = filtered_df.groupby(['Customer Location City', 'T-15 min KPI Flag']).size().reset_index(
        name='Record Count')

    # Sort the dataframes by 'Record Count'
    record_count_df = record_count_df.sort_values(by='Record Count')
    city_count_df = city_count_df.sort_values(by='Record Count')

    # Calculate Start SoC % and End SoC %
    start_soc_stats = filtered_df.groupby('EPOD Name')['Actual SoC_Start'].agg(['max', 'min', 'mean'])
    end_soc_stats = filtered_df.groupby('EPOD Name')['Actual Soc_End'].agg(['max', 'min', 'mean'])
    start_soc_stats = start_soc_stats.sort_values(by='EPOD Name')
    end_soc_stats = end_soc_stats.sort_values(by='EPOD Name')

    # Create the Streamlit application
    st.markdown(
        "<div class='logo-container'; width: 50px; height: 50px; margin-right: 10px><img src='data:image/png;base64,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';base64,iVBORw0KGgoAAAANSUhEUgAA...'></div>",
        unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: blue'>Hopcharge Executive Dashboard</h1>",
                unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; font-size: 36px'>T-15 KPI Dashboard</h1>", unsafe_allow_html=True)

    # Create two columns for the layout
    col1, col2 = st.columns([1, 2])

    # Create the pie chart in the first column
    with col1:
        st.subheader("T-15 KPI (Overall)")
        fig_pie = px.pie(filtered_df, values='T-15 min KPI Flag', names='Actual Date', title='T-15 KPI (Overall)')
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Create the first vertical stack graph in the second column (T-15 KPI Pod Wise)
    with col2:
        st.subheader("T-15 KPI (Pod Wise)")
        fig_stack1 = px.bar(record_count_df, x='EPOD Name', y='Record Count', color='T-15 min KPI Flag',
                            barmode='stack', title='T-15 KPI (Pod Wise)')
        fig_stack1.update_layout(xaxis={'categoryorder': 'total descending'},
                                 yaxis={'categoryorder': 'total ascending'})
        fig_stack1.update_layout(height=600)
        st.plotly_chart(fig_stack1, use_container_width=True)

    # Create the second vertical stack graph in the third column (T-15 KPI HSZ Wise)
    with col2:
        st.subheader("T-15 KPI (HSZ Wise)")
        fig_stack2 = px.bar(city_count_df, x='Customer Location City', y='Record Count', color='T-15 min KPI Flag',
                            barmode='stack', title='T-15 KPI (HSZ Wise)')
        fig_stack2.update_layout(xaxis={'categoryorder': 'total descending'},
                                 yaxis={'categoryorder': 'total ascending'})
        fig_stack2.update_layout(height=600)
        st.plotly_chart(fig_stack2, use_container_width=True)
def about():
    st.title("State of Charge Analysis")
    st.write("This is the SoC Analysis!")

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

    # Calculate Start SoC % and End SoC %
    start_soc_stats = filtered_df.groupby('EPOD Name')['Actual SoC_Start'].agg(['max', 'min', 'mean'])
    end_soc_stats = filtered_df.groupby('EPOD Name')['Actual Soc_End'].agg(['max', 'min', 'mean'])
    start_soc_stats = start_soc_stats.sort_values(by='EPOD Name')
    end_soc_stats = end_soc_stats.sort_values(by='EPOD Name')

    st.markdown("<h1 style='text-align: center; font-size: 36px'>E-POD's SoC Status</h1>", unsafe_allow_html=True)

    # Create a button to toggle between Start SoC % and End SoC %
    soc_selection = st.radio("Select SoC Calculation", ("Start SoC %", "End SoC %"))

    if soc_selection == "Start SoC %":
        soc_stats = start_soc_stats
        title = "Start SoC % Statistics"
    else:
        soc_stats = end_soc_stats
        title = "End SoC % Statistics"

    # Create a section for SoC % statistics
    st.subheader(title)
    fig = go.Figure(data=[
        go.Bar(name='Max', x=soc_stats.index, y=soc_stats['max']),
        go.Bar(name='Min', x=soc_stats.index, y=soc_stats['min']),
        go.Bar(name='Avg', x=soc_stats.index, y=soc_stats['mean'])
    ])

    # Add values as text on the bars
    for trace in fig.data:
        fig.update_traces(text=trace.y, textposition='inside')

    fig.update_layout(barmode='stack')
    fig.update_layout(uniformtext=dict(mode='hide', minsize=10))
    fig.update_layout(showlegend=True)
    fig.update_layout(legend=dict(x=1, y=1))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(font_color='white')

    fig.update_layout(barmode='stack')
    st.plotly_chart(fig, use_container_width=True)

    # Calculate Start SoC % and End SoC % within the filtered date range
    start_soc_max = filtered_df['Actual SoC_Start'].max()
    start_soc_min = filtered_df['Actual SoC_Start'].min()
    start_soc_avg = filtered_df['Actual SoC_Start'].mean()

    end_soc_max = filtered_df['Actual Soc_End'].max()
    end_soc_min = filtered_df['Actual Soc_End'].min()
    end_soc_avg = filtered_df['Actual Soc_End'].mean()

    # Write Start SoC % and End SoC % statistics for each EPOD Name
    st.subheader("Start SoC % Statistics by EPOD Name")
    st.write(start_soc_stats)

    st.subheader("End SoC % Statistics by EPOD Name")
    st.write(end_soc_stats)

    # Create the Streamlit application
    st.markdown("<h1 style='text-align: center; font-size: 36px'>State of Charge Analysis</h1>", unsafe_allow_html=True)

    # Create a section for Start SoC % statistics
    st.subheader("Start SoC % Statistics")
    st.write("Max Start SoC %:", start_soc_max)
    st.write("Min Start SoC %:", start_soc_min)
    st.write("Avg Start SoC %:", start_soc_avg)

    # Create a section for End SoC % statistics
    st.subheader("End SoC % Statistics")
    st.write("Max End SoC %:", end_soc_max)
    st.write("Min End SoC %:", end_soc_min)
    st.write("Avg End SoC %:", end_soc_avg)



def contact():
    st.title("Calculations")
    st.write("Personalised Calculations!")

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

    # Get unique EPOD Names
    epod_names = filtered_df['EPOD Name'].unique()

    st.markdown("<h1 style='text-align: center; font-size: 36px'>KM Per EPOD</h1>", unsafe_allow_html=True)
    # Dropdown for selecting EPOD Name
    selected_epod = st.selectbox("Select EPOD Name", epod_names)

    # Filter the data based on the selected EPOD Name
    epod_df = filtered_df[filtered_df['EPOD Name'] == selected_epod]

    # Convert 'KM Travelled for Session' column to numeric
    epod_df['KM Travelled for Session'] = pd.to_numeric(epod_df['KM Travelled for Session'], errors='coerce')

    # Calculate the KM Travelled for Session by EPOD Name
    km_traveled = epod_df['KM Travelled for Session'].sum()

    # Display the KM Travelled for Session by EPOD Name
    st.subheader("KM Travelled for Session")

    st.write(f"Total KM Travelled for EPOD '{selected_epod}': {km_traveled}")

    st.markdown("<h1 style='text-align: center; font-size: 36px'>KWH per Session</h1>", unsafe_allow_html=True)

    # Get unique EPOD Names
    epod_names = filtered_df['EPOD Name'].unique()

    # Generate a unique key for the selectbox widget
    selectbox_key = "epod_selectbox"

    # Dropdown for selecting EPOD Name
    selected_epod = st.selectbox("Select EPOD Name", epod_names, key=selectbox_key)

    # Filter the data based on the selected EPOD Name
    epod_df = filtered_df[filtered_df['EPOD Name'] == selected_epod]

    # Convert 'KWH Pumped Per Session' column to numeric
    epod_df['KWH Pumped Per Session'] = pd.to_numeric(epod_df['KWH Pumped Per Session'], errors='coerce')

    # Calculate the total KWH Pumped per Session by EPOD Name
    total_kwh_pumped = epod_df['KWH Pumped Per Session'].sum()

    # Display the total KWH Pumped per Session by EPOD Name
    st.subheader("KWH Pumped Per Session")
    st.write(f"Total KWH Pumped for EPOD '{selected_epod}': {total_kwh_pumped}")

    st.markdown("<h1 style='text-align: center; font-size: 36px'>Sessions Per Operator</h1>", unsafe_allow_html=True)

    # Get unique names in the filtered data
    unique_names = filtered_df['Actual OPERATOR NAME'].unique()

    # Dropdown for selecting a name
    selected_name = st.selectbox("Select the Operator", unique_names)

    # Count the occurrences of the selected name in the 'Actual OPERATOR NAME' column
    selected_count = filtered_df[filtered_df['Actual OPERATOR NAME'] == selected_name].shape[0]

    # Display the count for the selected name
    st.subheader("Sessions per Operator")
    st.write(f"Sessions per Operator within the selected date range: {selected_count}")

    st.markdown("<h1 style='text-align: center; font-size: 36px'>Average Duration Per EPOD</h1>",
                unsafe_allow_html=True)

    # Get unique EPOD names in the filtered data
    unique_epods = filtered_df['EPOD Name'].unique()

    # Dropdown for selecting an EPOD Name
    selected_epod = st.selectbox("Select an EPOD Name", unique_epods)

    # Filter the data for the selected EPOD Name
    epod_data = filtered_df[filtered_df['EPOD Name'] == selected_epod]

    # Calculate the duration per session in minutes
    start_time = pd.to_datetime(epod_data['Actual Session Start Time'].astype(str))
    end_time = pd.to_datetime(epod_data['Actual Session End Time'].astype(str))

    # Calculate the duration per session
    epod_data['Duration'] = abs((end_time - start_time).dt.total_seconds() / 60)

    # Calculate the average duration per session
    average_duration = epod_data['Duration'].mean()

    # Display the average duration per session for the selected EPOD Name
    st.subheader("Average Duration per Session")
    st.write(
        f"Average duration per session for selected EPOD Name within the selected date range: {average_duration:.2f} minutes")



def main():
    # Create a dropdown menu to select the page
    pages = {
        "T-15 KPI": hop,
        "State of Charge": about,
        "Pattern Analysis": contact
    }
    page_selection = st.sidebar.selectbox("Let's Analyze Data", list(pages.keys()))

    # Call the selected page function
    pages[page_selection]()

if __name__ == "__main__":
    main()
