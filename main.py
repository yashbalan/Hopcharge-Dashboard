import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
import re
import matplotlib.pyplot as plt
import time

df1 = pd.DataFrame(pd.read_csv(
    'Ops_Session_Data.csv', encoding='latin1'))
df2 = pd.DataFrame(pd.read_csv(
    'past_bookings_May23.csv', encoding='latin1'))

df3 = pd.DataFrame(pd.read_csv(
    'possible_subscribers_May23.csv', encoding='latin1'))

df2["Customer Name"] = df2["firstName"].str.cat(df2["lastName"], sep=" ")

df1 = df1.dropna(subset=["uid"])


def subtract_12_from_pm_time(time_str):
    time_pattern = r"(\d{1,2}):(\d{2})\s?(AM|PM|pm|am)"
    match = re.match(time_pattern, time_str)

    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        am_pm = match.group(3)

        if am_pm.lower() == "pm" and hour < 12:
            hour += 12

        return "{:02d}:{:02d}".format(hour, minute)

    return time_str


df1['Booking Session time'] = df1['Booking Session time'].apply(
    subtract_12_from_pm_time)

df2['updated'] = pd.to_datetime(df2['updated'], format='%d/%m/%Y, %H:%M:%S')
df2['fromTime'] = pd.to_datetime(df2['fromTime'], format='%Y-%m-%dT%H:%M')

# Calculate the time difference in hours between 'fromTime' and 'updated'
time_diff = df2['fromTime'] - df2['updated']
time_diff_hours = time_diff / timedelta(hours=1)

# Create the 'cancelledPenalty' column based on the conditions
df2['cancelledPenalty'] = 0
df2.loc[(df2['canceled'] == True) & (
    time_diff_hours < 2), 'cancelledPenalty'] = 1


def calculate_t_minus_15(row):
    booking_time_str = row['Booking Session time']
    arrival_time_str = row['E-pod Arrival Time @ Session location']

    booking_time = datetime.strptime(booking_time_str, "%H:%M")

    arrival_time = datetime.strptime(arrival_time_str, "%I:%M:%S %p")

    if arrival_time_str.endswith("PM") and booking_time.hour < 12:
        booking_time = booking_time + timedelta(hours=12)

    time_diff = booking_time - arrival_time

    if time_diff >= timedelta(minutes=15):
        return 1
    elif time_diff < timedelta(seconds=0):
        return 2
    else:
        return 0


df1['t-15_kpi'] = df1.apply(calculate_t_minus_15, axis=1)
merged_df = pd.merge(df2, df1, on=["uid"])


df3.set_index('uid', inplace=True)


merged_df = merged_df.join(df3['type'], on='location.user_id_x')

grouped_df = merged_df.groupby("uid").agg(
    {"Actual SoC_Start": "min", "Actual Soc_End": "max"}).reset_index()


grouped_df = grouped_df.rename(
    columns={"Actual SoC_Start": "Actual SoC_Start", "Actual Soc_End": "Actual Soc_End"})

merged_df = pd.merge(merged_df, grouped_df, on="uid", how="left")

merged_df = merged_df.drop(["Actual SoC_Start_x", "Actual Soc_End_x"], axis=1)

merged_df = merged_df.rename(columns={
                             "Actual SoC_Start_y": "Actual SoC_Start", "Actual Soc_End_y": "Actual Soc_End"})

merged_df = merged_df.drop_duplicates(subset="uid", keep="first")


merged_df = merged_df.reset_index(drop=True)

start_time = pd.to_datetime(merged_df['optChargeStartTime_x'].astype(str))

end_time = pd.to_datetime(merged_df['optChargeEndTime_x'].astype(str))

merged_df['Duration'] = abs((end_time - start_time).dt.total_seconds() / 60)
merged_df['Duration'] = merged_df['Duration'].round(2)

merged_df['Day'] = pd.to_datetime(merged_df['Actual Date']).dt.day_name()


requiredColumns = ['uid', 'Actual Date', 'Customer Name_x', 'EPOD Name', 'Actual OPERATOR NAME', 'Duration', 'Day',
                   'E-pod Arrival Time @ Session location', 'Actual SoC_Start', 'Actual Soc_End', 'Booking Session time', 'Customer Location City', 'canceled_x', 'cancelledPenalty', 't-15_kpi', 'type', 'KM Travelled for Session', 'KWH Pumped Per Session']
merged_df = merged_df[requiredColumns]
merged_df["Actual SoC_Start"] = merged_df["Actual SoC_Start"].str.rstrip("%")
merged_df["Actual Soc_End"] = merged_df["Actual Soc_End"].str.rstrip("%")
merged_df["KM Travelled for Session"] = merged_df["KM Travelled for Session"].str.replace(
    r'[a-zA-Z]', '', regex=True)
merged_df['EPOD Name'] = merged_df['EPOD Name'].str.extract(
    r'^(.*?)\s+\(.*\)$')[0]
merged_df['EPOD Name'] = merged_df['EPOD Name'].fillna('EPOD006')

merged_df.to_csv('mergeddata.csv')
st.set_page_config(page_title="Hopcharge Dashboard",
                   page_icon=":bar_chart:", layout="wide")

st.markdown(
    """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
        padding_top=1, padding_bottom=1
    ),
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    body {
        overflow-x: hidden;
        overflow-yr: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)


df = merged_df

tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive Dashboard", "Charge Pattern Insights", "EPod Stats", "Operator Stats"])


with tab1:
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        df['Actual Date'] = pd.to_datetime(df['Actual Date'], errors='coerce')
        min_date = df['Actual Date'].min().date()
        max_date = df['Actual Date'].max().date()
        start_date = st.date_input(
            'Start Date', min_value=min_date, max_value=max_date, value=min_date, key="ex-date-start")

    with col1:
        end_date = st.date_input(
            'End Date', min_value=min_date, max_value=max_date, value=max_date, key="ex-date-end")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = df[(df['Actual Date'] >= start_date)
                     & (df['Actual Date'] <= end_date)]
    filtered_df['Actual SoC_Start'] = pd.to_numeric(
        filtered_df['Actual SoC_Start'], errors='coerce')
    filtered_df['Actual Soc_End'] = pd.to_numeric(
        filtered_df['Actual Soc_End'], errors='coerce')

    record_count_df = filtered_df.groupby(
        ['EPOD Name', 't-15_kpi']).size().reset_index(name='Record Count')

    city_count_df = filtered_df.groupby(['Customer Location City', 't-15_kpi']).size().reset_index(
        name='Record Count')
    record_count_df = record_count_df.sort_values(by='Record Count')
    city_count_df = city_count_df.sort_values(by='Record Count')
    start_soc_stats = filtered_df.groupby('EPOD Name')['Actual SoC_Start'].agg([
        'max', 'min', 'mean', 'median'])

    end_soc_stats = filtered_df.groupby('EPOD Name')['Actual Soc_End'].agg([
        'max', 'min', 'mean', 'median'])
    start_soc_stats = start_soc_stats.sort_values(by='EPOD Name')
    end_soc_stats = end_soc_stats.sort_values(by='EPOD Name')
    kpi_flag_data = filtered_df['t-15_kpi']

    # Calculate counts
    before_time_count = (kpi_flag_data == 1).sum()
    on_time_count = (kpi_flag_data == 0).sum()
    delay_count = (kpi_flag_data == 2).sum()

    # Calculate percentages
    total_count = before_time_count + delay_count + on_time_count
    before_time_percentage = (before_time_count / total_count) * 100
    on_time_percentage = (on_time_count / total_count) * 100
    delay_percentage = (delay_count / total_count) * 100
    labels = ['Before Time', 'Delay']

    start_soc_avg = start_soc_stats['mean'].values[0]
    start_soc_median = start_soc_stats['median'].values[0]

    end_soc_avg = end_soc_stats['mean'].values[0]
    end_soc_median = end_soc_stats['median'].values[0]

    col2.metric("Before Time", f"{before_time_percentage.round(2)}%")
    col3.metric("Delay", f"{delay_percentage.round(2)}%")
    col4.metric("On Time", f"{on_time_percentage.round(2)}%")
    col5.metric("Avg Start SoC", f"{start_soc_avg.round(2)}%")
    col6.metric("Avg End SoC", f"{end_soc_avg.round(2)}%")

    total_sessions = filtered_df['t-15_kpi'].count()
    fig = go.Figure(data=[go.Pie(labels=['Before Time', 'Delay', 'On Time'],
                                 values=[before_time_count,
                                         delay_count, on_time_count],
                                 hole=0.6,
                                 sort=False,
                                 textinfo='label+percent+value',
                                 textposition='outside',
                                 marker=dict(colors=['green', 'red', 'blue']))])

    fig.add_annotation(text='Total Sessions',
                       x=0.5, y=0.5, font_size=15, showarrow=False)

    fig.add_annotation(text=str(total_sessions),
                       x=0.5, y=0.45, font_size=15, showarrow=False)
    fig.update_layout(
        title='T-15 KPI (Overall)',
        showlegend=False,
        height=400,
        width=380
    )

    with col2:
        st.plotly_chart(fig, use_container_width=False)

    allowed_cities = ["Gurgaon", "Noida", "Delhi", "Ghaziabad", "Faridabad"]
    city_count_df = city_count_df[city_count_df['Customer Location City'].isin(
        allowed_cities)]

    # Create a new figure
    fig_group = go.Figure()

    color_mapping = {0: 'red', 1: 'green', 2: 'blue'}
    city_count_df['Percentage'] = city_count_df['Record Count'] / \
        city_count_df.groupby('Customer Location City')[
        'Record Count'].transform('sum') * 100

# Create the figure
    fig_group = go.Figure()

    for flag in city_count_df['t-15_kpi'].unique():
        # Filter the dataframe for the specific flag value
        df_flag = city_count_df[city_count_df['t-15_kpi'] == flag]

    # Add a trace for the specific flag with the corresponding color
        fig_group.add_trace(go.Bar(
            x=df_flag['Customer Location City'],
            y=df_flag['Percentage'],
            name=str(flag),
            text=df_flag['Percentage'].round(0).astype(str) + '%',
            marker=dict(color=color_mapping[flag]),
            textposition='auto'
        ))

    # Set the layout of the grouped bar graph
    fig_group.update_layout(
        barmode='group',
        title='T-15 KPI (HSZ Wise)',
        xaxis={'categoryorder': 'total descending'},
        yaxis={'tickformat': '.2f', 'title': 'Percentage'},
        height=400,
        width=570,
        margin=dict(t=50, b=50, l=50, r=50),
        showlegend=False
    )

    with col4:
        st.plotly_chart(fig_group)

    filtered_city_count_df = city_count_df[city_count_df['t-15_kpi'] == 1]

    max_record_count_city = filtered_city_count_df.loc[
        filtered_city_count_df['Record Count'].idxmax(), 'Customer Location City']
    min_record_count_city = filtered_city_count_df.loc[
        filtered_city_count_df['Record Count'].idxmin(), 'Customer Location City']

    col7.metric("City with Maximum Sessions", max_record_count_city)
    col7.metric("City with Minimum Sessions", min_record_count_city)

    start_soc_max = start_soc_stats['max'].values.max()

    start_soc_min = start_soc_stats['min'].values.min()

    start_soc_avg = start_soc_stats['mean'].values.mean()
    start_soc_median = np.median(start_soc_stats['median'].values)

    gauge_range = [0, 100]

    start_soc_max_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=start_soc_max,
        title={'text': "Max Start SoC %", 'font': {'size': 15}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': gauge_range}},
    ))
    start_soc_max_gauge.update_layout(width=150, height=250)

    start_soc_min_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=start_soc_min,
        title={'text': "Min Start SoC %", 'font': {'size': 15}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': gauge_range}}
    ))
    start_soc_min_gauge.update_layout(width=150, height=250)

    start_soc_avg_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=start_soc_avg,
        title={'text': "Avg Start SoC %", 'font': {'size': 15}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': gauge_range}}
    ))
    start_soc_avg_gauge.update_layout(width=150, height=250)

    start_soc_median_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=start_soc_median,
        title={'text': "Median Start SoC %", 'font': {'size': 15}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': gauge_range}}
    ))
    start_soc_median_gauge.update_layout(width=150, height=250)
    with col3:
        for i in range(1, 27):
            st.write("\n")
        st.write("#### Start SoC Stats")

    with col6:
        for i in range(1, 27):
            st.write("\n")
        st.write("#### End SoC Stats")
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col1:

        st.plotly_chart(start_soc_min_gauge)

    with col2:

        st.plotly_chart(start_soc_max_gauge)

    with col3:

        st.plotly_chart(start_soc_avg_gauge)

    with col4:

        st.plotly_chart(start_soc_median_gauge)

    end_soc_max = end_soc_stats['max'].values.max()
    end_soc_min = end_soc_stats['min'].values.min()
    end_soc_avg = end_soc_stats['mean'].values.mean()
    end_soc_median = np.median(end_soc_stats['median'].values)

    # Create gauge chart for maximum End SoC
    end_soc_max_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=end_soc_max,
        title={'text': "Max End SoC %", 'font': {'size': 15}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': gauge_range}}
    ))

    # Create gauge chart for minimum End SoC
    end_soc_min_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=end_soc_min,
        title={'text': "Min End SoC %", 'font': {'size': 15}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': gauge_range}}
    ))
    end_soc_max_gauge.update_layout(width=150, height=250)
    end_soc_min_gauge.update_layout(width=150, height=250)

    end_soc_avg_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=end_soc_avg,
        title={'text': "Avg End SoC %", 'font': {'size': 15}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': gauge_range}}
    ))
    end_soc_avg_gauge.update_layout(width=150, height=250)
    # Create gauge chart for median End SoC
    end_soc_median_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=end_soc_median,
        title={'text': "Median End SoC %", 'font': {'size': 15}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': gauge_range}}
    ))
    end_soc_median_gauge.update_layout(width=150, height=250)
    with col5:
        st.plotly_chart(end_soc_min_gauge)

    with col6:
        st.plotly_chart(end_soc_max_gauge)
    with col7:
        st.plotly_chart(end_soc_avg_gauge)
    with col8:
        st.plotly_chart(end_soc_median_gauge)

    for city in allowed_cities:
        st.subheader(city)

with tab2:

    CustomerNames = df['Customer Name_x'].unique()
    SubscriptionNames = df['type'].unique()
    HSZs = df['Customer Location City'].unique()
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        df['Actual Date'] = pd.to_datetime(df['Actual Date'], errors='coerce')
        min_date = df['Actual Date'].min().date()
        max_date = df['Actual Date'].max().date()
        start_date = st.date_input(
            'Start Date', min_value=min_date, max_value=max_date, value=min_date, key="cpi-date-start")

    with col2:
        end_date = st.date_input(
            'End Date', min_value=min_date, max_value=max_date, value=max_date, key="cpi-date-end")
    with col5:

        Name = st.multiselect(label='Select The Customers',
                              options=['All'] + CustomerNames.tolist(),
                              default='All')

    with col4:
        HSZ_Filter = st.multiselect(label='Select HSZ',
                                    options=['All']+HSZs.tolist(),
                                    default='All')
    with col3:
        Sub_filter = st.multiselect(label='Select Subscription',
                                    options=['All'] +
                                    SubscriptionNames.tolist(),
                                    default='All')

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_data = df[(df['Actual Date'] >= start_date)
                       & (df['Actual Date'] <= end_date)]

    if 'All' in Name:
        Name = CustomerNames

    if 'All' in Sub_filter:
        Sub_filter = SubscriptionNames

    if 'All' in HSZ_Filter:
        HSZ_Filter = HSZs
    filtered_data = filtered_data[
        (filtered_data['Customer Name_x'].isin(Name)) &
        (filtered_data['type'].isin(Sub_filter)) &
        (filtered_data['Customer Location City'].isin(
            HSZ_Filter))
    ]

    def generate_multiline_plot(data):
        fig = go.Figure()
        color_map = {0: 'blue', 1: 'green', 2: 'red'}
        names = {0: "T-15 Not Fulfilled", 1: "T-15 Fulfilled", 2: "Delayed"}
        for kpi_flag in data['t-15_kpi'].unique():
            subset = data[data['t-15_kpi'] == kpi_flag]
            fig.add_trace(go.Scatter(x=subset['Day'], y=subset['count'], mode='lines+markers',
                                     name=names[kpi_flag], line_color=color_map[kpi_flag]))

        total_count = data.groupby('Day')['count'].sum().reset_index()
        fig.add_trace(go.Scatter(x=total_count['Day'], y=total_count['count'], mode='lines+markers',
                                 name='Total Count', line_color='yellow'))

        fig.update_layout(
            xaxis_title='Day', yaxis_title='Count', legend=dict(x=0, y=1.2, orientation='h'))
        fig.update_yaxes(title='Count', range=[
                         0, total_count['count'].max() * 1.2])
        for trace in fig.data:
            fig.add_trace(go.Scatter(
                x=trace.x,
                y=trace.y,
                mode='text',
                text=trace.y,
                textposition='top center',
                showlegend=False
            ))
        fig.update_layout(width=500, height=380)
        return fig

    day_order = ['Monday', 'Tuesday', 'Wednesday',
                 'Thursday', 'Friday', 'Saturday', 'Sunday']
    filtered_data['Day'] = pd.Categorical(
        filtered_data['Day'], categories=day_order, ordered=True)

    daily_count = filtered_data.groupby(
        ['Day', 't-15_kpi']).size().reset_index(name='count')
    maxday = filtered_data.groupby(['Day']).size().reset_index(name='count')

    maxday['count'] = maxday['count'].astype(int)

    max_count_index = maxday['count'].idxmax()

    max_count_day = maxday.loc[max_count_index, 'Day']

    minday = filtered_data.groupby(['Day']).size().reset_index(name='count')

    minday['count'] = minday['count'].astype(int)

    min_count_index = minday['count'].idxmin()

    min_count_day = minday.loc[min_count_index, 'Day']

    with col7:
        for i in range(1, 10):
            st.write("\n")
        st.markdown("Most Sessions on Day")
        st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                    max_count_day+"</span>", unsafe_allow_html=True)
    with col7:
        st.markdown("Min Sessions on Day")
        st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                    min_count_day+"</span>", unsafe_allow_html=True)
    multiline_plot = generate_multiline_plot(daily_count)

    with col4:

        st.plotly_chart(multiline_plot)

    def count_t15_kpi(df):
        try:
            return df.groupby(
                ['t-15_kpi']).size()['1']
        except KeyError:
            return 0

    def count_sessions(df):
        return df.shape[0]

    def count_cancelled(df):
        try:
            return df.groupby(['canceled_x']).size()[True]
        except KeyError:
            return 0

    def count_cancelledPenalty(df):
        try:
            return df.groupby(['cancelledPenalty']).size()[1]
        except KeyError:
            return 0

    total_sessions = count_sessions(filtered_data)
    cancelled_sessions = count_cancelled(filtered_data)
    cancelled_sessions_with_penalty = count_cancelledPenalty(filtered_data)
    labels = ['Total Sessions', 'Cancelled Sessions',
              'Cancelled with Penalty']
    values = [total_sessions-cancelled_sessions-cancelled_sessions_with_penalty, cancelled_sessions,
              cancelled_sessions_with_penalty]

    colors = ['blue', 'orange', 'red']

    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=0.7, textinfo='label+value+percent', marker=dict(colors=colors))])
    fig.update_layout(
        showlegend=True, width=500)
    fig.add_annotation(
        text=f"Overall Sessions: {total_sessions}", x=0.5, y=0.5, font_size=15, showarrow=False)
    fig.update_layout(width=500, height=400)
    with col1:
        st.plotly_chart(fig)

    def generate_multiline_plot(data):
        fig = go.Figure()
        color_map = {0: 'blue', 1: 'green', 2: 'red'}
        names = {0: "T-15 Not Fulfilled", 1: "T-15 Fulfilled", 2: "Delayed"}

        for kpi_flag in data['t-15_kpi'].unique():
            subset = data[data['t-15_kpi'] == kpi_flag]
            fig.add_trace(go.Scatter(x=subset['Booking Session time'], y=subset['count'], mode='lines+markers',
                                     name=names[kpi_flag], line_color=color_map[kpi_flag]))

        total_count = data.groupby('Booking Session time')[
            'count'].sum().reset_index()
        fig.add_trace(go.Scatter(x=total_count['Booking Session time'], y=total_count['count'], mode='lines+markers',
                                 name='Total Count', line_color='yellow'))

        fig.update_layout(xaxis_title='Booking Session Time',
                          yaxis_title='Count', legend=dict(x=0, y=1.2, orientation='h'))
        fig.update_yaxes(title='Count', range=[
                         0, total_count['count'].max() * 1.2])

        for trace in fig.data:
            fig.add_trace(go.Scatter(
                x=trace.x,
                y=trace.y,
                mode='text',
                text=trace.y,
                textposition='top center',
                showlegend=False
            ))
        fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(
            range(24)), ticktext=list(range(24))))
        fig.update_layout(width=1100, height=350)

        return fig

    filtered_data['Booking Session time'] = pd.to_datetime(
        filtered_data['Booking Session time'], format='%H:%M').dt.hour

    daily_count = filtered_data.groupby(
        ['Booking Session time', 't-15_kpi']).size().reset_index(name='count')
    maxmindf = filtered_data.groupby(
        ['Booking Session time']).size().reset_index(name='count')
    max_count_index = maxmindf['count'].idxmax()

# Retrieve the time with the maximum count
    max_count_time = maxmindf.loc[max_count_index, 'Booking Session time']

# Find the index of the row with the minimum count
    min_count_index = maxmindf['count'].idxmin()

# Retrieve the time with the minimum count
    min_count_time = maxmindf.loc[min_count_index, 'Booking Session time']
    with col7:
        for i in range(1, 18):
            st.write("\n")
        st.markdown("Max Sessions at Time")
        st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                    str(max_count_time)+"</span>", unsafe_allow_html=True)

    with col7:
        st.markdown("Min Sessions at Time")
        st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                    str(min_count_time)+"</span>", unsafe_allow_html=True)

    multiline_plot = generate_multiline_plot(daily_count)

    with col1:
        st.plotly_chart(multiline_plot)

with tab3:
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        df['Actual Date'] = pd.to_datetime(df['Actual Date'], errors='coerce')
        min_date = df['Actual Date'].min().date()
        max_date = df['Actual Date'].max().date()
        start_date = st.date_input(
            'Start Date', min_value=min_date, max_value=max_date, value=min_date, key="epod-date-start")
    with col2:
        end_date = st.date_input(
            'End Date', min_value=min_date, max_value=max_date, value=max_date, key="epod-date-end")
    df['EPOD Name'] = df['EPOD Name'].str.replace('-', '')

    epods = df['EPOD Name'].unique()
    with col3:
        EPod = st.multiselect(label='Select The EPOD',
                              options=['All'] + epods.tolist(),
                              default='All')
    with col1:
        st.markdown(":large_green_square: T-15 fulfilled")

    with col2:
        st.markdown(":large_blue_square: T-15 Not fulfilled")
    with col3:
        st.markdown(":large_red_square: Delay")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_data = df[(df['Actual Date'] >= start_date)
                       & (df['Actual Date'] <= end_date)]
    if 'All' in EPod:
        EPod = epods

    filtered_data = filtered_data[
        (filtered_data['EPOD Name'].isin(EPod))]

    record_count_df = filtered_data.groupby(
        ['EPOD Name', 't-15_kpi']).size().reset_index(name='Record Count')
    color_mapping = {0: 'blue', 1: 'green', 2: 'red'}
    record_count_df['Color'] = record_count_df['t-15_kpi'].map(color_mapping)

    record_count_df = record_count_df.sort_values('EPOD Name')
    y_axis_range = [0, record_count_df['Record Count'].max() * 1.2]

    total_duration = filtered_data.groupby(
        'EPOD Name')['Duration'].sum().reset_index()

    average_duration = filtered_data.groupby(
        'EPOD Name')['Duration'].mean().reset_index().round(1)

    avgdur = average_duration['Duration'].mean().round(2)

    with col5:
        st.markdown("Average Duration/Session")
        st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                    str(avgdur)+"</span>", unsafe_allow_html=True)

    filtered_data['KM Travelled for Session'] = filtered_data['KM Travelled for Session'].replace(
        '', np.nan)
    filtered_data['KM Travelled for Session'] = filtered_data['KM Travelled for Session'].astype(
        float)
    average_kms = filtered_data.groupby(
        'EPOD Name')['KM Travelled for Session'].mean().reset_index().round(1)
    avgkm = average_kms['KM Travelled for Session'].mean().round(2)
    with col4:
        st.markdown("Average Kms/EPod")
        st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                    str(avgkm)+"</span>", unsafe_allow_html=True)

    filtered_data['KWH Pumped Per Session'] = filtered_data['KWH Pumped Per Session'].replace(
        '', np.nan)

    filtered_data = filtered_data[filtered_data['KWH Pumped Per Session'] != '#VALUE!']

    filtered_data['KWH Pumped Per Session'] = filtered_data['KWH Pumped Per Session'].astype(
        float)
    filtered_data['KWH Pumped Per Session'] = filtered_data['KWH Pumped Per Session'].abs()
    average_kwh = filtered_data.groupby(
        'EPOD Name')['KWH Pumped Per Session'].mean().reset_index().round(1)
    avgkwh = average_kwh['KWH Pumped Per Session'].mean().round(2)
    with col6:
        st.markdown("Average kWh/EPod")
        st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                    str(avgkwh)+"</span>", unsafe_allow_html=True)

    fig = go.Figure()
    for color, kpi_group in record_count_df.groupby('Color'):

        fig.add_trace(go.Bar(
            x=kpi_group['EPOD Name'],
            y=kpi_group['Record Count'],
            text=kpi_group['Record Count'],
            textposition='auto',
            name=color,
            marker=dict(color=color),
            width=0.38,
            showlegend=False


        ))

    fig.update_layout(

        xaxis={'categoryorder': 'array',
               'categoryarray': record_count_df['EPOD Name'],
               'fixedrange': True},
        yaxis={'categoryorder': 'total ascending', 'range': y_axis_range,
               'fixedrange': True},
        xaxis_title='EPOD Name',
        yaxis_title='Sessions',
        height=340,
        width=600,
        title="T-15 for each EPod",
        legend=dict(
            title_font=dict(size=14),
            font=dict(size=12),
            x=0,
            y=1.1,
            orientation='h',


        )
    )

    with col1:
        st.plotly_chart(fig)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=average_duration['EPOD Name'],
        y=average_duration['Duration'],
        text=average_duration['Duration'],
        textposition='auto',
        name='Average Duration',

    ))

    fig.update_layout(
        xaxis_title='EPOD Name',
        yaxis_title='Average Duration',
        barmode='group',
        width=600,
        height=340,
        title="Avg Duration Per EPod",

    )
    with col4:
        for i in range(1, 4):
            st.write("\n")
        st.plotly_chart(fig)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=average_kms['EPOD Name'],
        y=average_kms['KM Travelled for Session'],
        text=average_kms['KM Travelled for Session'],
        textposition='auto',
        name='Average KM Travelled for Session',

    ))

    fig.update_layout(
        xaxis_title='EPOD Name',
        yaxis_title='Average KM Travelled for Session',
        barmode='group',
        width=600,
        height=340,
        title="Avg KMs Per EPod",
    )
    with col1:
        st.plotly_chart(fig)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=average_kwh['EPOD Name'],
        y=average_kwh['KWH Pumped Per Session'],
        text=average_kwh['KWH Pumped Per Session'],
        textposition='auto',
        name='Average KWH Pumped Per Session',

    ))
    fig.update_layout(
        xaxis_title='EPOD Name',
        yaxis_title='Average KWH Pumped Per Session',
        barmode='group',
        width=600,
        height=340,
        title="Avg kWh Per EPod",
    )

    with col4:
        st.plotly_chart(fig)

    for city in df['Customer Location City'].dropna().unique():
        with col1:
            st.write("\n")
            st.subheader(city)
        citydf = filtered_data[
            (filtered_data['Customer Location City'] == city)]

        record_count_df = citydf.groupby(
            ['EPOD Name', 't-15_kpi']).size().reset_index(name='Record Count')
        color_mapping = {0: 'blue', 1: 'green', 2: 'red'}
        record_count_df['Color'] = record_count_df['t-15_kpi'].map(
            color_mapping)

        record_count_df = record_count_df.sort_values('EPOD Name')
        y_axis_range = [0, record_count_df['Record Count'].max() * 1.2]

        average_kms = citydf.groupby(
            'EPOD Name')['KM Travelled for Session'].mean().reset_index().round(1)
        avgkm = average_kms['KM Travelled for Session'].mean().round(2)
        with col2:
            for i in range(1, 46):
                st.write("\n")
            st.markdown("Average Kms/EPod")
            st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                        str(avgkm)+"</span>", unsafe_allow_html=True)

        total_duration = citydf.groupby(
            'EPOD Name')['Duration'].sum().reset_index()

        average_duration = citydf.groupby(
            'EPOD Name')['Duration'].mean().reset_index().round(1)

        avgdur = average_duration['Duration'].mean().round(2)
        with col3:
            for i in range(1, 46):
                st.write("\n")
            st.markdown("Average Duration/Session")
            st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                        str(avgdur)+"</span>", unsafe_allow_html=True)
        average_kwh = citydf.groupby(
            'EPOD Name')['KWH Pumped Per Session'].mean().reset_index().round(1)
        avgkwh = average_kwh['KWH Pumped Per Session'].mean().round(2)
        with col4:
            st.markdown("Average kWh/EPod")
            st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                        str(avgkwh)+"</span>", unsafe_allow_html=True)

        fig = go.Figure()
        for color, kpi_group in record_count_df.groupby('Color'):

            fig.add_trace(go.Bar(
                x=kpi_group['EPOD Name'],
                y=kpi_group['Record Count'],
                text=kpi_group['Record Count'],
                textposition='auto',
                name=color,
                marker=dict(color=color),
                width=0.38,
                showlegend=False


            ))

        fig.update_layout(

            xaxis={'categoryorder': 'array',
                   'categoryarray': record_count_df['EPOD Name'],
                   'fixedrange': True},
            yaxis={'categoryorder': 'total ascending', 'range': y_axis_range,
                   'fixedrange': True},
            xaxis_title='EPOD Name',
            yaxis_title='Sessions',
            height=340,
            width=600,
            title="T-15 for each EPod",
            legend=dict(
                title_font=dict(size=14),
                font=dict(size=12),
                x=0,
                y=1.1,
                orientation='h',


            )
        )
        with col1:
            for i in range(1, 5):
                st.write("\n")
            st.plotly_chart(fig)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=average_duration['EPOD Name'],
            y=average_duration['Duration'],
            text=average_duration['Duration'],
            textposition='auto',
            name='Average Duration',

        ))

        fig.update_layout(
            xaxis_title='EPOD Name',
            yaxis_title='Average Duration',
            barmode='group',
            width=600,
            height=340,
            title="Avg Duration Per EPod",

        )
        with col4:
            for i in range(1, 4):
                st.write("\n")
            st.plotly_chart(fig)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=average_kms['EPOD Name'],
            y=average_kms['KM Travelled for Session'],
            text=average_kms['KM Travelled for Session'],
            textposition='auto',
            name='Average KM Travelled for Session',

        ))

        fig.update_layout(
            xaxis_title='EPOD Name',
            yaxis_title='Average KM Travelled for Session',
            barmode='group',
            width=600,
            height=340,
            title="Avg KMs Per EPod",
        )
        with col1:
            st.plotly_chart(fig)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=average_kwh['EPOD Name'],
            y=average_kwh['KWH Pumped Per Session'],
            text=average_kwh['KWH Pumped Per Session'],
            textposition='auto',
            name='Average KWH Pumped Per Session',

        ))
        fig.update_layout(
            xaxis_title='EPOD Name',
            yaxis_title='Average KWH Pumped Per Session',
            barmode='group',
            width=600,
            height=340,
            title="Avg kWh Per EPod",
        )
        with col4:
            st.plotly_chart(fig)

with tab4:

    min_date = df['Actual Date'].min().date()
    max_date = df['Actual Date'].max().date()
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col1:
        start_date = st.date_input(
            'Start Date', min_value=min_date, max_value=max_date, value=min_date)
    with col2:
        end_date = st.date_input(
            'End Date', min_value=min_date, max_value=max_date, value=max_date)

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered_df = df[(df['Actual Date'] >= start_date)
                     & (df['Actual Date'] <= end_date)]

    max_sessions = filtered_df.groupby('Actual OPERATOR NAME')[
        'Actual Date'].count().reset_index()
    max_sessions.columns = ['Actual OPERATOR NAME', 'Max Sessions']

    # Calculate the count of unique cities
    unique_cities = filtered_df.groupby(['Actual OPERATOR NAME', 'Actual Date'])[
        'Customer Location City'].nunique().reset_index()
    unique_cities = unique_cities.groupby('Actual OPERATOR NAME')[
        'Customer Location City'].max().reset_index()
    unique_cities.columns = ['Actual OPERATOR NAME', 'Unique Cities']

    # Calculate the count of KPI flags
    kpi_flags = filtered_df.groupby(['Actual OPERATOR NAME', 'Actual Date'])[
        't-15_kpi'].unique().reset_index()
    kpi_flags['t-15_kpi'] = kpi_flags['t-15_kpi'].apply(
        lambda x: x[0])
    kpi_counts = kpi_flags.groupby('Actual OPERATOR NAME')[
        't-15_kpi'].value_counts().unstack().reset_index()
    kpi_counts = kpi_counts.fillna(0)

    # Calculate the count of working days
    working_days = filtered_df.groupby('Actual OPERATOR NAME')[
        'Actual Date'].nunique().reset_index()
    working_days.columns = ['Actual OPERATOR NAME', 'Working Days']

    # Merge all the metrics into a single DataFrame
    merged_df = max_sessions.merge(
        unique_cities, on='Actual OPERATOR NAME', how='left')
    merged_df = merged_df.merge(
        kpi_counts, on='Actual OPERATOR NAME', how='left')
    merged_df = merged_df.merge(
        working_days, on='Actual OPERATOR NAME', how='left')

    # Define weights for each metric
    weights = {
        'Max Sessions': 1,
        'Unique Cities': 2,
        'Flag 1 Count': 3,
        'Flag 2 Count': -1,
        'Working Days': 1
    }

    # Filter the weights for only the metrics present in the merged DataFrame
    weights = {metric: weight for metric,
               weight in weights.items() if metric in merged_df.columns}

    # Calculate scores for each metric
    for metric, weight in weights.items():
        merged_df[metric + ' Score'] = pd.to_numeric(
            merged_df[metric], errors='coerce') * weight

    # Calculate total score
    merged_df['Total Score'] = merged_df[[
        metric + ' Score' for metric in weights]].sum(axis=1)

    # Sort the DataFrame by total score and assign rank
    merged_df = merged_df.sort_values(
        'Total Score', ascending=False).reset_index(drop=True)
    merged_df['Rank'] = merged_df.index + 1

    # Display the operator with the lowest rank
    lowest_rank_operator = merged_df.loc[merged_df['Rank'].idxmin(
    ), 'Actual OPERATOR NAME']

    # Display the operator with the highest rank
    highest_rank_operator = merged_df.loc[merged_df['Rank'].idxmax(
    ), 'Actual OPERATOR NAME']

    with col5:
        st.markdown("Lowest Scoring:")
        st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                    highest_rank_operator+"</span>", unsafe_allow_html=True)
    with col6:
        st.markdown("Highest Scoring:")
        st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                    lowest_rank_operator+"</span>", unsafe_allow_html=True)

    grouped_df = filtered_df.groupby(
        ['Actual OPERATOR NAME', 'Customer Location City']).size().reset_index()
    grouped_df.columns = ['Operator', 'City', 'Count']

    # Filter cities
    cities_to_include = ["Gurgaon", "Delhi", "Faridabad", "Noida", "Ghaziabad"]
    grouped_df = grouped_df[grouped_df['City'].isin(cities_to_include)]

    # Pivot the data to create a matrix of operator-city counts
    pivot_df = grouped_df.pivot(
        index='Operator', columns='City', values='Count').fillna(0)

    # Customize the plot parameters
    figure_width = 1.9  # Adjust the figure width
    figure_height = 6  # Adjust the figure height
    font_size_heatmap = 5  # Adjust the font size for the heatmap
    font_size_labels = 4  # Adjust the font size for x-axis and y-axis labels

    # Create the heatmap plot using seaborn
    plt.figure(figsize=(figure_width, figure_height), facecolor='none')

    sns.heatmap(pivot_df, cmap='YlGnBu', annot=True, fmt='g', linewidths=0.5, cbar=False,
                annot_kws={'fontsize': font_size_heatmap})

    plt.title('Operator v/s Locations',
              fontsize=8, color='white')
    plt.xlabel('Customer Location City',
               fontsize=font_size_labels, color='white')
    plt.ylabel('Operator', fontsize=font_size_labels, color='white')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0, ha='center',
               fontsize=font_size_labels, color='white')
    plt.yticks(fontsize=font_size_labels, color='white')
    with col1:
        st.pyplot(plt, use_container_width=False)

    grouped_df = filtered_df.groupby(
        ['Actual OPERATOR NAME', 'Customer Location City']).size().reset_index()
    grouped_df.columns = ['Operator', 'City', 'Count']

    cities_to_include = ["Gurgaon", "Delhi",
                         "Faridabad", "Noida", "Ghaziabad"]
    grouped_df = grouped_df[grouped_df['City'].isin(cities_to_include)]

    # Create a list of unique cities for the dropdown
    cities = np.append(grouped_df['City'].unique(), "All")

    with col3:
        selected_city = st.selectbox('Select City', cities)

    # Filter data based on selected city
    if selected_city == "All":
        city_df = grouped_df
    else:
        city_df = grouped_df[grouped_df['City'] == selected_city]

    grouped_df = filtered_df.groupby(
        ['Actual OPERATOR NAME', 'Customer Location City']).size().reset_index()
    grouped_df.columns = ['Operator', 'City', 'Count']
    total_sessions = city_df.groupby('Operator')['Count'].sum().reset_index()

    # Calculate the working days per operator
    working_days = filtered_df.groupby('Actual OPERATOR NAME')[
        'Actual Date'].nunique().reset_index()
    working_days.columns = ['Operator', 'Working Days']

    # Merge total sessions and working days dataframes
    merged_df = pd.merge(total_sessions, working_days, on='Operator')

    # Calculate the average sessions per operator
    avg_sessions = pd.DataFrame()
    avg_sessions['Operator'] = merged_df['Operator']
    avg_sessions['Avg. Sessions'] = merged_df['Count'] / \
        merged_df['Working Days']
    avg_sessions['Avg. Sessions'] = avg_sessions['Avg. Sessions'].round(0)
    fig_sessions = go.Figure()
    fig_sessions.add_trace(go.Bar(
        x=total_sessions['Operator'],
        y=total_sessions['Count'],
        name='Total Sessions',
        text=total_sessions['Count'],
        textposition='auto',
        marker=dict(color='blue'),
        width=0.5
    ))
    fig_sessions.add_trace(go.Bar(
        x=avg_sessions['Operator'],
        y=avg_sessions['Avg. Sessions'],
        name='Average Sessions',
        text=avg_sessions['Avg. Sessions'],
        textposition='auto',
        marker=dict(color='green'),
        width=0.38
    ))
    fig_sessions.update_layout(
        title='Total Sessions and Average Sessions per Operator',
        xaxis=dict(title='Operator'),
        yaxis=dict(title='Count / Average Sessions'),
        margin=dict(l=50, r=50, t=80, b=80),
        legend=dict(yanchor="top", y=1.1, xanchor="left",
                    x=0.01, orientation="h"),
        width=1050,
        height=500
    )
    with col4:
        for i in range(1, 10):
            st.write("\n")
        st.plotly_chart(fig_sessions)

    # Calculate working days per operator
    working_days = filtered_df.groupby('Actual OPERATOR NAME')[
        'Actual Date'].nunique().reset_index()
    working_days.columns = ['Operator', 'Working Days']

    # Filter working days based on selected city
    if selected_city == "All":
        selected_working_days = working_days
    else:
        selected_working_days = working_days[working_days['Operator'].isin(
            city_df['Operator'])]

    with col7:
        st.markdown(f"Most Operator in {selected_city}")
        most_operator = city_df.loc[city_df['Count'].idxmax(), 'Operator']
        st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                    most_operator+"</span>", unsafe_allow_html=True)

    with col8:
        st.markdown(f"Least Operator in {selected_city}")
        least_operator = city_df.loc[city_df['Count'].idxmin(), 'Operator']
        st.markdown(
            "<span style = 'font-size:25px;line-height: 0.8;'>"+least_operator+"</span>",
            unsafe_allow_html=True)
    fig_working_days = go.Figure(data=go.Bar(
        x=selected_working_days['Operator'],
        y=selected_working_days['Working Days'],
        marker=dict(color='lightgreen'),
        text=selected_working_days['Working Days']
    ))
    fig_working_days.update_layout(
        title='Working Days per Operator',
        xaxis=dict(title='Operator'),
        yaxis=dict(title='Working Days'),
        margin=dict(l=50, r=50, t=80, b=80),
        width=800,
        height=500
    )
    with col4:
        st.plotly_chart(fig_working_days)
