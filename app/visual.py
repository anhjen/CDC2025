import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

def create_world_astronaut_map(df):
    """
    Create an interactive world map showing astronaut statistics by country
    """
    if df.empty:
        st.warning("No international astronaut data found.")
        return None, None
    
    # Calculate country statistics
    country_stats = df.groupby('country').agg({
        'Name': 'count',
        'total_flights': 'sum',
        'total_time': 'sum',
        'gender': lambda x: Counter(x),
    }).reset_index()
    
    # Rename columns for clarity
    country_stats.columns = ['Country', 'Total_Astronauts', 'Total_Flights', 'Total_Time_Minutes', 'Gender_Dist']
    
    # Convert time to hours for better readability
    country_stats['Total_Time_Hours'] = country_stats['Total_Time_Minutes'] / 60
    
    # Calculate averages
    country_stats['Avg_Flights_Per_Astronaut'] = (country_stats['Total_Flights'] / country_stats['Total_Astronauts']).round(2)
    country_stats['Avg_Time_Hours_Per_Astronaut'] = (country_stats['Total_Time_Hours'] / country_stats['Total_Astronauts']).round(1)
    
    # Country name mapping for plotly (some countries need to match plotly's country names)
    country_mapping = {
        'Soviet Union': 'Russia',  # Map Soviet Union to Russia for visualization
        'East Germany': 'Germany',  # Map East Germany to Germany
        'Czechoslovakia': 'Czech Republic',  # Map to Czech Republic
        # Add more mappings as needed
    }
    
    # Apply country mapping for the map visualization
    country_stats['Country_Map'] = country_stats['Country'].replace(country_mapping)
    
    if country_stats.empty:
        st.warning("No country data available for mapping.")
        return None, country_stats
    
    # Create the world choropleth map
    fig = px.choropleth(
        country_stats,
        locations='Country_Map',
        locationmode='country names',
        color='Total_Astronauts',
        hover_name='Country',  # Show original country name
        hover_data={
            'Country_Map': False,  # Hide the mapped country name
            'Total_Astronauts': ':.0f',
            'Total_Flights': ':.0f',
            'Total_Time_Hours': ':.1f',
            'Avg_Flights_Per_Astronaut': ':.2f',
            'Avg_Time_Hours_Per_Astronaut': ':.1f'
        },
        color_continuous_scale='Viridis',
        range_color=[0, country_stats['Total_Astronauts'].max()],
        title='Global Astronauts by Country',
        labels={
            'Total_Astronauts': 'Number of Astronauts',
            'Country': 'Country',
            'Total_Flights': 'Total Flights',
            'Total_Time_Hours': 'Total Time (Hours)',
            'Avg_Flights_Per_Astronaut': 'Avg Flights/Astronaut',
            'Avg_Time_Hours_Per_Astronaut': 'Avg Hours/Astronaut'
        }
    )
    
    # Update layout for better interactivity and visual appeal
    fig.update_layout(
        title=dict(
            text='International Astronaut Distribution by Country',
            x=0.5,
            font=dict(size=18)
        ),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth',
            bgcolor='rgba(0,0,0,0)'
        ),
        height=600,
        width=1200,
        coloraxis_colorbar=dict(
            title="Number of Astronauts",
            thickness=15,
            len=0.7,
            x=1.02
        ),
        margin=dict(l=0, r=50, t=80, b=0)
    )
    
    # Enhance hover information
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>' +
                     'Astronauts: %{z}<br>' +
                     'Total Flights: %{customdata[2]}<br>' +
                     'Total Hours: %{customdata[3]:.1f}<br>' +
                     'Avg Flights/Astronaut: %{customdata[4]:.2f}<br>' +
                     'Avg Hours/Astronaut: %{customdata[5]:.1f}<br>' +
                     '<extra></extra>',
        customdata=country_stats[['Country_Map', 'Total_Astronauts', 'Total_Flights', 'Total_Time_Hours', 'Avg_Flights_Per_Astronaut', 'Avg_Time_Hours_Per_Astronaut']].values
    )
    
    return fig, country_stats

def create_country_details_card(country_data, df):
    """
    Create a detailed information card for a selected country
    """
    if country_data.empty:
        return None
    
    country_name = country_data['Country'].iloc[0]
    
    # Get astronauts from this country
    country_astronauts = df[df['country'] == country_name]
    
    if country_astronauts.empty:
        st.warning(f"No astronaut data found for {country_name}")
        return
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Astronauts", len(country_astronauts))
    
    with col2:
        total_flights = country_astronauts['total_flights'].sum()
        st.metric("Total Flights", int(total_flights))
    
    with col3:
        total_hours = country_astronauts['total_time'].sum() / 60  # Convert minutes to hours
        st.metric("Total Flight Hours", f"{total_hours:.1f}")
    
    with col4:
        avg_flights = total_flights / len(country_astronauts) if len(country_astronauts) > 0 else 0
        st.metric("Avg Flights/Astronaut", f"{avg_flights:.2f}")
    
    # Show top astronauts from this country
    st.markdown("### Top Astronauts")
    top_astronauts = country_astronauts.nlargest(5, 'total_flights')[
        ['Name', 'total_flights', 'total_time', 'gender', 'Flights']
    ].copy()
    
    # Convert time to hours for display
    top_astronauts['total_time_hours'] = (top_astronauts['total_time'] / 60).round(1)
    top_astronauts = top_astronauts.drop('total_time', axis=1)
    top_astronauts.columns = ['Name', 'Flights', 'Flight Hours', 'Gender', 'Missions']
    
    st.dataframe(top_astronauts, use_container_width=True)
    
    # Show distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Gender Distribution")
        gender_counts = country_astronauts['gender'].value_counts()
        if not gender_counts.empty:
            fig_gender = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title=f"Gender Distribution - {country_name}",
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
            )
            st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        st.markdown("### Flight Experience Distribution")
        # Create flight experience categories
        def categorize_flights(flights):
            if flights == 1:
                return "Single Flight"
            elif flights <= 3:
                return "2-3 Flights"
            elif flights <= 5:
                return "4-5 Flights"
            else:
                return "6+ Flights"
        
        country_astronauts_copy = country_astronauts.copy()
        country_astronauts_copy['flight_category'] = country_astronauts_copy['total_flights'].apply(categorize_flights)
        flight_dist = country_astronauts_copy['flight_category'].value_counts()
        
        if not flight_dist.empty:
            fig_flights = px.bar(
                x=flight_dist.values,
                y=flight_dist.index,
                orientation='h',
                title=f"Flight Experience - {country_name}",
                color=flight_dist.values,
                color_continuous_scale='Blues'
            )
            fig_flights.update_layout(showlegend=False)
            st.plotly_chart(fig_flights, use_container_width=True)

def create_us_states_map(df):
    """
    Create an interactive US states map showing astronaut statistics by birth state
    """
    us_df = df[df['country'] == 'United States'].copy()
    
    if us_df.empty:
        st.warning("No US astronaut data found.")
        return None, None
    
    # Clean and standardize state names from birth_place
    def extract_state(birth_place):
        if pd.isnull(birth_place):
            return 'Unknown'
        
        state_abbrev = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
            'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
            'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
            'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
            'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
            'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
            'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
            'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
            'DC': 'District of Columbia'
        }
        
        birth_place = str(birth_place).strip()
        
        if birth_place.upper() in state_abbrev:
            return state_abbrev[birth_place.upper()]
        
        for abbrev, full_name in state_abbrev.items():
            if full_name.lower() in birth_place.lower():
                return full_name
            if abbrev.lower() == birth_place.lower():
                return full_name
        
        return 'Unknown'
    
    us_df['State'] = us_df['birth_place'].apply(extract_state)  # Updated column name
    
    # Calculate state statistics (updated column names)
    state_stats = us_df.groupby('State').agg({
        'Name': 'count',
        'Total Flights': 'sum',
        'Flight_Time_Hours': 'sum',
        'gender': lambda x: Counter(x),  # Updated column name
        'Status': lambda x: Counter(x),
        'Undergraduate_Major_Grouped': lambda x: Counter(x).most_common(1)[0][0] if len(x) > 0 else 'Unknown'
    }).reset_index()
    
    # Rename columns for clarity
    state_stats.columns = ['State', 'Total_Astronauts', 'Total_Flights', 'Total_Flight_Hours', 'Gender_Dist', 'Status_Dist', 'Top_Major']
    
    # Calculate averages
    state_stats['Avg_Flights_Per_Astronaut'] = (state_stats['Total_Flights'] / state_stats['Total_Astronauts']).round(2)
    state_stats['Avg_Flight_Hours_Per_Astronaut'] = (state_stats['Total_Flight_Hours'] / state_stats['Total_Astronauts']).round(1)
    
    # Filter out 'Unknown' states for the map
    map_stats = state_stats[state_stats['State'] != 'Unknown'].copy()
    
    if map_stats.empty:
        st.warning("No state data available for mapping.")
        return None, state_stats
    
    # Create the US states choropleth map with better interactivity
    fig = px.choropleth(
        map_stats,
        locations='State',
        locationmode='USA-states',
        color='Total_Astronauts',
        hover_name='State',
        hover_data={
            'Total_Astronauts': ':.0f',
            'Total_Flights': ':.0f',
            'Total_Flight_Hours': ':.1f',
            'Avg_Flights_Per_Astronaut': ':.2f',
            'Top_Major': True
        },
        color_continuous_scale='Blues',
        range_color=[0, map_stats['Total_Astronauts'].max()],
        title='US Astronauts by Birth State',
        labels={
            'Total_Astronauts': 'Number of Astronauts',
            'State': 'State'
        }
    )
    
    # Update layout for better interactivity and visual appeal
    fig.update_layout(
        title=dict(
            text='Interactive US Astronaut Distribution by State',
            x=0.5,
            font=dict(size=18)
        ),
        geo=dict(
            scope='usa',
            showlakes=True,
            lakecolor='rgb(255, 255, 255)',
            projection_type='albers usa',
            bgcolor='rgba(0,0,0,0)'
        ),
        height=600,
        width=1000,
        coloraxis_colorbar=dict(
            title="Number of Astronauts",
            thickness=15,
            len=0.7,
            x=1.02
        ),
        margin=dict(l=0, r=50, t=80, b=0)
    )
    
    # Enhance hover information
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>' +
                     'Astronauts: %{z}<br>' +
                     'Total Flights: %{customdata[1]}<br>' +
                     'Flight Hours: %{customdata[2]:.1f}<br>' +
                     'Avg Flights/Astronaut: %{customdata[3]:.2f}<br>' +
                     'Top Major: %{customdata[4]}<br>' +
                     '<extra></extra>',
        customdata=map_stats[['Total_Astronauts', 'Total_Flights', 'Total_Flight_Hours', 'Avg_Flights_Per_Astronaut', 'Top_Major']].values
    )
    
    return fig, state_stats

def create_state_details_card(state_data, df):
    """
    Create a detailed information card for a selected US state
    """
    if state_data.empty:
        return None
    
    state_name = state_data['State'].iloc[0]
    
    # Get US astronauts from this state (updated column name)
    us_df = df[df['country'] == 'United States'].copy()
    
    # Apply the same state extraction logic
    def extract_state(birth_place):
        if pd.isnull(birth_place):
            return 'Unknown'
        
        state_abbrev = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
            'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
            'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
            'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
            'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
            'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
            'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
            'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
            'DC': 'District of Columbia'
        }
        
        birth_place = str(birth_place).strip()
        if birth_place in state_abbrev.values():
            return birth_place
        if birth_place.upper() in state_abbrev:
            return state_abbrev[birth_place.upper()]
        for abbrev, full_name in state_abbrev.items():
            if full_name.lower() in birth_place.lower():
                return full_name
            if abbrev.lower() == birth_place.lower():
                return full_name
        return 'Unknown'
    
    us_df['State'] = us_df['birth_place'].apply(extract_state)  # Updated column name
    state_astronauts = us_df[us_df['State'] == state_name]
    
    if state_astronauts.empty:
        st.warning(f"No astronaut data found for {state_name}")
        return
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Astronauts", len(state_astronauts))
    
    with col2:
        total_flights = state_astronauts['Total Flights'].sum()
        st.metric("Total Flights", int(total_flights))
    
    with col3:
        total_hours = state_astronauts['Flight_Time_Hours'].sum()
        st.metric("Total Flight Hours", f"{total_hours:.1f}")
    
    with col4:
        avg_flights = total_flights / len(state_astronauts) if len(state_astronauts) > 0 else 0
        st.metric("Avg Flights/Astronaut", f"{avg_flights:.2f}")
    
    # Show top astronauts from this state
    st.markdown("### Top Astronauts")
    # Create flight time display for new dataset format
    top_astronauts_temp = state_astronauts.nlargest(5, 'Total Flights').copy()
    top_astronauts_temp['Flight_Time_Display'] = top_astronauts_temp['Flight_Time_Hours'].apply(
        lambda x: f"{int(x//24):03d}:{int(x%24):02d}:{int((x%1)*60):02d}"
    )
    top_astronauts = top_astronauts_temp[
        ['Name', 'Total Flights', 'Flight_Time_Display', 'Status', 'Undergraduate_Major_Grouped']
    ]
    st.dataframe(top_astronauts, use_container_width=True)
    
    # Show distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Gender Distribution")
        gender_counts = state_astronauts['gender'].value_counts()  # Updated column name
        if not gender_counts.empty:
            fig_gender = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title=f"Gender Distribution - {state_name}",
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        st.markdown("### Major Distribution")
        major_counts = state_astronauts['Undergraduate_Major_Grouped'].value_counts().head(5)
        if not major_counts.empty:
            fig_major = px.bar(
                x=major_counts.values,
                y=major_counts.index,
                orientation='h',
                title=f"Top 5 Majors - {state_name}",
                color=major_counts.values,
                color_continuous_scale='Blues'
            )
            fig_major.update_layout(showlegend=False)
            st.plotly_chart(fig_major, use_container_width=True)

def display_us_stats(df):
    """
    Display US astronaut statistics
    """
    st.markdown("## US Astronaut Statistics")
    
    # Filter for US astronauts (updated column name)
    us_df = df[df['country'] == 'United States']
    
    if us_df.empty:
        st.warning("No US astronaut data found.")
        return
    
    # US metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total US Astronauts", len(us_df))
    
    with col2:
        total_flights = us_df['Total Flights'].sum()
        st.metric("Total US Flights", int(total_flights))
    
    with col3:
        total_hours = us_df['Flight_Time_Hours'].sum()
        st.metric("Total US Flight Hours", f"{total_hours:.0f}")
    
    with col4:
        # Extract states for counting
        def extract_state(birth_place):
            if pd.isnull(birth_place):
                return 'Unknown'
            state_abbrev = {
                'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
                'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
                'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
                'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
                'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
                'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
                'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
                'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
                'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
                'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
                'DC': 'District of Columbia'
            }
            birth_place = str(birth_place).strip()
            if birth_place in state_abbrev.values():
                return birth_place
            if birth_place.upper() in state_abbrev:
                return state_abbrev[birth_place.upper()]
            for abbrev, full_name in state_abbrev.items():
                if full_name.lower() in birth_place.lower():
                    return full_name
            return 'Unknown'
        
        us_df_copy = us_df.copy()
        us_df_copy['State'] = us_df_copy['birth_place'].apply(extract_state)  # Updated column name
        states_represented = us_df_copy[us_df_copy['State'] != 'Unknown']['State'].nunique()
        st.metric("States Represented", states_represented)
    
    # Top states
    st.markdown("### Top 10 States by Astronaut Count")
    us_df_copy = us_df.copy()
    us_df_copy['State'] = us_df_copy['birth_place'].apply(extract_state)  # Updated column name
    top_states = us_df_copy[us_df_copy['State'] != 'Unknown']['State'].value_counts().head(10)
    
    if not top_states.empty:
        fig_states = px.bar(
            x=top_states.values,
            y=top_states.index,
            orientation='h',
            title="States with Most Astronauts",
            labels={'x': 'Number of Astronauts', 'y': 'State'},
            color=top_states.values,
            color_continuous_scale='Blues'
        )
        fig_states.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_states, use_container_width=True)