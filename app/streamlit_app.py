"""
Streamlit Application for Astronaut Mission Predictor

This module contains the main Streamlit application interface for predicting
astronaut missions using NASA and International models.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Try to import plotly, but handle gracefully if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    st.warning("Plotly not available. Some visualizations will be disabled.")
    PLOTLY_AVAILABLE = False

from nasa_model import load_nasa_models, predict_nasa_astronaut, find_similar_astronauts, hours_to_ddd_hh_mm, group_majors, group_military_branches
from international_model import load_international_data, predict_international_astronaut, get_country_stats
import visual

def load_nasa_data():
    """Load NASA astronaut data."""
    try:
        return pd.read_csv('data/nasa_master_clean.csv')
    except FileNotFoundError:
        st.error("NASA data file not found. Please ensure 'data/nasa_master_clean.csv' exists.")
        return None

def create_nasa_tab():
    """Create the NASA prediction tab."""
    st.header("🚀 NASA Astronaut Mission Predictor")
    
    # Load data and models
    nasa_data = load_nasa_data()
    nasa_models = load_nasa_models()
    
    if nasa_data is None or nasa_models is None:
        st.error("Unable to load NASA data or models. Please check your data files.")
        return
    
    # Input form
    st.subheader("Enter Astronaut Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        birth_state = st.selectbox(
            "Birth State", 
            sorted(nasa_data['birth_place'].dropna().unique()),
            help="Select the astronaut's birth state (from birth_place column)"
        )
        undergrad_major = st.selectbox(
            "Undergraduate Major",
            sorted(nasa_data['undergraduate_major'].dropna().unique()),
            help="Select the undergraduate field of study"
        )
        military_branch = st.selectbox(
            "Military Branch",
            sorted(nasa_data['Military Branch'].dropna().unique()),
            help="Select military service branch (if applicable)"
        )
    
    with col2:
        grad_major = st.selectbox(
            "Graduate Major",
            sorted(nasa_data['Graduate Major'].dropna().unique()),
            help="Select the graduate field of study"
        )
        
        military_rank = st.selectbox(
            "Military Rank",
            sorted(nasa_data['Military Rank'].dropna().unique()),
            help="Select military rank (if applicable)"
        )
        
        year = st.number_input(
            "Year of Selection", 
            min_value=1950, 
            max_value=datetime.now().year + 10,
            value=datetime.now().year,
            help="Year the astronaut was selected"
        )
    
    # Predict button
    if st.button("🔮 Predict Mission Statistics", type="primary"):
        # Prepare input data
        input_data = {
            'birth_place': birth_state,
            'undergraduate_major': undergrad_major,
            'Graduate Major': grad_major,
            'Military Branch': military_branch,
            'Military Rank': military_rank,
            'Year': year
        }
        
        # Make predictions
        with st.spinner("Making predictions..."):
            flights_pred, time_pred, performance = predict_nasa_astronaut(nasa_models, input_data)
        
        if flights_pred is not None and time_pred is not None:
            # Display results
            st.success("🎯 Prediction Complete!")
            # Create metrics display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Predicted Total Flights",
                    f"{flights_pred:.1f}",
                    help="Expected number of space missions"
                )
            with col2:
                # Convert minutes to hours for display, and show both
                time_hours = time_pred / 60
                time_formatted = hours_to_ddd_hh_mm(time_hours)
                st.metric(
                    "Predicted Total Time in Space",
                    f"{time_pred:.0f} min ({time_hours:.1f} hr)",
                    help="Expected total time spent in space (minutes and hours)"
                )
                st.caption(f"DDD:HH:MM format: {time_formatted}")
            with col3:
                if performance:
                    st.metric(
                        "Model Accuracy (R²)",
                        f"{performance.get('flights_r2', 0):.3f}",
                        help="Model prediction accuracy"
                    )
            
            # Show similar astronauts
            st.subheader("👨‍🚀 Similar Astronauts")
            similar_astronauts = find_similar_astronauts(input_data, nasa_data, top_n=5)
            
            if similar_astronauts:
                for i, (name, score, astronaut_data) in enumerate(similar_astronauts, 1):
                    with st.expander(f"{i}. {name} (Similarity Score: {score})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Birth State:** {astronaut_data['birth_place']}")
                            st.write(f"**Military Branch:** {astronaut_data['Military Branch']}")
                            st.write(f"**Year Selected:** {astronaut_data['Year']}")
                        with col2:
                            st.write(f"**Graduate Major:** {astronaut_data['Graduate Major']}")
                            st.write(f"**Total Flights:** {astronaut_data['Total Flights']}")
                            if not pd.isna(astronaut_data['Total time in space (hours)']):
                                time_formatted = hours_to_ddd_hh_mm(astronaut_data['Total time in space (hours)'])
                                st.write(f"**Total Time in Space:** {time_formatted}")

def create_international_tab():
    """Create the International prediction tab."""
    st.header("🌍 International Astronaut Mission Predictor")
    
    # Load international data
    international_data = load_international_data()
    if international_data is None:
        return
    
    # Input form
    st.subheader("Enter Astronaut Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        country = st.selectbox(
            "Country",
            sorted(international_data['country'].unique()),
            help="Select the astronaut's country"
        )
    
    with col2:
        gender = st.selectbox(
            "Gender",
            sorted(international_data['gender'].dropna().unique()),
            help="Select gender"
        )
    
    # Predict button
    if st.button("🔮 Predict International Mission", type="primary"):
        with st.spinner("Making predictions..."):
            prediction_result = predict_international_astronaut(country, gender)
        if prediction_result:
            st.success("🎯 Prediction Complete!")
            st.subheader("Prediction Results")
            st.write(f"**Predicted Total Flights:** {prediction_result['total_flights']:.1f}")
            total_time_min = prediction_result['total_time']
            total_time_hr = total_time_min / 60
            st.write(f"**Predicted Total Time:** {total_time_min:.1f} min ({total_time_hr:.1f} hr)")
            st.caption(f"DDD:HH:MM format: {hours_to_ddd_hh_mm(total_time_hr)}")

def create_visualizations_tab():
    """Create the data visualizations tab."""
    st.header("📊 Data Visualizations")
    
    # Load data
    nasa_data = load_nasa_data()
    international_data = load_international_data()
    
    if nasa_data is None:
        st.error("NASA data not available for visualizations")
        return
    
    # Visualization options
    viz_type = st.selectbox(
        "Select Visualization Type",
        [
            "NASA Astronaut Statistics",
            "Mission Duration Trends",
            "Country Comparisons",
            "Gender Distribution",
            "Educational Background Analysis"
        ]
    )
    
    try:
        if viz_type == "NASA Astronaut Statistics":
            visual.plot_nasa_statistics(nasa_data)
        elif viz_type == "Mission Duration Trends":
            visual.plot_mission_trends(nasa_data)
        elif viz_type == "Country Comparisons" and international_data is not None:
            visual.plot_country_comparison(international_data)
        elif viz_type == "Gender Distribution":
            if international_data is not None:
                visual.plot_gender_distribution(international_data)
            else:
                st.warning("International data not available for gender distribution analysis")
        elif viz_type == "Educational Background Analysis":
            visual.plot_education_analysis(nasa_data)
        else:
            st.info("Select a visualization type to display charts")
    except Exception as e:
        st.error(f"Error generating visualization: {e}")

def create_about_tab():
    """Create the about/information tab."""
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## 🚀 Astronaut Mission Predictor
    
    This application uses machine learning to predict astronaut mission characteristics based on historical data from NASA and international space agencies.
    
    ### 📊 Data Sources
    - **NASA Astronaut Database**: Historical data on NASA astronauts including missions, backgrounds, and performance
    - **International Astronaut Database**: Global astronaut data with mission information
    
    ### 🤖 Machine Learning Models
    - **Random Forest Regression**: For predicting mission duration and flight count
    - **Feature Engineering**: Grouping of educational backgrounds and military branches
    - **Similarity Matching**: Finding comparable astronauts based on background characteristics
    
    ### 📈 Features
    - **NASA Predictions**: Predict total flights and space time for potential NASA astronauts
    - **International Predictions**: Estimate mission duration for international astronauts
    - **Data Visualizations**: Interactive charts showing trends and patterns
    - **Similar Astronauts**: Find historical astronauts with similar backgrounds
    
    ### 🎯 Accuracy
    The models are trained on historical data and provide estimates based on past patterns. 
    Actual mission assignments depend on many factors not captured in the available data.
    
    ### 🔧 Technical Details
    - Built with **Streamlit** for the web interface
    - **Scikit-learn** for machine learning models
    - **Plotly** for interactive visualizations
    - **Pandas** for data manipulation
    """)
    
    # Model performance metrics
    nasa_models = load_nasa_models()
    if nasa_models and nasa_models.get('performance_metrics'):
        st.subheader("📊 Model Performance")
        performance = nasa_models['performance_metrics']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Flights Prediction R²", f"{performance.get('flights_r2', 0):.3f}")
        with col2:
            st.metric("Time Prediction R²", f"{performance.get('time_r2', 0):.3f}")

def main():
    """Main application function."""
    # Page config
    st.set_page_config(
        page_title="Astronaut Mission Predictor",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # App title and description
    st.title("🚀 Astronaut Mission Predictor")
    st.markdown("*Predict astronaut mission characteristics using machine learning*")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🇺🇸 NASA Predictions", 
        "🌍 International Predictions", 
        "📊 Visualizations", 
        "ℹ️ About"
    ])
    
    with tab1:
        create_nasa_tab()
    
    with tab2:
        create_international_tab()
    
    with tab3:
        create_visualizations_tab()
    
    with tab4:
        create_about_tab()
    
    # Sidebar information
    with st.sidebar:
        st.header("🎛️ Application Info")
        st.info("This application predicts astronaut mission characteristics based on historical data.")
        
        st.header("📋 Quick Guide")
        st.markdown("""
        1. **NASA Tab**: Predict flights and space time for NASA candidates
        2. **International Tab**: Predict mission duration for international astronauts  
        3. **Visualizations**: Explore data trends and patterns
        4. **About**: Learn more about the models and data
        """)
        
        st.header("🔗 Navigation")
        st.markdown("""
        - Use the tabs above to switch between different prediction modes
        - Fill in the form fields and click predict
        - Explore similar astronauts and visualizations
        """)

if __name__ == "__main__":
    main()