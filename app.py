import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Load the trained model
model_file = "audience_rating_model-final.pkl"
model = joblib.load(model_file)

# Title and description for Streamlit app
st.title("🎬 Audience Rating Prediction App")
st.write(
    "Use this tool to predict the audience rating for a movie based on adjustable tomatometer ratings and other movie details.")

# Function to create movie data
def create_movie_data(tomatometer_rating):
    """
    Prepare a DataFrame with user-input movie data and a given tomatometer rating.
    """
    # Get inputs from session state or use defaults
    movie_title = st.session_state.get('movie_title', 'Blockbuster Movie')
    movie_info = st.session_state.get('movie_info', 'An outstanding critically acclaimed movie')
    critics_consensus = st.session_state.get('critics_consensus', 'Overwhelmingly positive reviews')
    rating = st.session_state.get('rating', 'PG-13')
    genre = st.session_state.get('genre', 'Drama')
    directors = st.session_state.get('directors', 'Famous Director')
    writers = st.session_state.get('writers', 'Top Screenwriter')
    cast = st.session_state.get('cast', 'Famous Actor A, Famous Actor B')
    in_theaters_date = st.session_state.get('in_theaters_date', '2024-12-01')
    on_streaming_date = st.session_state.get('on_streaming_date', '2025-01-01')
    runtime = st.session_state.get('runtime', 150)
    studio_name = st.session_state.get('studio_name', 'Top Studio')
    tomatometer_status = st.session_state.get('tomatometer_status', 'Certified Fresh')
    tomatometer_count = st.session_state.get('tomatometer_count', 550)

    data = {
        "movie_title": [movie_title],
        "movie_info": [movie_info],
        "critics_consensus": [critics_consensus],
        "rating": [rating],
        "genre": [genre],
        "directors": [directors],
        "writers": [writers],
        "cast": [cast],
        "in_theaters_date": [in_theaters_date],
        "on_streaming_date": [on_streaming_date],
        "runtime_in_minutes": [runtime],
        "studio_name": [studio_name],
        "tomatometer_status": [tomatometer_status],
        "tomatometer_rating": [tomatometer_rating],
        "tomatometer_count": [tomatometer_count],
    }
    return pd.DataFrame(data)


# User inputs in the sidebar
st.sidebar.header("📋 Enter Movie Details")
st.sidebar.text_input("Movie Title", "Blockbuster Movie", key="movie_title")
st.sidebar.text_area("Movie Info", "An outstanding critically acclaimed movie", key="movie_info")
st.sidebar.text_input("Critics Consensus", "Overwhelmingly positive reviews", key="critics_consensus")
st.sidebar.selectbox("Rating", ["G", "PG", "PG-13", "R", "NC-17"], index=2, key="rating")
st.sidebar.text_input("Genre", "Drama", key="genre")
st.sidebar.text_input("Director(s)", "Famous Director", key="directors")
st.sidebar.text_input("Writer(s)", "Top Screenwriter", key="writers")
st.sidebar.text_input("Cast", "Famous Actor A, Famous Actor B", key="cast")
st.sidebar.date_input("In Theaters Date", key="in_theaters_date")
st.sidebar.date_input("On Streaming Date", key="on_streaming_date")
st.sidebar.number_input("Runtime (in minutes)", min_value=1, max_value=500, value=150, key="runtime")
st.sidebar.text_input("Studio Name", "Top Studio", key="studio_name")
st.sidebar.selectbox("Tomatometer Status", ["Fresh", "Certified Fresh", "Rotten"], index=1, key="tomatometer_status")
st.sidebar.number_input("Tomatometer Count", min_value=1, value=550, key="tomatometer_count")

# Dropdown for single tomatometer rating selection
st.subheader("Select a Tomatometer Rating")
selected_tomatometer_rating = st.selectbox(
    "Choose a single Tomatometer Rating:",
    list(range(10, 101))  # Ratings from 10 to 100 (inclusive)
)

# Generate predictions for the selected rating
if st.button("Predict Audience Rating"):
    # Create movie data for the selected rating
    movie_data = create_movie_data(selected_tomatometer_rating)

    # Predict audience rating
    predicted_rating = model.predict(movie_data)[0]

    # Display the result
    st.subheader("🎯 Prediction Results")
    st.write(f"**Selected Tomatometer Rating:** {selected_tomatometer_rating}")
    st.write(f"**Predicted Audience Rating:** {predicted_rating:.2f}")

    # Display final movie features
    st.write("**Movie Features Used for Prediction:**")
    st.dataframe(movie_data)
