import streamlit as st
import joblib
import pandas as pd

# Load movies dataset
movies = joblib.load("model/movies.pkl")

# Fix Year column
movies["Year"] = pd.to_numeric(movies["Year"], errors="coerce")

st.title("🎬 Indian Movie Recommendation System")

# Filters
language = st.selectbox("Select Language", movies["Language"].dropna().unique())
genre = st.selectbox("Select Genre", movies["Genre"].dropna().unique())
year = st.selectbox("Select Year", sorted(movies["Year"].dropna().unique()))





# Movie selection
movie = st.selectbox("Select Movie", movies["Movie Name"].values)

# Recommendation
if st.button("Recommend"):

    selected_movie = movies[movies["Movie Name"] == movie].iloc[0]

    recommendations = movies[
        (movies["Genre"] == selected_movie["Genre"]) &
        (movies["Language"] == selected_movie["Language"]) &
        (movies["Movie Name"] != movie)
    ].head(5)

    st.subheader("Recommended Movies")

    for m in recommendations["Movie Name"]:
        st.write("👉", m)