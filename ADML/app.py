from flask import Flask, request

import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and the scaled dataframe
with open('movie_recommendation_model.pkl', 'rb') as file:
    kmeans_model, scaled_df, original_df = pickle.load(file)

@app.route("/")
def hello_world():
    
    return f'''
        <h1>Welcome to Advanced ML!</h1>
        <form action="/recommend" method="get">
            <label for="movie">Enter a movie name:</label>
            <input type="text" id="movie" name="movie">
            <input type="submit" value="Get Recommendations">
        </form>
    '''

@app.route("/recommend", methods=["GET"])
def recommend():
    movie_name = request.args.get('movie')
    if not movie_name:
        return "<p>Please provide a movie name.</p>"

    if movie_name not in original_df['Title'].values:
        return "<p>Movie not found in the dataset.</p>"

    # Get the index of the provided movie
    movie_index = original_df[original_df['Title'] == movie_name].index[0]

    # Get the cluster of the provided movie
    movie_cluster_label = kmeans_model.labels_[movie_index]

    # Get all movies in the same cluster
    similar_movies_indices = [index for index, label in enumerate(kmeans_model.labels_) if label == movie_cluster_label]

    # Exclude the provided movie from the recommendations
    similar_movies_indices.remove(movie_index)

    # Get the top 5 recommendations
    recommended_movies = original_df.iloc[similar_movies_indices]['Title'].head(5).tolist()

    # Generate HTML response
    response = f"<h1>Recommendations for {movie_name}</h1><ul>"
    for movie in recommended_movies:
        response += f"<li>{movie}</li>"
    response += "</ul>"

    return response

if __name__ == "__main__":
    app.run(debug=True)