import tensorflow as tf
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from surprise import accuracy
import random

# print(tf.__version__)

# Load the movie and ratings data from CSV files into pandas DataFrames.
movies = pd.read_csv('movies.csv', nrows=20000, low_memory=False)
print("Movies file loaded successfully")
ratings = pd.read_csv('ratings.csv', low_memory=False)
print("Rating file loaded successfully")
# makes dtypes int64 or float64
movies['id'] = pd.to_numeric(movies['id'], errors='coerce', downcast='integer')

# debugging
# Basic overview of the movies and ratings data
# print(movies.info())   # Movies dataset structure
# print(ratings.info())  # Ratings dataset structure
# Basic statistical summary of the numerical columns
# print(movies.describe())
# print(ratings.describe())
# (ratings.dtypes)

# sample_movies = pd.read_csv('movies.csv', nrows=1000)
# print(sample_movies.dtypes)
# print(len(ratings.rating))
# print(f"Number of unique users: {ratings['userId'].nunique()}")
# (f"Number of unique movies: {ratings['movieId'].nunique()}")


# Data Cleaning

# REMOVE DUPLICATES
movies.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)

# Check for missing values
movies.isnull().sum()
ratings.isnull().sum()

# Missing values
# Removes any duplicate rows in both tables if they exist, resulting in cleaner data.
movies.dropna(subset=['id', 'title'], inplace=True)
ratings.drop_duplicates(subset=['userId', 'movieId'], keep='first', inplace=True)
ratings.dropna(subset=['movieId', 'rating', 'userId'], inplace=True)
ratings.dropna(subset=['rating'], inplace=True)  # Drop rows where 'rating' is NaN

# FORMATTING
movies['id'] = movies['id'].astype(int)
ratings['movieId'] = ratings['movieId'].astype(int)
ratings['userId'] = ratings['userId'].astype(int)
ratings['rating'] = ratings['rating'].astype(float)

# Split the genre string into a list of individual genres
movies['genres'] = movies['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])

# Filter ratings outside the expected range (1-5) & Remove any rows where rating values are outside the range [1, 5].
ratings = ratings[(ratings['rating'] >= 1) & (ratings['rating'] <= 5)]

# Pivot the ratings to create a user-item matrix
# user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Convert ratings data into Surprise format
# Define the rating scale (1 to 5) and load the DataFrame into a format compatible with the Surprise library.
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

#  Initialize the SVD model and train it using the training set
print("Starting model training...")
model = SVD()
print("Dataset type:", type(data))
print("Trainset type:", type(trainset))
print(ratings[['userId', 'movieId', 'rating']].head())
print(ratings.dtypes)
print("loading please give time for data to process")
model.fit(trainset)


# a function to recommend movies to a specified user based on the SVD model’s predictions.
def recommend_movies(user_id, model, movies_df, ratings_df, n_recommendations=10):
    """
    Recommends movies for a given user based on the SVD model.

    Parameters:
    - user_id (int): The ID of the user for whom to make recommendations.
    - model (SVD): Trained SVD model.
    - movies_df (DataFrame): DataFrame containing movie metadata.
    - ratings_df (DataFrame): DataFrame containing user ratings.
    - n_recommendations (int): Number of recommendations to return.

    Returns:
    - list of movie titles recommended for the user.
    """
    # Get all unique movie IDs
    all_movie_ids = movies_df['movieId'].unique()
    print("Unique movie id collected")
    # Find movies that the user has already rated
    user_rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
    print("loaded movies user rated")
    # Filter out movies the user has already rated
    movies_to_predict = [movie_id for movie_id in all_movie_ids if movie_id not in user_rated_movie_ids]
    print("Rated movies removed from predictions")
    # Predict ratings for the movies the user hasn't rated
    predictions = [model.predict(user_id, movie_id) for movie_id in movies_to_predict]
    print(type(movies_df['movieId']))
    print(predictions)
    # Sort predictions by estimated rating in descending order
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:n_recommendations]
    #print(recommendations)
    # Get the top N recommended movie IDs
    recommended_movie_ids = [pred.iid for pred in recommendations]
    #print(recommended_movie_ids)

    # Return the movie titles for the top recommendations
    return movies_df[movies_df['movieId'].isin(recommended_movie_ids)]['title'].values


# Step 5: Get Recommendations for a User

# Specify a user ID (use any user ID present in the dataset)
user_id = 1000  # For example
# print("Movies: " + movies + "\n userID: " + user_id)
top_recommendations = recommend_movies(user_id, model, movies, ratings)
print("Top recommendations for user", user_id, ":", top_recommendations)

# Step 6: Model Evaluation

# Predict ratings for the test set
predictions = model.test(testset)

#  Calculate the Root Mean Squared Error (RMSE), a measure of how accurately the model predicts ratings.
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')


