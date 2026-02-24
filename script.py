import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Setting the API
API_KEY = "your api"
BASE_URL = "your base url"
# Setting the data
movies = pd.read_csv("movie.csv")
print(f"movies loaded: {movies.shape}")

# Setting the volum for the movies
movies['genres'] = movies['genres'].fillna('')

# Converting gnereas to numbers
trfidf = TfidfVectorizer(stop_words='english')
trfidf_martix = trfidf.fit_transform(movies['genres'])

# Calculating teh similarity of the movies
similarity = cosine_similarity(trfidf_martix, trfidf_martix)
print("Similarity matrix created!")
# Building the recommendation
def search_movie(movie_title):
    url = f"{BASE_URL}/search/movie"
    params = {"api_key": API_KEY, "query": movie_title}
    response = requests.get(url, params=params)
    results = response.json().get("results", [])

    if not results:
        print(f"Movie '{movie_title}' not found!")
        return None

    movie = results[0]
    print(f"Found: {movie['title']} ({movie.get('release_date', 'N/A')[:4]})")
    return movie

# Step 2: Get recommendations
def recommend(movie_title, num_recommendations=5):
    movie = search_movie(movie_title)
    if not movie:
        return

    movie_id = movie["id"]

    url = f"{BASE_URL}/movie/{movie_id}/recommendations"
    params = {"api_key": API_KEY}
    response = requests.get(url, params=params)
    results = response.json().get("results", [])

    if not results:
        print(f"No recommendations found for '{movie_title}'")
        return

    print(f"\n Because you liked '{movie_title}', we recommend:")
    for i, m in enumerate(results[:num_recommendations], 1):
        year = m.get("release_date", "N/A")[:4]
        rating = m.get("vote_average", "N/A")
        print(f"  {i}. {m['title']} ({year}) ⭐ {rating}/10")

# Step 3: Test it!
recommend("Toy Story")
recommend("The Dark Knight")
recommend("Parasite")
recommend("Inception")
recommend("Recep Ivedik ")

