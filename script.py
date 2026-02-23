from operator import index
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
def recommed(movie_title, num_recommendations=5):
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

    # Check if movie exists
    if movie_title not in indices:
        print(f"Movie '{movie_title}' not found")
        return

    # Get recommendations
    indx = indices[movie_title]
    sim_scores = list(enumerate(similarity[indx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies['title'].iloc[movie_indices]

    print(f"\n Because you liked '{movie_title}', we recommend:")
    for i, movie in enumerate(recommendations, 1):
        print(f"  {i}. {movie}")


# Test
recommed("Toy Story (1995)")
recommed("James Bond")
recommed("GoodFellas (1990)")
recommed("Jumanji (1995)")

