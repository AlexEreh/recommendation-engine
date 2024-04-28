import pandas as pd
from pandas import DataFrame
from rapidfuzz import process
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def get_collaborative_knn_recs(movie_names: DataFrame, ratings_data: DataFrame, movie_name: str):
    movie_names = movie_names[['title', 'genres']]
    movie_data = pd.concat([ratings_data, movie_names], axis=1)
    trend = pd.DataFrame(movie_data.groupby('title')['rating'].mean())
    trend['total number of ratings'] = pd.DataFrame(movie_data.groupby('title')['rating'].count())
    movies_users = ratings_data.pivot(index=['userId'], columns=['movieId'], values='rating').fillna(0)
    mat_movies_users = csr_matrix(movies_users.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=30, n_jobs=-1)
    model_knn.fit(mat_movies_users)
    movie_index = process.extractOne(movie_name, movie_names['title'])[2]
    distances, indices = model_knn.kneighbors(mat_movies_users[movie_index], n_neighbors=20)
    recc_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    recommend_frame = []
    for val in recc_movie_indices:
        recommend_frame.append({'Title': movie_names['title'][val[0]], 'Distance': val[1]})
    df = pd.DataFrame(recommend_frame, index=range(1, 20))
    return df
