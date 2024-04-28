import pandas as pd
from rapidfuzz import process
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def get_collaborative_knn_recs(movie_names: pd.DataFrame, ratings_data: pd.DataFrame, movie_name: str):
    # Дата фрейм с названием фильма и его жанром
    movie_names = movie_names[['title', 'genres']]
    # Дата фрейм, в котором
    movies_users: pd.DataFrame = ratings_data.pivot(index=['userId'], columns=['movieId'], values='rating').fillna(0)
    # Преобразовываем в разреженную матрицу (CSR)
    mat_movies_users: csr_matrix = csr_matrix(movies_users.values)
    model_knn: NearestNeighbors = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=30, n_jobs=-1)
    model_knn.fit(mat_movies_users)
    movie_index: int = process.extractOne(movie_name, movie_names['title'])[2]
    distances, indices = model_knn.kneighbors(mat_movies_users[movie_index], n_neighbors=20)
    recc_movie_indices: list = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                                      key=lambda x: x[1])[:0:-1]
    # Список с рекомендациями
    recommend_list = []
    # На каждый индекс рекомендаций
    for val in recc_movie_indices:
        # Добавляем в датафрейм рекомендаций названий фильма и расстояние
        recommend_list.append({'Title': movie_names['title'][val[0]], 'Distance': val[1]})
    # Датафрейм с рекомендациями
    df = pd.DataFrame(recommend_list, index=range(1, 20))
    return df
