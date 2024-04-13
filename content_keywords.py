import pandas as pd
from pandas import DataFrame
from ast import literal_eval
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    # Return empty list in case of missing/malformed data
    return []


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, metadata, indices, cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]


def create_soup(x):
    return ' '.join(str(x['keywords'])) + ' ' + ' '.join(str(x['cast'])) + ' ' + str(x['director']) + ' ' + ' '.join(str(x['genres']))


def get_content_keywords_recs(
        title: str
) -> DataFrame:
    metadata = pd.read_csv('dataset/movies_metadata.csv', low_memory=False)
    credits = pd.read_csv('dataset/credits.csv')
    keywords = pd.read_csv('dataset/keywords.csv')
    metadata = metadata.drop([19730, 29503, 35587])
    # Convert IDs to int. Required for merging
    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')
    metadata['id'] = metadata['id'].astype('int')
    # Merge keywords and credits into your main metadata dataframe
    metadata = metadata.merge(credits, on='id')
    metadata = metadata.merge(keywords, on='id')

    features = ['cast', 'crew', 'keywords', 'genres']

    for feature in features:
        metadata[feature] = metadata[feature].apply(literal_eval)
    metadata['director'] = metadata['crew'].apply(get_director)
    for feature in features:
        metadata[feature] = metadata[feature].apply(get_list)
    for feature in features:
        metadata[feature] = metadata[feature].apply(clean_data)
    # Create a new soup feature
    metadata['soup'] = metadata.apply(create_soup, axis=1)
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(metadata['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    metadata = metadata.reset_index()
    indices = pd.Series(metadata.index, index=metadata['title'])
    return get_recommendations(title, metadata, indices, cosine_sim)
