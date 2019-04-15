'''
movie recommend

https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7
'''

import numpy as np
import pandas as pd
import json
import time
import os
import pickle


def parse_genres(genres_str):
    genres = json.loads(genres_str.replace('\'', '"'))
    genres_list=[]
    for g in genres:
        genres_list.append(g['name'])
    return genres_list



meta = pd.read_csv('dataset/movies_metadata.csv')
print(meta.head())
meta=meta[['id', 'original_title', 'original_language', 'genres']]
meta=meta.rename(columns={'id':'movieId'})
meta=meta[meta['original_language']=='en']
meta['genres']=meta['genres'].apply(parse_genres)
meta.movieId = pd.to_numeric(meta.movieId, errors='coerce')
print(meta.head())


if os.path.exists('dataset/rating.pkl'):
    with open('dataset/rating.pkl', 'rb') as fp:
        matrix = pickle.load(fp)
else:
    # ratings = pd.read_csv('dataset/ratings_small.csv')
    ratings = pd.read_csv('dataset/ratings.csv')
    ratings = ratings[['userId', 'movieId', 'rating']]
    ratings.movieId = pd.to_numeric(ratings.movieId, errors='coerce')
    print(ratings.head())
    print( ratings.describe() )

    data = pd.merge(ratings, meta, on='movieId', how='inner')
    matrix = data.pivot_table(index='userId', columns='original_title', values='rating')
    with open('dataset/rating.pkl', 'wb') as fp:
        pickle.dump(matrix, fp)


print(matrix.head(20))

GENRE_WEIGHT=0.1

def pearsonR(s1, s2):
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    tmp2 = np.sqrt(np.sum(s1_c**2)*np.sum(s2_c**2))
    cor = np.sum(s1_c* s2_c) / tmp2
    return cor

def recommend(input_movie, matrix, n, similar_genre=True):
    input_genres = meta[meta['original_title']==input_movie]['genres']
    if input_genres.count()==0:
        print('not found movie...')
        return

    input_genres = input_genres.iloc[0]
    print('input_genres', input_genres)
    try:
        print('input movie info=', matrix[input_movie])
    except KeyError:
        print('not found rating...')
        return

    result=[]
    for title in matrix.columns:
        if title==input_movie:
            continue

        cor = pearsonR(matrix[input_movie], matrix[title])
        if similar_genre and len(input_genres)>0 :
            temp_genres = meta[meta['original_title']==title]['genres'].iloc[0]
            same_count = np.sum(np.isin(input_genres, temp_genres))
            cor+= (GENRE_WEIGHT*same_count)

        if np.isnan(cor):
            continue
        else:
            result.append((title, '{:.2f}'.format(cor), temp_genres))
    result.sort(key=lambda r: r[1], reverse=True)
    return result[:n]


print('predict')

# input_movie = 'Zodiac'
# input_movie = 'Heat'
# input_movie = 'Toy Story'
input_movie = 'Sleepness in Seattle'
print('genres=', meta[meta['original_title']==input_movie]['genres'])
recommend_result = recommend(input_movie, matrix, 10, similar_genre=True)
# recommend movie
print( pd.DataFrame(recommend_result, columns=['Title', 'Correlation', 'Genres']) )




