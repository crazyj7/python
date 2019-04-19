'''
movie recommend

https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7
'''

import numpy as np
import pandas as pd
import json
import time
import os, sys
import pickle
# from sklearn.externals import joblib


def parse_genres(genres_str):
    genres = json.loads(genres_str.replace('\'', '"'))
    genres_list=[]
    for g in genres:
        genres_list.append(g['name'])
    return genres_list



meta = pd.read_csv('datasets/movies_metadata.csv')
meta=meta[['id', 'original_title', 'original_language', 'genres']]
meta=meta.rename(columns={'id':'movieId'})
meta=meta[meta['original_language']=='en']
meta['genres']=meta['genres'].apply(parse_genres)
meta.movieId = pd.to_numeric(meta.movieId, errors='coerce')
print(meta.head())


# input_movie = 'Zodiac'
# input_movie = 'Heat'
# input_movie = 'Toy Story'
input_movie = 'The Dark Knight'
# input_movie='fast and the furious'
# input_movie='beauty and the beast'

foundi=-1
for i, x in enumerate(meta['original_title']):
    ot = x.strip().upper()
    if ot.find(input_movie.upper())>=0:
        foundi=i
        print(i,'movieId=', meta['movieId'].values[i], 'title=', x)
        break

if foundi<0:
    print('not found movie')
    sys.exit()
else:
    print('found movie index=', i, 'title=', x, 'len=',len(x))
    input_movie = x

print('genres=', meta[meta['original_title']==input_movie]['genres'])


if meta[meta['original_title']==input_movie]['genres'].count()==0:
    print('not found')
    sys.exit(0)

if os.path.exists('datasets/rating.pkl'):
    print('load matrix...')
    with open('datasets/rating.pkl', 'rb') as fp:
        matrix = pickle.load(fp)
    # matrix = joblib.load('datasets/rating.job')
    print(matrix.head(20))
    # sys.exit()
else:
    ratings = pd.read_csv('datasets/ratings_small.csv')
    # ratings = pd.read_csv('datasets/ratings.csv')
    ratings = ratings[['userId', 'movieId', 'rating']]
    ratings.movieId = pd.to_numeric(ratings.movieId, errors='coerce')
    print(ratings.head())
    print( ratings.describe() )
    print('making matrix...')
    data = pd.merge(ratings, meta, on='movieId', how='inner')
    print('data head')
    print(data.head())
    matrix = data.pivot_table(index='userId', columns='original_title', values='rating')
    print('matrix head')
    print(matrix.head())
    print('save matrix...')
    with open('datasets/rating.pkl', 'wb') as fp:
        pickle.dump(matrix, fp, protocol=4)
    # joblib.dump(matrix, 'datasets/rating.job')


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

print('genres=', meta[meta['original_title']==input_movie]['genres'])
recommend_result = recommend(input_movie, matrix, 10, similar_genre=True)
# recommend movie
print( pd.DataFrame(recommend_result, columns=['Title', 'Correlation', 'Genres']) )




