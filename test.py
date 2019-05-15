
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sklearn.metrics.pairwise as pw

from data import BUSINESSES, CITIES

def dfmakercategories():
    businesslist = list()
    categorieslist = list()

    for city in CITIES:
        for i in BUSINESSES[city]:
            data = json.loads(json.dumps(i))
            categories = data["categories"]
            for j in list(categories.split(",")):
                    newj = j.replace(" ", "")
                    businesslist.append(data['business_id'])
                    categorieslist.append(newj)

    series_ratings = pd.DataFrame(columns=['business_id', 'categories'])
    series_ratings['business_id'] = businesslist
    series_ratings['categories'] = categorieslist
    return series_ratings


def dfmakerratings():
    # moet nog aangepast worden!
    businesslist = list()
    categorieslist = list()

    for city in CITIES:
        for i in BUSINESSES[city]:
            data = json.loads(json.dumps(i))
            categories = data["categories"]
            for j in list(categories.split(",")):
                    newj = j.replace(" ", "")
                    businesslist.append(data['business_id'])
                    categorieslist.append(newj)

    series_ratings = pd.DataFrame(columns=['business_id', 'categories'])
    series_ratings['business_id'] = businesslist
    series_ratings['categories'] = categorieslist
    return series_ratings

def pivot_genres(df):
    """Create a one-hot encoded matrix for genres.
    
    Output:
    a matrix containing '0' or '1' in each cell.
    1: the movie has the genre
    0: the movie does not have the genre
    """
    return df.pivot_table(index = 'business_id', columns = 'categories', aggfunc = 'size', fill_value=0)

def create_similarity_matrix_categories(matrix):
    """Create a  """
    npu = matrix.values
    m1 = npu @ npu.T
    diag = np.diag(m1)
    m2 = m1 / diag
    m3 = np.minimum(m2, m2.T)
    return pd.DataFrame(m3, index = matrix.index, columns = matrix.index)

def predict_ratings(similarity, utility, to_predict):
    """Predicts the predicted rating for the input test data.
    
    Arguments:
    similarity -- a dataFrame that describes the similarity between items
    utility    -- a dataFrame that contains a rating for each user (columns) and each movie (rows). 
                  If a user did not rate an item the value np.nan is assumed. 
    to_predict -- A dataFrame containing at least the columns movieId and userId for which to do the predictions
    """
    # copy input (don't overwrite)
    ratings_test_c = to_predict.copy()
    # apply prediction to each row
    ratings_test_c['predicted rating'] = to_predict.apply(lambda row: predict_ids(similarity, utility, row['userId'], row['movieId']), axis=1)
    return ratings_test_c

### Helper functions for predict_ratings_item_based ###

def predict_ids(similarity, utility, userId, itemId):
    # select right series from matrices and compute
    if userId in utility.columns and itemId in similarity.index:
        return predict_vectors(utility.loc[:,userId], similarity[itemId])
    return 0

def predict_vectors(user_ratings, similarities):
    # select only movies actually rated by user
    relevant_ratings = user_ratings.dropna()
    
    # select corresponding similairties
    similarities_s = similarities[relevant_ratings.index]
    
    # select neighborhood
    similarities_s = similarities_s[similarities_s > 0.0]
    relevant_ratings = relevant_ratings[similarities_s.index]
    
    # if there's nothing left return a prediction of 0
    norm = similarities_s.sum()
    if(norm == 0):
        return 0
    
    # compute a weighted average (i.e. neighborhood is all) 
    return np.dot(relevant_ratings, similarities_s)/norm

def mse(predicted_ratings):
    """Computes the mean square error between actual ratings and predicted ratings
    
    Arguments:
    predicted_ratings -- a dataFrame containing the columns rating and predicted rating
    """
    diff = predicted_ratings['rating'] - predicted_ratings['predicted rating']
    return (diff**2).mean()

series_ratings = dfmakercategories()
df_utility_ratings = dfmakerratings()

df_utility_genres = pivot_genres(series_ratings)
df_similarity_genres = create_similarity_matrix_categories(df_utility_genres)
print(df_similarity_genres.head())
predicted_genres = predict_ratings(df_similarity_genres, df_utility_ratings, df_ratings_test[['userId', 'movieId', 'rating']])
mse_top_rated_content_based = mse(predicted_genres[predicted_genres['predicted rating'] > 4.5])
