
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sklearn.metrics.pairwise as pw

from data import BUSINESSES, CITIES, REVIEWS, USERS

def dfmaker():
    businesslist = list()
    categorieslist = list()
    starlist = list()
    useridlist = list()
    for city in CITIES:
        for review in REVIEWS[city]:
            useridlist.append(review['user_id'])
            starlist.append(review['stars'])
            businesslist.append(review['business_id'])
            for city in CITIES:
                for business in BUSINESSES[city]:
                    if business['business_id'] == review['business_id']:
                        data = json.loads(json.dumps(business))
                        categories = data["categories"]
                        categorieslist.append(categories)
                    
    df_ratings = pd.DataFrame(columns=['user_id', 'business_id', 'rating', 'categories'])
    df_ratings['user_id'] = useridlist
    df_ratings['business_id'] = businesslist
    df_ratings['categories'] = categorieslist
    df_ratings['rating'] = starlist 
    # df_ratings = df_ratings.sort_values(by='user_id')
    # df_ratings = df_ratings.groupby('user_id')
    # print(df_ratings)
    return df_ratings


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
    businesslist = list()
    categorieslist = list()
    starlist = list()
    useridlist = list()
    for city in CITIES:
        for review in REVIEWS[city]:
            useridlist.append(review['user_id'])
            starlist.append(review['stars'])
            businesslist.append(review['business_id'])
            # for city in CITIES:
            #     for business in BUSINESSES[city]:
            #         if business['business_id'] == review['business_id']:
            #             data = json.loads(json.dumps(business))
            #             categories = data["categories"]
            #             categorieslist.append(categories)
                    
    df_ratings = pd.DataFrame(columns=['user_id', 'business_id', 'rating'])
    df_ratings['user_id'] = useridlist
    df_ratings['business_id'] = businesslist
    df_ratings['rating'] = starlist 
    # df_ratings = df_ratings.sort_values(by='user_id')
    # df_ratings = df_ratings.drop_duplicates()
    # print(df_ratings)
    return df_ratings

def pivot_genres(df):
    """Create a one-hot encoded matrix for genres.
    
    Output:
    a matrix containing '0' or '1' in each cell.
    1: the movie has the genre
    0: the movie does not have the genre
    """
    return df.pivot_table(index = 'business_id', columns = 'categories', aggfunc = 'size', fill_value=0)

def pivot_ratings(df):
    """Creates a utility matrix for user ratings for movies
    
    Arguments:
    df -- a dataFrame containing at least the columns 'movieId' and 'genres'
    
    Output:
    a matrix containing a rating in each cell. np.nan means that the user did not rate the movie
    """
    return df.pivot_table(values='rating', columns='user_id', index='business_id')

def create_similarity_matrix_categories(matrix):
    """Create a  """
    npu = matrix.values
    m1 = npu @ npu.T
    diag = np.diag(m1)
    m2 = m1 / diag
    m3 = np.minimum(m2, m2.T)
    # print(pd.DataFrame(m3, index = matrix.index, columns = matrix.index))
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
    ratings_test_c['predicted rating'] = to_predict.apply(lambda row: predict_ids(similarity, utility, row['user_id'], row['business_id']), axis=1)
    # print(ratings_test_c)
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
    # print("MSE:", diff**2)
    return (diff**2).mean()

def split_data(data, d = 0.75):
    """Split data in a training and test set.
    
    Arguments:
    data -- any dataFrame.
    d    -- the fraction of data in the training set
    """
    np.random.seed(seed=5)
    mask_test = np.random.rand(data.shape[0]) < d
    return data[mask_test], data[~mask_test]

def dfmakersuited(city, user_id):
    businesslist = list()
    categorieslist = list()
    starlist = list()
    useridlist = list()
    
    for review in REVIEWS[city]:
        if review['user_id'] == user_id:
            useridlist.append(review['user_id'])
            starlist.append(review['stars'])
            businesslist.append(review['business_id'])

        
    for business in BUSINESSES[city]:
        if business['business_id'] not in businesslist:
            businesslist.append(business['business_id'])
            starlist.append(np.nan)
            useridlist.append(user_id)
                    
    df_ratings = pd.DataFrame(columns=['user_id', 'business_id', 'rating'])
    df_ratings['user_id'] = useridlist
    df_ratings['business_id'] = businesslist
    df_ratings['rating'] = starlist 
    # df_ratings = df_ratings.sort_values(by='user_id')
    # df_ratings = df_ratings.groupby('user_id')
    # print(df_ratings)
    return df_ratings


# new test
dfratings = dfmakersuited("sun city", "DAIpUGIsY71noX0wNuc27w") 
dfutility = pivot_genres(dfmakercategories())
dfsimilarity = create_similarity_matrix_categories(dfutility)
dfutilityratings = pivot_ratings(dfratings)
predicted_genres = predict_ratings(dfsimilarity, dfutilityratings, dfratings[['user_id', 'business_id', 'rating']])
print(mse(predicted_genres))
# old test
# dfutility = pivot_genres(dfmakercategories())
# dfsimilarity = create_similarity_matrix_categories(dfutility)
# dfutilityratings = pivot_ratings(dfmakerratings())

# df_ratings_training, df_ratings_test = split_data(dfmaker(), d=0.9)
# df = dfmaker()





# print(dfutility)
# print(df)

# predicted_genres = predict_ratings(dfsimilarity, dfutilityratings, df[['user_id', 'business_id', 'rating']])
# mse_top_rated_content_based = mse(predicted_genres[predicted_genres['predicted rating'] > 0.1])


# df_ratings_test_copy = df_ratings_test.copy()
# df_ratings_test_copy['predicted rating'] = np.random.uniform(low=0.5, high=5, size=(len(df_ratings_test),))

# mse_random = mse(df_ratings_test_copy)
# print(f'mse for random prediction: {mse_random:.2f}')
