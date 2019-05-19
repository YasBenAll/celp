from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sklearn.metrics.pairwise as pw
from test import dfmakersuited, pivot_genres, create_similarity_matrix_categories, pivot_ratings, predict_ratings, dfmakercategories
from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS, get_business, load
DATA_DIR = "data"
import random



def recommend(user_id=None, business_id=None, city=None, n=10):
    """
    Returns n recommendations as a list of dicts.
    Optionally takes in a user_id, business_id and/or city.
    A recommendation is a dictionary in the form of:
        {
            business_id:str
            stars:str
            name:str
            city:str
            adress:str
        }
    """
    # if not user_id:
    #     user_id = "DAIpUGIsY71noX0wNuc27w"
    if not city:
        city = random.choice(CITIES)
    # if not business_id:
    #     business_id = "XieY4CeZOw9bMBk965BNTw"
    # print(user_id)
    
    dfratings = dfmakersuited(city, user_id)
    dfutility = pivot_genres(dfmakercategories())
    dfsimilarity = create_similarity_matrix_categories(dfutility)
    dfutilityratings = pivot_ratings(dfratings)
    predicted_genres = predict_ratings(dfsimilarity, dfutilityratings, dfratings[['user_id', 'business_id', 'rating']])
    sortedpredicted = predicted_genres.sort_values(by='predicted rating', ascending = False).iloc[0:n]
    sorteddict = sortedpredicted.set_index('business_id')['predicted rating'].to_dict()
    # print(sorteddict.keys())
    # return random.sample(BUSINESSES[city], n)
    recommendlist = list()
    cities = list()
    cities.append(city)
    for i in sorteddict.keys():
        recommendlist.append(get_business(city, i))
    # print(random.choice(list(load(cities, "business").values())[0])["business_id"])
    recommendlist.append(get_business(city, random.choice(list(load(cities, "business").values())[0])["business_id"]))
    # print(recommendlist)
    # print(predicted_genres)
    return recommendlist

# print(recommend())



