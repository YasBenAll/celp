
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sklearn.metrics.pairwise as pw

from data import BUSINESSES, CITIES

def dfmaker():
    # r=root, d=directories, f = files
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

series_ratings = dfmaker()

df_utility_genres = pivot_genres(series_ratings)
df_similarity_genres = create_similarity_matrix_categories(df_utility_genres)
print(df_similarity_genres.head())