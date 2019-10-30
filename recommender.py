"""
Movie Recommender System

Utilizes Collaboratorive Filtering to provide you movie recommendations.

Pandas, Sci-Kit Learn, postgreSQL, fuzzywuzzy
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import NMF
from sqlalchemy import create_engine, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fuzzywuzzy import process
import os

user = 'postgres'
pwd = os.getenv('PG_PWD')
db_name = 'rec_project'

db = f'postgres://{user}:{pwd}@localhost/{db_name}'
engine = create_engine(db, encoding='latin1')
base = declarative_base(engine)
Session = sessionmaker(engine)
session = Session()

metadata = base.metadata
ratings = Table('ratings', metadata, autoload=True)
movies = Table('movies', metadata, autoload=True)
tags = Table('tags', metadata, autoload=True)
umr = Table('userr', metadata, autoload=True)
usmore = pd.read_sql_table('userr', db)
dfmovies = pd.read_sql_table('movies', db)
usmore.index = usmore['movieId']
del usmore['movieId']

def main():
    train_nmf()
    print("Welcome to my Movie Recommender")
    movie_in_1 = input("Please provide a movie title: ")
    movie_in_2 = input("Please provide another movie title: ")
    movie_in_3 = input("Please provide another movie title: ")
    user_input = {'1':movie_in_1, '2':movie_in_2, '3':movie_in_3}
    mlist, movid = fuzz_lookup(user_input)
    print(mlist)
    result = recommendations(mlist, movid)
    print()
    print("Here are your recommendations: ")
    print()
    print(result.head(5))
    
def train_nmf():
    m = NMF(n_components=2, init='nndsvd', random_state=42, alpha=0)
    m.fit(usmore)
    nmfpickle = open("nmf.pkl", 'wb')
    pickle.dump(m, nmfpickle)
    nmfpickle.close()
    return

def recommendations(mlist, movid):
    picklep = 'nmf.pkl'
    nmf_unpickle = open(picklep, 'rb')   
    m = pickle.load(nmf_unpickle)
    userslist = pd.Series(
            np.zeros(usmore.shape[0]),
            index=usmore.index)
    #print(userslist)
    for i in movid:
        userslist[i] = 5
    query = m.transform(usmore)
    profile = np.dot(userslist, query)
    ranking = np.dot(profile, query.T)
    ranking = pd.Series(ranking,
                        index=usmore.index)
    result = pd.DataFrame({'title': dfmovies['title'], 'rank': ranking})
    result.sort_values('rank', ascending=False, inplace=True)
    result = result['title'].iloc[0:5]
    return result

def fuzz_lookup(user_input):
    movietitle = dfmovies['title'].tolist()
    movie1 = user_input['1']
    movie2 = user_input['2']
    movie3 = user_input['3']
    movie1_clean = process.extractOne(movie1, movietitle)[0]
    movie2_clean = process.extractOne(movie2, movietitle)[0]
    movie3_clean = process.extractOne(movie3, movietitle)[0]
    mlist = [movie1_clean, movie2_clean, movie3_clean]
    movid = []
    for ml in mlist:
        mi = dfmovies['movieId'][dfmovies['title'] == ml]
        movid.append(mi)
    #print(mlist, movid)
    return mlist, movid


if __name__ == '__main__':
    main()