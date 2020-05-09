from sklearn.metrics import mean_squared_log_error
from datetime import datetime
import pandas as pd
import numpy as np
import argparse
import pickle
import string
import math
import ast


def one_hot_encode_test(df, attribute, relevant_list):
    for val in relevant_list:
        filterd_val = val.lower().replace('productions', '').replace('pictures', '')
        df[str(val)] = [
            1 if filterd_val in [xx.lower().replace('productions', '').replace('pictures', '') for xx in x] else 0 for x
            in df[str(attribute)].values]
    return df


def get_total_oscars(names_list, winning_dic):
    total_oscars = 0
    for x in names_list:
        x = x.lower()
        if x in winning_dic.keys():
            total_oscars += winning_dic[x]
    return total_oscars


def get_famous(names_list, famous_list):
    total_famous = 0
    for x in names_list:
        x = x.lower()
        if x in famous_list:
            total_famous += 1000 - famous_list.index(x)
    return total_famous


# Load external data sources
mean_budget_decade = pd.read_pickle("mean_budget_decade.pkl")
mean_budget_year = pd.read_pickle("mean_budget_year.pkl")

medians = pd.read_pickle("medians.pkl")
means = pd.read_pickle("means.pkl")
total_movies = pd.read_pickle("total_movies.pkl")
collection_count = pd.read_pickle("collection_count.pkl")

actors_winnings = pd.read_pickle("actors_winnings.pkl")
actors_winnings_dic = {name:win for name,win in zip(actors_winnings['name'].values,actors_winnings['winner'].values)}

actors_nominies = pd.read_pickle("actors_nominies.pkl")
actors_nominies_dic = {name:win for name,win in zip(actors_nominies['name'].values,actors_nominies['winner'].values)}

movies_winnings = pd.read_pickle("movies_winnings.pkl")
movies_nominies = pd.read_pickle("movies_nominies.pkl")

famous_actors_df = pd.read_pickle("famous_actors_df.pkl")
famous_actors = [x.lower() for x in famous_actors_df['Name'].values]

famous_directors_df = pd.read_pickle("famous_directors_df.pkl")
famous_directors = [x.lower() for x in famous_directors_df['Name'].values]

table = str.maketrans(dict.fromkeys(string.punctuation)) #to be used to convert names

genres_list = pickle.load(open("genres_list.pkl", "rb"))
p_companies_list = pickle.load(open("p_companies_list.pkl", "rb"))
p_countries_list = pickle.load(open("p_countries_list.pkl", "rb"))
languages_list = pickle.load(open("languages_list.pkl", "rb"))

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
data = pd.read_csv(args.tsv_path, sep="\t")

# Data cleansing
data = data.fillna(0)
data['release_date'] = [datetime.strptime(x, '%Y-%m-%d').date() for x in data['release_date'].values]
data['belongs_to_collection'] = [eval(x)['name'] if x != 0 else 0 for x in data['belongs_to_collection'].values]
data['genres'] = [[y['name'] for y in ast.literal_eval(x)] if x != 0 else 0 for x in data['genres'].values]
data['production_companies'] = [[y['name'] for y in ast.literal_eval(x)] if x != 0 else 0 for x in data['production_companies'].values]
data['production_countries'] = [[y['iso_3166_1'] for y in ast.literal_eval(x)] if x != 0 else 0 for x in data['production_countries'].values]
data['spoken_languages'] = [[y['iso_639_1'] for y in ast.literal_eval(x)] if x != 0 else 0 for x in data['spoken_languages'].values]
data['Keywords'] = [[y['name'] for y in ast.literal_eval(x)] if x != 0 else 0 for x in data['Keywords'].values]
data['cast'] = [[y['name'] for y in ast.literal_eval(x)] if x != 0 else 0 for x in data['cast'].values]
data['crew'] = [[y['name'] for y in ast.literal_eval(x)] if x != 0 else 0 for x in data['crew'].values]
data['release_year'] = [x.year for x in data['release_date'].values]
data['release_month'] = [x.month for x in data['release_date'].values]
data['release_day'] = [x.day for x in data['release_date'].values]
data['release_day_week'] = [x.weekday() for x in data['release_date'].values]
data['days_diff'] = [(datetime.today().date() - x).days for x in data['release_date'].values]

# Missing values imputation
data['decade'] = [math.floor(x % 10000 / 1000) * 1000 + math.floor(x % 1000 / 100) * 100 + math.floor((x) % 100 / 10) * 10 for x in data['release_year'].values]
data = pd.merge(data, mean_budget_decade, left_on='decade', right_on='decade') # join with decade budget
data = pd.merge(data, mean_budget_year, left_on='release_year', right_on='release_year') # join with decade budget
data['budget'] = [x if x != 0 else (y if y != 0 else z) for x, y, z in zip(data['budget'].values, data['mean_budget_year'].values, data['mean_budget_decade'].values)]

# Extra features
data['genres_count'] = [len(x) for x in data['genres'].values]
data['production_companies_count'] = [len(x) for x in data['production_companies'].values]
data['production_countries_count'] = [len(x) for x in data['production_countries'].values]
data['languages_count'] = [len(x) for x in data['spoken_languages'].values]
data['Keywords_count'] = [len(x) for x in data['Keywords'].values]
data['cast_count'] = [len(x) for x in data['cast'].values]
data['crew_count'] = [len(x) for x in data['crew'].values]

# One hot incode
data = one_hot_encode_test(data, 'genres', genres_list)
data = one_hot_encode_test(data, 'production_companies', p_companies_list)
data = one_hot_encode_test(data, 'production_countries', p_countries_list)
data = one_hot_encode_test(data, 'spoken_languages', languages_list)

# Statistical features
data = pd.merge(data, total_movies, left_on='release_year', right_on='release_year')
data = pd.merge(data, medians, left_on='release_year', right_on='release_year')
data = pd.merge(data, means, left_on='release_year', right_on='release_year')
data = pd.merge(data, collection_count, left_on='belongs_to_collection', right_on='belongs_to_collection')

# External data sources
data['total_actors_oscars'] = [get_total_oscars(x, actors_winnings_dic) for x in data['cast'].values]
data['total_actors_nominies'] = [get_total_oscars(x, actors_nominies_dic) for x in data['cast'].values]
data['original_title'] = [x.lower().translate(table) for x in data['original_title'].values]
data = pd.merge(data, movies_winnings, left_on='original_title', right_on='name', how ='left').fillna(0)
data = pd.merge(data, movies_nominies, left_on='original_title', right_on='name', how ='left').fillna(0)
data = data.drop(columns = ['name_x', 'name_y'])
data['famous_actors'] = [get_famous(x, famous_actors) for x in data['cast'].values] #get famous actors score
data['famous_directors'] = [get_famous(x, famous_directors) for x in data['crew'].values] #get famous directors score
data['famous_crew'] = [get_famous(x, famous_actors) for x in data['crew'].values] #get famous actors (in crew) score

# Define useless columns
bad_columns = ['backdrop_path', 'belongs_to_collection','genres','homepage','id','imdb_id','original_title','poster_path','production_companies',
               'production_countries','spoken_languages','status','tagline','video','Keywords','cast','crew','overview','release_date','title',    'original_language']

# Prepare data for model execution
my_test = data[[x for x in data.columns if x not in bad_columns]]
X_test = my_test[[x for x in my_test.columns if x != 'revenue']]
Y_test = my_test[[ 'revenue']]


# Load trained XGBoost model and predict Revenue
xgboost_clf = pickle.load(open("xgboost_clf.pkl", "rb"))
predictions = [200000 if x < 0 else x for x in xgboost_clf.predict(X_test)] #replace negative predicted values with 200,000 $

# Save results
prediction_df = pd.DataFrame(columns=['id', 'revenue'])
prediction_df['id'] = data['id']
prediction_df['revenue'] = predictions

print(np.sqrt(mean_squared_log_error(Y_test.values,predictions)))

# Export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)
