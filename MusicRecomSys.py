import pandas
from sklearn.model_selection import train_test_split
import numpy as np
import Recommenderclass


#This step might take time to download data from external sources
triplets_file = open (r'C:\Users\Roya Rafati\Desktop\RoyaRafati_Music Recomandation System\Project\PY Code\10000.txt')
song_df_2 = pandas.read_csv (r'C:\Users\Roya Rafati\Desktop\RoyaRafati_Music Recomandation System\Project\PY Code\song_data.csv')

song_df_1 = pandas.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

#Read song  metadata
#song_df_2 =  pandas.read_csv(songs_metadata_file)

#Merge the two dataframes above to create input dataframe for recommender systems
song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

#creat a subset of data set:
#song_df = song_df.head()
#print(song_df)
#print(len(song_df))

#Most popular Songs : 
song_grouped = song_df.groupby(['title']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'title'], ascending = [0,1])


#Count number of unique users :
users = song_df['user_id'].unique()
#print(len(users))

#Count number of unique songs :
songs = song_df['title'].unique()
#print(len(songs))

#Creat a Recommander :
train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)

pm = Recommenderclass.popularity_recommender_py()
pm.create(train_data, 'user_id', 'title')
#user the popularity model to make some prediction
user_id = users[5]
result=pm.recommend(user_id)
print(result)

#----------------------------------------------------
#item similarity model :

is_model = Recommenderclass.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'title')



#Print the songs for the user in training data
user_id = users[5]
user_items = is_model.get_user_items(user_id)
#
print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
is_model.recommend(user_id)