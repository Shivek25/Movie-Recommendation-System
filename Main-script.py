# %%
import numpy as np
import pandas as pd
import warnings

# %%
column_names = ["uer_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("ml-100k/u.data", sep= '\t', names=column_names)

# %%
print(df.head)
df['uer_id'].nunique()

# %%
movies_titles = pd.read_csv("ml-100k/u.item", sep='\|',header = None)
movies_titles.shape

# %%
movies_titles = movies_titles[[0,1]]
movies_titles.columns = ['item_id', 'title']
movies_titles.head

# %%
df = pd.merge(df, movies_titles, on = "item_id")
df

# %%
import seaborn 
import matplotlib as plt

# %%
df.groupby('title').mean()['rating'].sort_values(ascending=False)

# %%
df.groupby('title').count()['rating'].sort_values(ascending=False)

# %%
ratings = pd.DataFrame(df.groupby('title').mean()['rating'])
ratings.head()

# %%
ratings['num of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])
ratings

# %%
movie_mat = df.pivot_table(index="uer_id", columns="title", values="rating")
movie_mat.head()

# %%
ratings.sort_values('num of ratings', ascending=False)

# %%
starwars_user_ratings = movie_mat['Star Wars (1977)']
starwars_user_ratings.head()

# %%
similar_to_starwars = movie_mat.corrwith(starwars_user_ratings)
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()

# %%
corr_starwars.sort_values('Correlation', ascending=False).head(10)

# %%
corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()


# %%
def main_function(movie_name):
    movie_user_rating = movie_mat("movie_name")
    similiar_to_movie = movie_mat.corrwith(movie_user_rating)
        
    corr_movie = pd.DataFrame(similiar_to_movie, columns=['Correlation'])
    corr_movie.dropna(inplace=True)
        
    corr_movie = corr_movie.join(ratings['num of ratings'])
    prediction = corr_movie[corr_movie['num of ratings']>100].sort_values('correlation', ascending=False)
    
    return prediction
prediction = main_function("Hollow Reed (1996)")
prediction.head()

# %%



