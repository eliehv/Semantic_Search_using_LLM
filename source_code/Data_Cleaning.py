################################################################################
###                         Required Modules
################################################################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import string
import re
from collections import Counter
import os
import spacy
import pathlib
from transformers import AutoTokenizer, AutoModel
import torch

################################################################################
###                load the dataset from Google drive
################################################################################
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/wiki_movie.csv')


################################################################################
###                         Data Cleaning
################################################################################
titles = df.iloc[:,0]
print(f'titles shape : {titles.shape}')

#***************** Drop Duplicate Titles **********************
df_duplicated = df[df.duplicated(subset=['Title'],keep=False)]
print(f'number of duplications : {df_duplicated.shape}')

#*** select rows based on the column value ****
print(f'number of row before drop title duplicates : {df.shape}')
df.drop_duplicates(subset = ['Title'], inplace= True, keep = 'first')
print(f'number of row after drop title duplicates : {df.shape}')
#*******************************************************************

#**** add  number of items to the dataframe **************
df['Number'] = df.reset_index().index
#print(df.index)
#***** reset the index to default index range from 0 to the number of rows
df.reset_index(inplace=True)
#**** rearrange the order of columns ************
df = df.reindex(columns= ['Number','Title','Genre','Origin','Director','ReleaseYear','Cast','Plot','Wiki Page'])


# **** define Title as index to be able to search in data based on Title as unique value for each item
df.set_index(['Title'], inplace= True)
#print(df.index)
#print(df.loc['Alice in Wonderland',:]) # we can search based on specific title when we set it as index
#******* find the columns with NAN values *****
print(df.columns[df.isna().any()].tolist())
print(f'number of NAN values per column: {df.isna().sum()}')
print(f'number of NAN values in Release Year: {df["ReleaseYear"].isna().sum()}')
print(f'number of not numberic values in RElease year: {df[~np.isreal (df["ReleaseYear"].values)].shape[0]}')
print('\n---------------------- \n')

#***** Drop Null value ***************
#df.dropna(subset = ['Cast'], inplace= True)
#print(f'number of row after Cast dropna : {df.shape}')

#************************************************
print(f'unique Genre names : {df["Genre"].unique()}')
print(f'number of unique Genre names : {len(df["Genre"].unique())}')
print(f'unknown values in Genre field: {df[df["Genre"] == "unknown"].shape[0]}')
#*********************************************

#***** Drop the rows w that have NAN Cast and 'unknown' Genre ***********
#df[(df['Genre'] == 'unknown') & (pd.isna(df['Cast']))]
print(f'df shape:{df.shape }')
df.drop(df[(df['Genre'] == 'unknown') & (pd.isna(df['Cast']))].index , inplace= True)
print(f'df shape after dropped Cast Nan and Genre unknown at the same time {df.shape}')


#**************************** Clean Genre field **********************************
nlp = spacy.load('en_core_web_sm')
punctuations = string.punctuation
def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc)# nlp(doc,disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-' and not tok.is_stop and not tok.like_num and  not tok.is_punct] #
        #tokens = [tok for tok in tokens if  tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


# apply clean_genre() on Genre column of all rows in df
#   ******  test by an example ******
#x = df[df['Genre'] == 'action/comedy']['Genre'].values #'16\xa0mm film' #'[140]'
x = df[:]['Genre'].values # all rows
print(x)
clean_Genre = cleanup_text(x)

df = df.assign(Clean_Genre = clean_Genre.values)

x_clean = clean_Genre.values
x_clean = ' '.join(x_clean).split()
print(x_clean)

#****** Plot the most common Genres **********************
x_counter = Counter(x_clean)
x_common_words = [word[0] for word in x_counter.most_common(20)]
x_common_counts = [word[1] for word in x_counter.most_common(20)]
fig = plt.figure(figsize=(18,6))
sns.barplot(x=x_common_words, y=x_common_counts)
plt.title('Most Common movie genres the dataset')
plt.show()
#****************************************
# modifying the Geners that are not correctly annotated to Unknown
df['Clean_Genre']= df['Clean_Genre'].replace(['', ' '], 'unknown')
df['Clean_Genre']= df['Clean_Genre'].replace(['serial'], 'unknown')

#  the Genres that belong to the same category but called differently, should be changed to the same name 
#sci fi science fiction
df['Clean_Genre'] = df['Clean_Genre'].replace('sci fi','scienceFiction')
df['Clean_Genre'] = df['Clean_Genre'].replace('science fiction','scienceFiction')
df['Clean_Genre'] = df['Clean_Genre'].replace('sci-fi','scienceFiction')

# **** anime animate animation **** these are not genre at all ****
df['Clean_Genre'] = df['Clean_Genre'].replace(['anime', 'animate', 'animation'],'unknown')

# **** war drama , world war ii *****
df['Clean_Genre'] = df['Clean_Genre'].replace(['world war ii', 'war drama'],'war')

# **** romantic , romance *****
df['Clean_Genre'] = df['Clean_Genre'].replace(['romantic'],'romance')

#***** in this kind of replacement only the items that has one genre will be modified --> we should do same for list of them
# *** make separate genres instead of 'drama comedy' --> ['drama' , 'comedy'] *********
df['List_Genre'] = df['Clean_Genre'].apply(lambda x: x if pd.isna(x) else x.split())

#****** create unique name for similar genres --- like sci fi, science fiction, sci-fi  *****************
def replace_genre_unique(x, pat, label):
  temp = []
  for genre in x:
    if(re.search(pat,genre)):
      temp.append(label)
    else:
      temp.append(genre)
  return temp


#sci fi science fiction
df['List_Genre'] = df['List_Genre'].apply(lambda x:  replace_genre_unique(x,'^sci.*|.*fi$','scienceFiction'))
df['List_Genre'] = df['List_Genre'].apply(lambda x:  replace_genre_unique(x,'^anim.*','unknown'))
df['List_Genre'] = df['List_Genre'].apply(lambda x:  replace_genre_unique(x,'^romantic.*','romance'))
#df['List_Genre'].apply(lambda x: print(x))
