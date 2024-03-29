# -*- coding: utf-8 -*-
"""Labeling-Method2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O8FBgcT_rJRw3QtvAsEHb0TGYPo_Uy1A
"""

###################################################################################################################
####  # here the is to compare whole chunks of the all top_k results related to the query q with q so,
#       for exmple if we consider 10 best matched chunks and more than 4 of them come from the same movie
#       then that movie would be the label
#       we can consider this with combination of BertScore and CrossEncoder
###################################################################################################################
for q in range(len(query_info)):
  print(query_info[q]['query'])
  final_score = []
  final_index = []
  final_chunks = []
  final_title = []
  final_genre = []
  print(type(query_info[q]['top_k']['chunks'][i]))

  for i in range(len(query_info[q]["top_k"]["CrossEncoder"])):
    for s in query_info[q]['top_k']['CrossEncoder'][i]:
      final_score.append(s)
    for c in query_info[q]['top_k']['chunks'][i]:
      final_chunks.append(c)
      final_title.append(df_new.iloc[int(query_info[q]['top_k']['score_idx'][1][i])]['index'])
      final_genre.append(df_new.iloc[int(query_info[q]['top_k']['score_idx'][1][i])]['Clean_Genre'])
      final_index.append(query_info[q]['top_k']['score_idx'][1][i].item())

  print(f'len score: {len(final_score)}')
  print(f'len chunks: {len(final_chunks)}')
  print(f'len title: {len(final_title)}')
  data = ({'score':final_score,
          'chunk':final_chunks,
            'title':final_title,
            'genre':final_genre,
            'index':final_index})
  df_tmp = pd.DataFrame(data)
  df_tmp.sort_values('score', ascending = True, inplace = True)
  #for index,row in df_tmp.iterrows():
    #print(index, '>>>>  ',row['score'], '>>>>  ', row['title'], '>>>>  ', row['genre'],'>>>>', row['chunk'])

  #df_tmp = df_tmp.iloc[:200]
  print('df shape: ', df_tmp.shape)
  print(df_tmp.groupby("title", sort = True)['title'].count())
  df_tmp_group = df_tmp.groupby("title", sort = True)
  #df_tmp_group = df_tmp_group.apply(lambda x: x['title'].count())
  print(df_tmp_group.filter(lambda x: x['title'].count()> 10)['title'].reset_index().groupby("title", sort = True).count())


  print('\n\n-------------------------------------------\n\n')
  ##################################################
  ###           BertScore
  ##################################################

  final_bscore = []
  final_bindex = []
  final_bchunks = []
  final_btitle = []
  final_bgenre = []

  for i in range(len(query_info[q]["top_k"]["BertScore_F1"])):
    for s in query_info[q]['top_k']['BertScore_F1'][i]:
      final_bscore.append(s)
    for c in query_info[q]['top_k']['chunks'][i]:
      final_bchunks.append(c)
      final_btitle.append(df_new.iloc[int(query_info[q]['top_k']['score_idx'][1][i])]['index'])
      final_bgenre.append(df_new.iloc[int(query_info[q]['top_k']['score_idx'][1][i])]['Clean_Genre'])
      final_bindex.append(query_info[q]['top_k']['score_idx'][1][i].item())
  print(f'len score: {len(final_bscore)}')
  print(f'len chunks: {len(final_bchunks)}')
  print(f'len title: {len(final_btitle)}')
  bdata = ({'score':final_bscore,
          'chunk':final_bchunks,
            'title':final_btitle,
            'genre':final_bgenre,
            'index':final_bindex})
  bdf_tmp = pd.DataFrame(bdata)
  bdf_tmp.sort_values('score', ascending = False, inplace = True)
  bdf_tmp = bdf_tmp[(bdf_tmp['score'] > -1) & (bdf_tmp['score'] < 1) ]

  print('\n\n-------------------------------------------\n\n')
  #bdf_tmp = bdf_tmp.iloc[:200]
  print('bdf_tmp shape: ',bdf_tmp.shape)
  print(bdf_tmp.groupby("index", sort = True)['index'].count())
  bdf_tmp_group = bdf_tmp.groupby("index", sort = True)

  ###########################################################################################
  tmp = df_tmp_group.filter(lambda x: x['index'].count()> 10)['index'].reset_index().groupby("index", sort = True)
  crossEncode_tops = list(tmp.groups.keys())

  btmp = bdf_tmp_group.filter(lambda x: x['index'].count()> 10)['index'].reset_index().groupby("index", sort = True)
  bertScore_tops = list(btmp.groups.keys())

  labels = list(set(bertScore_tops).intersection(crossEncode_tops))
  query_info[q]['labels'] = labels
  print(labels)
  print('\n\n============================\n\n')