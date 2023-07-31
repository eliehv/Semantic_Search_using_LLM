# -*- coding: utf-8 -*-
"""Cross-Encoder-Reranker.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/115M1H_KUN3d67bPpSWsBkZxvGmmhYn3r
"""

##########################################################################################
#####                               Reranking
#####           calculate CrossEncoder for the top_k returned by BiEncoder
##########################################################################################
# the top_k results by BiEncoder for all queries is stored in ==> all_q_results
# all_q_results for each query contains top_k results with the following structure ==>
#it has two columns, first column is the score and second is the index of the item in the dataframe
from sentence_transformers import SentenceTransformer, util, CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')


st = time.process_time()

for q in range(len(query_info)):
  query_info[q]["top_k"]["CrossEncoder"] = []
  for i in range(len(query_info[q]['top_k']['chunks'])):
    cross_inp = []
    for j in range(len(query_info[q]['top_k']['chunks'][i])):
      cross_inp.append([queries[q], query_info[q]['top_k']['chunks'][i][j]])



    #print(f"len chunks in the first top_k : {len(query_info[q]['top_k']['chunks'][i])}")
    #print(cross_inp)
    cross_s = cross_encoder.predict(cross_inp)
    query_info[q]["top_k"]["CrossEncoder"].append(cross_s)

    #print('len cross_s: ',len(cross_s), '\n------------------------\n')
et = time.process_time()
res = et - st
print('\n\n****** CPU Execution time *******: ', res, 'seconds')