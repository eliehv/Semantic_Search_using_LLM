
# ********************* BiEncoder peformance evaluation************************************
MRR = 0
#Precision and Recall
percision = []
recall = []
for q in range(len(query_info_msmarco)):
  predicted_list = query_info_msmarco[q]['top_k']['score_idx'][1].tolist() #indeces of the predicted docs
  #predicted_list = predicted_list[:len(predicted_list)]
  print('predicted_list: ',(predicted_list))
  query_relevant_docs = relevant_list#query_info[q]['labels_idx']

  print('query_relevant_docs ',(query_relevant_docs))
  for rank, hit in enumerate(predicted_list):
      if hit in query_relevant_docs:
          MRR += 1.0 / (rank + 1)
          break

  num_correct = 0
  for rank, hit in enumerate(predicted_list):
      if hit in query_relevant_docs:
          num_correct += 1

  percision.append(num_correct / len(predicted_list)) # TP/(TP + FP)
  recall.append(num_correct / len(query_relevant_docs)) #TP/(TP + FN)
print('MRR@50: {:.2f}'.format(MRR/len(query_info_msmarco)))
print('\nprecision :  ')
for p in percision:
  print('{:.2f}'.format(p))
print('\nrecall  ')
for r in recall:
  print('{:.2f}'.format(r))

# ********************* CrossEncoder performance evaluation************************************
MRR = 0
#Precision and Recall
percision = []
recall = []
for q in range(len(query_info_msmarco)):
  cross_df = query_info_msmarco[q]['top_k']['CrossEncoder_results']
  predicted_list = cross_df['plot_idx'].tolist()#query_info[q]['top_k']['score_idx'][1].tolist()
  #predicted_list = predicted_list[:len(predicted_list)]
  print('predicted_list: ',(predicted_list))
  query_relevant_docs = relevant_list#query_info[q]['labels_idx']

  print('query_relevant_docs ',(query_relevant_docs))
  for rank, hit in enumerate(predicted_list):
      if hit in query_relevant_docs:
          MRR += 1.0 / (rank + 1)
          break

  num_correct = 0
  for rank, hit in enumerate(predicted_list):
      if hit in query_relevant_docs:
          num_correct += 1

  percision.append(num_correct / len(predicted_list)) # TP/(TP + FP)
  recall.append(num_correct / len(query_relevant_docs)) #TP/(TP + FN)

print('MRR@50: {:.2f}'.format(MRR/len(query_info_msmarco)))
print('\nprecision :  ')
for p in percision:
  print('{:.2f}'.format(p))
print('\nrecall  ')
for r in recall:
  print('{:.2f}'.format(r))

# ********************* BertScore performance evaluation ************************************
MRR = 0
#Precision and Recall
percision = []
recall = []
for q in range(len(query_info_msmarco)):
  cross_df = query_info_msmarco[q]['top_k']['BertScore_results']
  predicted_list = cross_df['plot_idx'].tolist()#query_info[q]['top_k']['score_idx'][1].tolist()
  #predicted_list = predicted_list[:len(predicted_list)]
  print('predicted_list: ',(predicted_list))
  query_relevant_docs = relevant_list#query_info[q]['labels_idx']

  print('query_relevant_docs ',(query_relevant_docs))
  for rank, hit in enumerate(predicted_list):
      if hit in query_relevant_docs:
          MRR += 1.0 / (rank + 1)
          break

  num_correct = 0
  for rank, hit in enumerate(predicted_list):
      if hit in query_relevant_docs:
          num_correct += 1

  percision.append(num_correct / len(predicted_list)) # TP/(TP + FP)
  recall.append(num_correct / len(query_relevant_docs)) #TP/(TP + FN)

print('MRR@50: {:.2f}'.format(MRR/len(query_info_msmarco)))
print('\nprecision :  ')
for p in percision:
  print('{:.2f}'.format(p))
print('\nrecall  ')
for r in recall:
  print('{:.2f}'.format(r))

#********************** Heatmap plot for the query and retrieved chunks of plot *********************
from bert_score import plot_example
for q in range(len(query_info_msmarco)):
  cross_df = query_info_msmarco[q]['top_k']['BertScore_results']
  predicted_list = cross_df['plot_idx'].tolist()#query_info[q]['top_k']['score_idx'][1].tolist()
  predicted_chunk = cross_df['chunk'].tolist()
  predicted_title = cross_df['title'].tolist()
  print('The query:', query_info_msmarco[q]['query'])
  print('The best Predicted item:',predicted_title[0])
  plot_example(query_info_msmarco[q]['query'],predicted_chunk[0] , lang="en")
  break
