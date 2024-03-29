# -*- coding: utf-8 -*-
"""Bi-Encoder.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kL7TsF1wFfyeYP2W4xNOrxr5Xe2FGWm1
"""

from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/msmarco-distilbert-dot-v5')
model = AutoModel.from_pretrained('sentence-transformers/msmarco-distilbert-dot-v5')
def get_embedding(sentences,tokenizer, model):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)
        #print(f'\n\n------- {model_output}\n\n')
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        #print(f'\n\n------- {token_embeddings.shape}\n\n')
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    #print(type(sentence_embeddings))
    return sentence_embeddings


corpus = df_new['Plot']

corpus_embeddings = np.array(df_new['embeddings'].values)
corpus_embeddings = [np.array(x) for x in corpus_embeddings]
corpus_embeddings = np.array(corpus_embeddings)
#print(corpus_embeddings.shape)
corpus_embeddings = corpus_embeddings.reshape((corpus_embeddings.shape[0],corpus_embeddings.shape[2]))
#print(corpus_embeddings.shape)
queries = question_list
top_k = min(50, len(corpus))
all_q_results = []
for query in queries:
  top_results = []
  query_embedding = np.array( get_embedding(query, tokenizer, model) )
  # We use cosine-similarity and torch.topk to find the highest scores
  cos_scores = util.dot_score(query_embedding, corpus_embeddings)[0]
  top_results = torch.topk(cos_scores, k=top_k)
  all_q_results.append(top_results)
  print("\n\n=================================================\n\n")
  print("Query:", query)
  print("\nTop 50 most similar sentences in corpus:")

  scores = []
  plot = []
  title = []
  genre = []
  for score, idx in zip(top_results[0], top_results[1]):
    #print(df_new.iloc[int(idx)]['index'],'-----', df_new.iloc[int(idx)]['Clean_Genre'],'-----', "(Score: {:.4f})".format(score))
    #print("(Score: {:.4f})".format(score))
    scores.append(score.item())
    plot.append(df_new.iloc[int(idx)]['Plot'])
    genre.append(df_new.iloc[int(idx)]['Clean_Genre'])
    title.append(df_new.iloc[int(idx)]['index'])
  data = ({'score':scores,
            'genre':genre,
            'plot':plot,
            'title':title})
  df_tmp = pd.DataFrame(data)
  df_tmp.sort_values('score', ascending = False, inplace = True)
  for index,row in df_tmp.iterrows():
    print(index, '>>>>  ',row['score'], '>>>>  ', row['title'], '>>>>  ')#, row['plot'])
  print('\n\n----------------------------------\n\n')