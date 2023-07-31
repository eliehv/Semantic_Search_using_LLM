
# #################################################################################################
#####                 Embedding Extraction using msmarco-distilbert-dot-v5 langusge model
####################################################################################################
# this part od code must be repeated for all parts of data
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/msmarco-distilbert-dot-v5')
model = AutoModel.from_pretrained('sentence-transformers/msmarco-distilbert-dot-v5')

def get_embedding(sentences,tokenizer, model):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings
#-----------------------------------------------------------------------------------------------

# ################################################################################
#####                 Embedding Extraction
#    the dataset is split into parts based on Genre categorization
#    because the available free memory and processing can't handle all in once
#    the embedding extraction function is applied for all parts of the dataset
##################################################################################
iter_group = iter(df.groupby('Clean_Genre', sort = True))
genre, frame = next(iter_group)
for genre, frame in iter_group:

  if genre == 'western':
    print(genre)
    st = time.process_time()
    frame['n_tokens'] = frame.Plot.apply(lambda x: len(tokenizer(x, padding=True, truncation=False, return_tensors='pt')[0]))
    frame['embeddings'] = frame.Plot.apply(lambda x:get_embedding(x, tokenizer, model))
    et = time.process_time()
    res = et - st
    print('CPU Execution time : ', res, 'seconds')
    with open('/content/drive/MyDrive/Colab Notebooks/Data/filename.pickle', 'wb') as f:
      pickle.dump(frame,f)

# ################################################################################
#####            concatenate  all data frames with embeddings and n_tokens
##################################################################################
import pathlib
path = pathlib.Path('/content/drive/MyDrive/Colab Notebooks/Data')
files = list(path.glob('*.pickle'))

df_new = pd.DataFrame(columns = frame.columns)

for fpath in files:
  with open(fpath, 'rb') as f:
    df_tmp = pickle.load(f)
    df_new = pd.concat([df_new, df_tmp])

print(f'df_new shape:{df_new.shape}')
print(f'Max number of tokens: {df_new.n_tokens.max()}')
print(f'Min number of tokens: {df_new.n_tokens.min()}')

with open('/content/drive/MyDrive/Colab Notebooks/Data/filename.pickle', 'wb') as f:
  pickle.dump(df_new,f)
