import pandas as pd
import numpy as np
import gensim
from transformers import AutoTokenizer, AutoModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from src.embedding import generate_embeddings
import torch
from tqdm import tqdm
import time

def preprocess_text(text, remove_stopwords=True):
    # Remove non-alphanumeric characters
    if not text:
        print("Issue with the current abstract")
    # print(text)
    text = re.sub(r'\W+', ' ', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Lowercase conversion
    tokens = [token.lower() for token in tokens]

    # Remove stop words
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def load_and_preprocess_data(file_name):
    df = pd.read_csv(file_name)
    print(df.shape)
    # Apply preprocessing without stop words
    df['preprocessed_trial_no_stopwords'] = df['brief_summary_description'].apply(
        lambda x: preprocess_text(x, remove_stopwords=True))

    # Apply preprocessing but keep stop words
    df['preprocessed_trial'] = df['brief_summary_description'].apply(
        lambda x: preprocess_text(x, remove_stopwords=False))
    return df

def embed_text_to_vec(df, column_to_embed, model_name):

    if model_name == "doc2vec":
        print("using doc2vec")
        # Step 3: Prepare Data for Doc2Vec
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in
                       enumerate(df[column_to_embed])]

        # Step 4: Initialize and Train Doc2Vec Model
        model = Doc2Vec(vector_size=100, min_count=2, epochs=40)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

        # Step 5: Vectorize Abstracts using Doc2Vec
        X = np.array([model.infer_vector(doc.words) for doc in tagged_data])

    else:
        # Step 3: Vectorize the abstracts
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df[column_to_embed])

    return X


if __name__ == '__main__':
    df = load_and_preprocess_data('data/annotated_aact/normalized_annotations_unique_19607_with_details.csv')
    column_to_embed = "preprocessed_trial_no_stopwords"

    ### BERT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("running on device: {}".format(device))
    abstracts = df[column_to_embed].tolist()
    # specifying model
    checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    # "dmis-lab/biobert-v1.1"
    # "dmis-lab/biobert-v1.1"
    # "allenai/scibert_scivocab_uncased"
    # "bert-base-uncased"
    # "dmis-lab/biobert-v1.1"
    # "allenai/scibert_scivocab_uncased"
    # "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    # "dmis-lab/biobert-base-cased-v1.2"
    # "bert-base-uncased"
    # "allenai/scibert_scivocab_uncased"
    # "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)
    mat = np.empty([len(abstracts), 768])
    abstract_batch = abstracts
    # Start timing
    start_time = time.time()

    # Process abstracts in batches and track progress with tqdm
    for i, abst in enumerate(tqdm(abstract_batch, desc="Generating Embeddings")):
        _, mat[i], _ = generate_embeddings(abst, tokenizer, model, device)
        last_iter = np.array([i])
        np.save('./data/variables/last_iter_batch_1', last_iter)

    # save embedding
    np.save(f'./data/embeddings/embeddings_{checkpoint.replace("/","_")}', mat)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time for generating embeddings: {elapsed_time:.2f} seconds")

    #### BASIC
    if False:
        model_name = "doc2vec"
        X = embed_text_to_vec(df, column_to_embed, model_name)




