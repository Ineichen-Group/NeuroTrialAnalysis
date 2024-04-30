import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from Snomed import Snomed
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cdist
import time
from multiprocessing import Pool, cpu_count, set_start_method
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm.auto import tqdm  # Use 'auto' to ensure compatibility with notebooks and scripts

def load_snomed_terminology(snomed):

    snomed_sf_id_pairs = []

    for snomed_id in tqdm(snomed.graph.nodes):

        node_descs = snomed.index_definition[snomed_id]
        for d in node_descs:
            snomed_sf_id_pairs.append((d, snomed_id))

    print(len(snomed_sf_id_pairs))

    return snomed_sf_id_pairs

def load_snomed_embeddings(path):
    # Define the directory where your files are stored
    directory_path = path

    # List all files in the directory that match the pattern
    files = [f for f in os.listdir(directory_path) if f.startswith('all_reps_emb_full_batch_') and f.endswith('.npy')]

    # Sort files to maintain the order, especially important if the batch index is used in processing
    files.sort()

    # Initialize an empty list to hold the data from each file
    all_data = []

    # Load each file and append the data to the list
    for file in files:
        file_path = os.path.join(directory_path, file)
        data = np.load(file_path)
        all_data.append(data)

    # Concatenate all the arrays from the list into one
    all_reps_emb_full = np.concatenate(all_data, axis=0)
    return all_reps_emb_full


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def map_to_snomed(model, tokenizer, snomed, all_reps_emb_full, query):
    query_toks = tokenizer.batch_encode_plus([query],
                                             padding="max_length",
                                             max_length=25,
                                             truncation=True,
                                             return_tensors="pt")
    query_output = model(**query_toks)
    query_cls_rep = query_output[0][:, 0, :]
    dist = cdist(query_cls_rep.cpu().detach().numpy(), all_reps_emb_full)
    nn_index = np.argmin(dist)
    min_distance = dist[0, nn_index]  # Extract the minimum distance

    # Calculate softmax probabilities from negative distances (as softmax is naturally maximization)
    probabilities = softmax(-dist[0])
    predicted_probability = probabilities[nn_index]

    # print ("predicted label:", snomed_sf_id_pairs[nn_index])
    term, term_id = snomed_sf_id_pairs[nn_index][0], snomed_sf_id_pairs[nn_index][1]
    return term, term_id, snomed[term_id]['desc'], round(min_distance, 4), round(predicted_probability, 4)


def process_conditions(row, model, tokenizer, snomed, all_reps_emb_full):
    if pd.isna(row) or not isinstance(row, str):
        # Return empty strings and empty dictionaries for all the values
        return "", "", "", {}, {}, "", ""

    terms = row.split('|')
    snomed_terms = []
    snomed_termids = []
    snomed_norms = []
    min_distances = []  # List to store minimum distances
    predicted_probabilities = []  # List to store predicted probabilities

    # Dictionaries to track mappings
    norm_to_terms = {}  # SNOMED norm as key, list of terms as values
    term_to_norm = {}  # Each term from the row and the SNOMED norm to which it was mapped

    for term in terms:
        if len(term) > 4 or term == 'pain':  # Conditional processing based on term criteria
            snomed_term, snomed_termid, snomed_norm, min_distance, predicted_probability = map_to_snomed(model, tokenizer, snomed, all_reps_emb_full, term)
            snomed_terms.append(snomed_term)
            snomed_termids.append(snomed_termid)
            snomed_norms.append(snomed_norm)
            min_distances.append(
                str(round(min_distance, 4)))  # Convert to string and store the rounded minimum distance
            predicted_probabilities.append(
                str(round(predicted_probability, 4)))  # Convert to string and store the rounded predicted probability

            # Populate dictionaries
            if snomed_norm in norm_to_terms:
                norm_to_terms[snomed_norm].append(term)
            else:
                norm_to_terms[snomed_norm] = [term]

            term_to_norm[term] = snomed_norm

    # Ensure unique terms in norm_to_terms dictionary
    for key in norm_to_terms:
        norm_to_terms[key] = list(set(norm_to_terms[key]))

    return '|'.join(snomed_terms), '|'.join(snomed_termids), '|'.join(snomed_norms), '|'.join(min_distances), '|'.join(
        predicted_probabilities), norm_to_terms, term_to_norm

def process_chunk(chunk, model, tokenizer, snomed, all_reps_emb_full, column_name):
    return chunk[column_name].apply(lambda x: pd.Series(process_conditions(x, model, tokenizer, snomed, all_reps_emb_full)))


if __name__ == '__main__':

    release_id = '20240401'
    SNOMED_PATH = '../data/snomed/SnomedCT_InternationalRF2_PRODUCTION_20240401T120000Z'  # you need to download your own SNOMED distribution
    snomed = Snomed(SNOMED_PATH, release_id=release_id)
    snomed.load_snomed()
    snomed_sf_id_pairs = load_snomed_terminology(snomed)

    embeddings_directory_path = '../data/embeddings/normalization'
    all_reps_emb_full = load_snomed_embeddings(embeddings_directory_path)

    df_all = pd.read_csv('../data/annotated_aact/normalized_annotations_unique_19607.csv', index_col=False)

    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

    start_time = time.time()
    tqdm.pandas(desc="Processing Conditions")  # This line prepares tqdm to work with pandas apply
    results_aact = df_all['canonical_aact_conditions'].progress_apply(
        lambda x: pd.Series(process_conditions(x,  model, tokenizer, snomed, all_reps_emb_full))
    )
    df_all[['aact_snomed_term', 'aact_snomed_termid', 'aact_snomed_term_norm', 'aact_cdist', 'aact_softmax_prob']] = \
    results_aact[[0, 1, 2, 3, 4]]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    df_all.to_csv('../data/annotated_aact/sapbert_normalized_annotations_aact_19607.csv')

    ### PARALLELISM
    # with Pool(num_cores) as pool:
    #    results = pool.starmap(process_chunk, [(chunk, model, tokenizer, snomed, all_reps_emb_full, target_column) for chunk in chunks])
    #    combined_results = pd.concat(results, ignore_index=True)

    target_column = 'canonical_aact_conditions'  # Parameterize the target column
    num_cores = cpu_count() - 2 # Get the number of CPU cores
    chunk_size = len(df_all) // num_cores  # Calculate the chunk size

    #print(f"Processing {len(df_all)} with {num_cores} cores.")

    # Split df_all into chunks
    chunks = [df_all.iloc[i:i + chunk_size] for i in range(0, len(df_all), chunk_size)]

    #df_all[['aact_snomed_term', 'aact_snomed_termid', 'aact_snomed_term_norm', 'aact_cdist', 'aact_softmax_prob']] = \
    #combined_results[[0, 1, 2, 3, 4]]

