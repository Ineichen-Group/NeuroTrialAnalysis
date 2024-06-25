import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from Snomed import Snomed
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cdist
import time
import torch
import json
from transformers import PreTrainedTokenizer, PreTrainedModel
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm.auto import tqdm  # Use 'auto' to ensure compatibility with notebooks and scripts
from typing import Any, Tuple, Dict, List, Union
from pathlib import Path
import logging
import argparse
import sys

logger = logging.getLogger(__name__)


def load_relevant_snomed_terminology(snomed_df: pd.DataFrame, concept_type_subset: List[str]) -> List[Tuple[str, int]]:
    """
    Loads SNOMED terminology and filters it based on a subset of concept types.
    
    This function processes a DataFrame containing SNOMED terms to identify and propagate
    canonical hierarchies to all synonyms. It then filters the terms based on a specified
    subset of concept types, and returns a list of name-ID pairs for the filtered terms.

    Args:
    snomed_df (pd.DataFrame): DataFrame containing SNOMED terms with columns 'concept_id', 'hierarchy', 
                              'name_type', and 'concept_name'.
    concept_type_subset (List[str]): List of hierarchy types to filter the SNOMED terms.

    Returns:
    List[Tuple[str, int]]: List of tuples, each containing a concept name and its corresponding concept ID
                           from the filtered set of SNOMED terms.
    """
    # Identify canonical concept_names and their hierarchy
    canonical_hierarchy = snomed_df[snomed_df['name_type'] == 'Canonical'][['concept_id', 'hierarchy']]

    # Merge to propagate the hierarchy to synonyms
    snomed_df = snomed_df.merge(canonical_hierarchy, on='concept_id', suffixes=('', '_canonical'), how='left')
    
    # Update the hierarchy column for synonyms
    snomed_df['hierarchy'] = snomed_df['hierarchy'].fillna(snomed_df['hierarchy_canonical'])

    # Drop the auxiliary column used for propagation
    snomed_df = snomed_df.drop(columns=['hierarchy_canonical'])

    # Filter the DataFrame based on the specified concept types
    filtered_df = snomed_df[snomed_df.hierarchy.isin(concept_type_subset)]

    # Extract lists of names and IDs
    all_names = filtered_df['concept_name'].values.tolist()
    all_ids = filtered_df['concept_id'].values.tolist()

    # Create list of tuples pairing names with IDs
    snomed_sf_id_pairs = [(snomed_name, snomed_id) for snomed_name, snomed_id in zip(all_names, all_ids)]

    return snomed_sf_id_pairs


def load_snomed_embeddings(path, files_prefix='all_reps_emb_full_batch_'):
    # Define the directory where your files are stored
    directory_path = path

    # List all files in the directory that match the pattern
    files = [f for f in os.listdir(directory_path) if f.startswith(files_prefix) and f.endswith('.npy')]

    print("SNOMED embeddings files: ", directory_path)
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


def map_query_to_snomed(query: str, 
                  tokenizer: PreTrainedTokenizer, 
                  model: PreTrainedModel, 
                  all_reps_emb_full: np.ndarray, 
                  snomed_sf_id_pairs: np.ndarray, 
                  canonical_mapping_dict: Dict[str, str],
                  n_entities: int = 3) -> Tuple[int, str, str, List[Tuple[str, int]], float]:
    
    """
    Map a query to the closest SNOMED concept using a pre-trained model and return its canonical form.

    Parameters:
    - query (str): The input query string to be mapped.
    - tokenizer (PreTrainedTokenizer): The tokenizer used for encoding the query.
    - model (PreTrainedModel): The pre-trained model used to generate embeddings for the query.
    - all_reps_emb_full (np.ndarray): The array of embeddings for all SNOMED concepts.
    - snomed_sf_id_pairs (np.ndarray): The array of SNOMED concept ID and label pairs.
    - canonical_mapping_dict (Dict[str, str]): A dictionary mapping SNOMED IDs to their canonical forms.
    - n_entities (int): The number of nearest entities to retrieve.

    Returns:
    - Tuple[int, str, str, List[Tuple[str, int]], float]: The predicted SNOMED concept ID, label, its canonical form,
      a list of the nearest entities, and the minimum distance.
    """
    # Move embeddings to GPU if available
    if torch.cuda.is_available():
        all_reps_emb_full= torch.tensor(all_reps_emb_full).to('cuda')

    if torch.cuda.is_available():
        model = model.to('cuda')
        
    # Encode the query
    query_toks = tokenizer.batch_encode_plus([query], 
                                             padding="max_length", 
                                             max_length=25, 
                                             truncation=True,
                                             return_tensors="pt")
    if torch.cuda.is_available():
        query_toks = query_toks.to('cuda')  # Move tensors to GPU
        
    # Get the model output
    with torch.no_grad():
        query_output = model(**query_toks)
    
    # Extract the CLS token representation
    query_cls_rep = query_output[0][:, 0, :]

    # Compute distances between query embedding and all SNOMED concept embeddings
    if torch.cuda.is_available():
        dist = torch.cdist(query_cls_rep, all_reps_emb_full)
        nn_index = torch.argmin(dist).item()  # This finds the index of the minimum value
        min_distance = dist[0, nn_index].item()  # Extract the minimum distance at that index
    else:
        dist = cdist(query_cls_rep.cpu().detach().numpy(), all_reps_emb_full)
        nn_index = np.argmin(dist).item()
        min_distance = dist[0, nn_index]  # Since dist is a numpy array, get the minimum distance at the index 

    # Retrieve the nearest n_entities
    nearest_n_entities = []
    if torch.cuda.is_available():
        nearest_n_indices = torch.argsort(dist[0])[:n_entities]  # Get indices of the n smallest distances
    else:
        nearest_n_indices = np.argsort(dist[0])[:n_entities]
    for idx in nearest_n_indices:
        nearest_n_entities.append(snomed_sf_id_pairs[idx.item()])
        
    # Get the predicted SNOMED concept ID and label
    predicted_label = snomed_sf_id_pairs[nn_index]
    predicted_id = predicted_label[1]

    # Get the canonical form from the dictionary
    canonical_form = canonical_mapping_dict.get(predicted_id, "Canonical form not found")

    # Return the predicted SNOMED concept ID, label, and canonical form
    return predicted_id, predicted_label[0], canonical_form, nearest_n_entities, round(min_distance, 4)


def process_row_annotations(
    row: Union[str, float], 
    tokenizer: Any, 
    model: Any, 
    all_reps_emb_full: Any, 
    snomed_sf_id_pairs: Dict[str, str], 
    canonical_mapping_dict: Dict[str, str]
) -> Tuple[str, str, str, str, str, Dict[str, List[str]], Dict[str, str]]:
    """
    Processes a row of annotations, mapping terms to SNOMED CT concepts and returning the results.

    Parameters:
    - row (Union[str, float]): A string of terms separated by '|', or NaN.
    - tokenizer (Any): The tokenizer used for mapping terms.
    - model (Any): The model used for mapping terms.
    - all_reps_emb_full (Any): The embeddings used for mapping terms.
    - snomed_sf_id_pairs (Dict[str, str]): Dictionary of SNOMED ID and term pairs.
    - canonical_mapping_dict (Dict[str, str]): Dictionary mapping terms to their canonical forms.

    Returns:
    - Tuple[str, str, str, str, str, Dict[str, List[str]], Dict[str, str]]:
        - Concatenated SNOMED terms.
        - Concatenated SNOMED term IDs.
        - Concatenated canonical forms of the SNOMED terms.
        - Concatenated closest 3 entities.
        - Concatenated minimum distances.
        - Dictionary mapping canonical forms to lists of terms.
        - Dictionary mapping terms to their canonical forms.
    """    
    if pd.isna(row) or not isinstance(row, str):
        # Return empty strings and empty dictionaries for all the values
        return "", "", "", {}, {}, "", ""
    
    terms = row.split('|')
    snomed_terms = []
    snomed_terms_canonical = []
    snomed_termids = []
    snomed_norms = []
    closest_3_entites = []
    min_distances = []  # List to store minimum distances

    # Dictionaries to track mappings
    norm_to_terms = {}  # SNOMED norm as key, list of terms as values
    term_to_norm = {}   # Each term from the row and the SNOMED norm to which it was mapped

    for term in terms:
        predicted_id, predicted_label, canonical_form, n_3_entities, nn_distance = map_query_to_snomed(term, tokenizer, model, all_reps_emb_full, snomed_sf_id_pairs, canonical_mapping_dict)
        snomed_terms.append(predicted_label)
        snomed_terms_canonical.append(canonical_form)
        snomed_termids.append(predicted_id)
        min_distances.append(nn_distance)
        closest_3_entites.append(n_3_entities)

        # Populate dictionaries
        #print(canonical_form)
        if canonical_form in norm_to_terms:
            norm_to_terms[canonical_form].append(term)
        else:
            norm_to_terms[canonical_form] = [term]

        term_to_norm[term] = canonical_form

    # Ensure unique terms in norm_to_terms dictionary
    for key in norm_to_terms:
        norm_to_terms[key] = list(set(norm_to_terms[key]))

    return '|'.join(snomed_terms), '|'.join(snomed_termids), '|'.join(snomed_terms_canonical), '|'.join([str(ents) for ents in closest_3_entites]), '|'.join([str(dist) for dist in min_distances]), norm_to_terms, term_to_norm

def load_snomed_ct_df(data_path: Path, release_id: str):
    """
    Create a SNOMED CT concept DataFrame.

    Derived from: https://github.com/CogStack/MedCAT/blob/master/medcat/utils/preprocess_snomed.py

    Returns:
        pandas.DataFrame: SNOMED CT concept DataFrame.
    """

    def _read_file_and_subset_to_active(filename):
        with open(filename, encoding="utf-8") as f:
            entities = [[n.strip() for n in line.split("\t")] for line in f]
            df = pd.DataFrame(entities[1:], columns=entities[0])
        return df[df.active == "1"]

    active_terms = _read_file_and_subset_to_active(
         f"{data_path}/sct2_Concept_Snapshot_INT_{release_id}.txt"
    )
    active_descs = _read_file_and_subset_to_active(
        f"{data_path}/sct2_Description_Snapshot-en_INT_{release_id}.txt"
    )

    df = pd.merge(active_terms, active_descs, left_on=["id"], right_on=["conceptId"], how="inner")[
        ["id_x", "term", "typeId"]
    ].rename(columns={"id_x": "concept_id", "term": "concept_name", "typeId": "name_type"})

    print("Loaded SNOMED size: ", df.shape)

    # active description or active synonym
    df["name_type"] = df["name_type"].replace(
        ["900000000000003001", "900000000000013009"], ["Canonical", "Synonym"]
    )
    active_snomed_df = df[df.name_type.isin(["Canonical", "Synonym"])]
    print("Active SNOMED size: ", active_snomed_df.shape)

    active_snomed_df["hierarchy"] = active_snomed_df["concept_name"].str.extract(
        r"\((\w+\s?.?\s?\w+.?\w+.?\w+.?)\)$"
    )

    return active_snomed_df

def load_snomed_df_and_embeddings(
    SNOMED_PATH: str, 
    release_id: str, 
    concept_type_subset: List[str], 
    embeddings_directory_path: str, 
    embeddings_prefix: str
) -> Tuple[pd.DataFrame, List[Tuple[str, int]], List[float]]:
    """
    Loads the SNOMED DataFrame and corresponding embeddings, filters the DataFrame based on a subset 
    of concept types, and ensures that the number of filtered SNOMED concepts matches the number of their embeddings.

    Args:
    SNOMED_PATH (str): Path to the SNOMED CT release data.
    release_id (str): Identifier for the SNOMED release.
    concept_type_subset (List[str]): List of hierarchy types to filter the SNOMED terms.
    embeddings_directory_path (str): Path to the directory containing the embeddings.
    embeddings_prefix (str): Prefix used for naming the embeddings files.

    Returns:
    Tuple[pd.DataFrame, List[Tuple[str, int]], List[float]]:
        - pd.DataFrame: Filtered SNOMED DataFrame.
        - List[Tuple[str, int]]: List of tuples, each containing a concept name and its corresponding concept ID.
        - List[float]: List of embeddings corresponding to the filtered SNOMED concepts.

    Raises:
    ValueError: If the number of filtered SNOMED concept pairs does not match the number of embeddings.
    """
    # Load relevant SNOMED concepts
    snomed_df = load_snomed_ct_df(SNOMED_PATH, release_id)
    snomed_sf_id_pairs = load_relevant_snomed_terminology(snomed_df, concept_type_subset)
    print(f'Length of all SNOMED concepts: {len(snomed_sf_id_pairs)}')

    # Load SNOMED concepts embeddings
    all_reps_emb_full = load_snomed_embeddings(embeddings_directory_path, embeddings_prefix)
    print(f'Length of all SNOMED concept embeddings: {len(all_reps_emb_full)}')

    # Check if the length of snomed_sf_id_pairs matches all_reps_emb_full
    if len(snomed_sf_id_pairs) != len(all_reps_emb_full):
        raise ValueError('Mismatch between the number of SNOMED concepts and their embeddings.')

    return snomed_df, snomed_sf_id_pairs, all_reps_emb_full

def extract_and_save_json_term_mappings(data_path, source_annotations_model, target_entity_type, data_size, term_to_norm_dict, norm_to_term_dict):
    ### Mapping dictionaries
    combined_term_to_norm = {}
    combined_norm_to_term = {}

    # Iterate through each dictionary in column index 4 of the DataFrame and update the combined dictionary
    for dict_item in term_to_norm_dict:
        combined_term_to_norm.update(dict_item)

    # Iterate through each dictionary in column index 4 of the DataFrame and update the combined dictionary
    for dict_item in norm_to_term_dict:
        combined_norm_to_term.update(dict_item)

    filename = f'{data_path}/annotated_aact/snomed_linking_outputs/{source_annotations_model}_combined_term_to_norm_dict_{data_size}_{target_entity_type}.json'
    filename_2 = f'{data_path}/annotated_aact/snomed_linking_outputs/{source_annotations_model}_combined_norm_to_term_dict_{data_size}_{target_entity_type}.json'

    # Save the dictionary to a JSON file
    with open(filename, 'w') as f:
        json.dump(combined_term_to_norm, f, indent=4)
    with open(filename_2, 'w') as f:
        json.dump(combined_norm_to_term, f, indent=4)

def run_sapbert_and_save(data_path, df_file_path_to_map, tokenizer, model, all_reps_emb_full, snomed_sf_id_pairs, canonical_mapping_dict, source_annotations_model, target_entity_type, target_column):
    df_all = pd.read_csv(df_file_path_to_map, index_col=False)
    df_all = df_all.head(10)
    start_time = time.time()
    tqdm.pandas(desc=f"Processing {target_entity_type}")  # This line prepares tqdm to work with pandas apply
    results_normalization = df_all[target_column].progress_apply(
        lambda x: pd.Series(process_row_annotations(x, tokenizer, model, all_reps_emb_full, snomed_sf_id_pairs, canonical_mapping_dict))
    )
    df_all[[f'{source_annotations_model}_snomed_term_{target_entity_type}', 
            f'{source_annotations_model}_snomed_termid_{target_entity_type}', f'{source_annotations_model}_snomed_term_canonical_{target_entity_type}', f'{source_annotations_model}_snomed_closest_n_{target_entity_type}', f'{source_annotations_model}_cdist_{target_entity_type}']] = results_normalization[[0, 1, 2, 3, 4]]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    df_all.to_csv(f'{data_path}/annotated_aact/snomed_linking_outputs/sapbert_normalized_annotations_{source_annotations_model}_{len(df_all)}_{target_entity_type}.csv')

    return df_all, results_normalization

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    release_id = '20240401'
    data_path = "./data"
    concept_type_subset = [
        "disorder",  
        "substance"
    ]
    SNOMED_PATH = f'{data_path}/snomed/SnomedCT_InternationalRF2_PRODUCTION_{release_id}T120000Z/Snapshot/Terminology'  # you need to download your own SNOMED distribution
    embeddings_directory_path = f'{data_path}/embeddings/snomed_normalization/'
    #target_entity_type = 'interventions' # or 'conditions'
    
    parser = argparse.ArgumentParser(description="Run SapBERT over named entities and link to SNOMED concepts.")
    parser.add_argument("--target_entity_type", default="conditions", type=str, help="Entity type that has been obtained from NER.")
    parser.add_argument("--source_annotations_model", default="linkbert", type=str, help="Model that was used for NER. Will be used to create the output column names.")
    parser.add_argument("--target_column_prefix", default="canonical_BioLinkBERT-base", type=str, help="Prefix in the column name where the NER annotations are. The suffic will be the target_entity_type.")

    args = parser.parse_args()

    target_entity_type = args.target_entity_type
    target_col_prefix = args.target_column_prefix
    source_annotations_model = args.source_annotations_model
    target_column = f'{target_col_prefix}_{target_entity_type}'

    # LOAD SNOMED concepts and their embeddings
    snomed_df, snomed_sf_id_pairs, all_reps_emb_full = load_snomed_df_and_embeddings(SNOMED_PATH, release_id, concept_type_subset, embeddings_directory_path, 'disorder_substance_emb_batch')

    # LOAD SNOMED mapping from synonyms to canonical forms
    dict_canoncial_json_file_path = f'{data_path}/snomed/mapping_dictionaries/disorder_substance_canonical_dict.json'
    with open(dict_canoncial_json_file_path, 'r') as json_file:
         canonical_mapping_dict = json.load(json_file)
    print(f'Length of synonyms to canonical mapping dictionary: {len(canonical_mapping_dict)}')

    ### LOAD SapBERT model
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

    ### File to annotate
    df_to_annotate_file_path = f'{data_path}/annotated_aact/ner_outputs/aggregated_ner_annotations_basic_dict_mapped_19632.csv' #TODO: can be more generic, use as argument

    ### RUN SapBERT normalization and save 
    df_all, results_normalization = run_sapbert_and_save(data_path, df_to_annotate_file_path, tokenizer, model, all_reps_emb_full, snomed_sf_id_pairs, canonical_mapping_dict, source_annotations_model, target_entity_type, target_column)

    extract_and_save_json_term_mappings(data_path, source_annotations_model, target_entity_type, len(df_all), results_normalization[6], results_normalization[5])

if __name__ == '__main__':
    main()

