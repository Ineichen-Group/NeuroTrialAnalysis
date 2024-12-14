# NeuroTrialAnalysis
Analysis of drug-testing interentional clinical trials registry data from ClinicalTrials.gov with a focus on neurological and psychiatric conditions.

# 1. Set up the environment
Python 3.11.7 was used for this project. Our conda environment can be re-created
as follows:
```bib
conda env create -f environment.yml

conda activate neurotrial-analysis
```
Please follow [this docu](https://saturncloud.io/blog/how-to-use-conda-environment-in-a-jupyter-notebook/) to make it accessible in the Notebooks environment.

# 2. Data

## AACT
For our project, a static copy of the AACT database was downloaded on March 07 2024. Following the [installation instructions](https://aact.ctti-clinicaltrials.org/snapshots), 
a local PostgreSQL database was populated from the database file.

The _conditions_ table was joined with our compiled disease list of neurological conditions. We kept only the trials, which had a match. 
This resulted in 40’842 unique trials related to neurological conditions (of which 35’969 were registered as interventional trials).

Below are the columns extracted from different tables of the database.

| **AACT Table**         | **Columns**                                                                                      |
|-------------------------|-------------------------------------------------------------------------------------------------|
| **studies**            | brief_title, study_official_title, start_date, completion_date, phase, overall_status            |
| **brief_summaries**    | brief_summary_description                                                                        |
| **conditions**         | name                                                                                            |
| **interventions**      | intervention_name, intervention_type                                                            |
| **countries**          | name                                                                                            |
| **designs**            | primary_purpose, allocation, masking                                                            |
| **calculated_values**  | number_of_facilities, were_results_reported, months_to_report_results, number_of_primary_outcomes_to_measure, number_of_secondary_outcomes_to_measure, number_of_other_outcomes_to_measure |
| **sponsors**           | agency_class, lead_or_collaborator                                                              |

The official title (from table _ctgov.studies_) of each trial together with its short description (from table _ctgov.brief_summaries_) was extracted to a csv file and prepared for the information extraction steps below.

# 3. Extraction of condition (disease) and drug names 
## 3.1. NER annotation with a pre-trained NER model
In a previous work, we developed a dataset of annotated clinical trial registries for condition and different intervention types. We fine-tuned different models
and showed that the best-performing one for condition and drug entities was BioLinkBERT. We utilize that same model to annotate the full clinical trials dataset. See [https://github.com/Ineichen-Group/NeuroTrialNER](https://github.com/Ineichen-Group/NeuroTrialNER).

We use the official title and brief summaries as the text from which we infer the named entities (see [01-AACT-for-NER.ipynb)](./01-AACT-for-NER.ipynb).
The resulting file [aact_texts_46376.csv](./data/aact_for_ner/aact_texts_46376.csv) is processed by [run_ner_annotation.py](./src/run_ner_annotation.py).
This will save as output the annotations obtained from BioLinkBERT in [ner_annotations_BioLinkBERT-base_46376_20240621](./data/annotated_aact/ner_outputs/ner_annotations_BioLinkBERT-base_46376_20240621.csv).


## 3.2. Entities aggregation to abstract level
In the notebook [01-process-NER-annotations](./01-process-NER-annotations.ipynb) we aggregate the NER annotations to abstract level. 
This includes the steps:
1. From each abstract text, extract a dictionary of abbreviations and their long forms.
2. Use this dictionary to replace the named entities which are abbreviations with their long form.
3. Keep only the unique entities per trial abstract (from the BioLink entities and the AACT field).
4. Use a basic dictionary mapping approach to map the condition and drug entities to a canoncial representation.
5. Keep only the trials which have an annotation for drug intervention either from AACT or BioLinkBERT.

The final output from this notebook is in: 
- [aggregated_ner_annotations_basic_dict_mapped_19632.csv](./data/annotated_aact/ner_outputs/aggregated_ner_annotations_basic_dict_mapped_19632.csv).

# 4. Entities linking to SNOMED

Some relevant code can be found in [snomed-ct-entity-linking](https://github.com/CogStack/MedCAT/blob/3603dd293753982867470bcc72ae712572eb7803/medcat/utils/preprocess_snomed.py)
and [MedCAT](https://github.com/drivendataorg/snomed-ct-entity-linking/blob/main/1st%20Place/src/process_data.py).

## 4.1. Prediction of entities using SapBERT
We use the [SapBERT method](https://github.com/cambridgeltl/sapbert) to link the entities found by the NER system to the SNOMED ontology.
This includes the steps:

1. Download the SNOMED CT files, e.g., from [NIH SNOMED CT International Edition](https://www.nlm.nih.gov/healthit/snomedct/international.html) -> Download RF2 files. 
   We worked with version INT_20240401.
2. Process them to extract concepts, synonyms, and relationships between concepts.
3. Keep only the ones relevant for the application, here disorder and substance.
4. For all remaining SNOMED concepts and synonyms generate SapBERT embeddings.
5. For each abstract-level named entity, generate a SapBERT embedding, and find the closest concepts from the SNOMED embeddings.
6. If it is a synonym, map it to its canonical form.

The code for the above steps can be seen in notebook [00-1-working-with-snomed](./00-1-working-with-snomed.ipynb). 
For running the code over the whole dataset we used the script [sapbert_normalization.py](./src/sapbert_normalization.py).
It was then scheduled on a cluster with GPUs as shown in the bash script [run_parallel_sapbert_norm.sh](./src/run_parallel_sapbert_norm.sh).

The outputs are saved in:
- [sapbert_normalized_annotations_linkbert_19632_conditions.csv](./data/annotated_aact/snomed_linking_outputs/sapbert_normalized_annotations_linkbert_19632_conditions.csv)
- [sapbert_normalized_annotations_linkbert_19632_interventions.csv](./data/annotated_aact/snomed_linking_outputs/sapbert_normalized_annotations_linkbert_19632_interventions.csv)

## 4.2. Entities aggregation to higher-level SNOMED concepts
In this step we want to map fine-grained entities to a higher-level representation based on the SNOMED hierarchy.
The code for that is in notebook [00-2-linked-snomed-to-hierarchy](./00-2-linked-snomed-to-hierarchy.ipynb) and includes the steps:

1. For each SNOMED concept, extract the children and parent nodes.
2. Merge the mapped annotations with the hierarchical nodes representations.
3. Get all the snomed_id nodes from the dataframe and exclude generic entities, which should not be used as parents (e.g, node 64572001: Disease (disorder)). 
Please note that currently those values are hard-coded and might need to be revised for a new dataset.
4. For each linked entity concept, filter its the parent nodes based on whether they appear in the snomed_id node column.
5. The elements which have no parent are in the highest level in their hierarchy tree and all of their children should be mapped to that representation.
6. For each linked entity concept, keep only the parent nodes which do not have a parent of their own in the current dataframe.
7. If the linked concept is of another type, e.g. we are working with disease annotations, but it was linked to subtance node, we would keep the original entity.

## 4.3. Combining BioLinkBERT and AACT Annotations
The code for this step is in notebook [00-2-linked-snomed-to-hierarchy](./00-2-linked-snomed-to-hierarchy.ipynb).

To achieve a comprehensive list of annotations, we merged the annotations from both BioLinkBERT and AACT models based on the `nct_id`. Our merging strategy was as follows:

1. Preferred BioLinkBERT Annotations: Where both models provided annotations, we prioritized those from BioLinkBERT.
2. Fallback to AACT Annotations: If BioLinkBERT did not provide an annotation, we used the corresponding AACT annotation.
3. Special Case: When the BioLinkBERT annotation could not be mapped to a SNOMED (disorder) or (substance) node, but the AACT annotation could, we preferred the AACT annotation to ensure precise disorder identification.


# 5. Data visualization

## 5.1. Trials Metadata

The notebook [./02-AACT-trials-metadata.ipynb](./02-AACT-trials-metadata.ipynb) analyzes clinical trial data, covering study design, outcomes, funding, and global distribution, with insights into allocation, masking, and lead-collaborator trends.


## 5.2. Top 20 drugs and conditions

The notebook [./03-AACT-condition-intervention.ipynb
](./03-AACT-condition-intervention.ipynb) processes annotated data by cleaning, formatting, and preparing it for analysis, including manual grouping of conditions. It explores disease-specific trends, such as completed trials, trial phases, and time-based progress. Detailed analyses include heatmaps, logistic S-curves, linear regression, and percentage growth for diseases and drugs. Additionally, it investigates intervention-disease relationships, highlighting trends and identifying the most frequently associated interventions.

# 6. TSNE embeddings and visualisation
The code in this section is based on the work in ["The landscape of biomedical research"](https://github.com/berenslab/pubmed-landscape/tree/main).

## 6.1. Prepare data for embedding
Select the trials you want to embed, see [./01-prepare-AACT-for-NER-and-TSNE.ipynb](./01-prepare-AACT-for-NER-and-TSNE.ipynb).

## 6.2. Generate high-dimensional embeddings using a BERT model
The script [./src/embed_trial_summaries.py](./src/embed_trial_summaries.py) loads and preprocesses data, generates an embedding for each trial using a BERT-based model, and saves the embeddings to a file.

## 6.3. Map to low-dimensional space using TSNE
The code in [./06-tsne-pipeline-BERT.ipynb](./06-tsne-pipeline-BERT.ipynb) performs the optimization of a t-SNE embedding.
This process ensures a high-quality t-SNE embedding by initially separating clusters strongly (early exaggeration), gradually reducing the exaggeration to fine-tune the embedding (exaggeration annealing), and finally optimizing without exaggeration for a refined result.
The outputs is a low-dimensional representation (2D) of the high-dimensional data after the t-SNE optimization process. This embedding is then used for visualization or further analysis.

## 6.4. Color coding and cluster labelling
The visualization of the 2D points is performed in [./07-tsne-pipeline-colors.ipynb](./07-tsne-pipeline-colors.ipynb). We first assign to each trial a cluster based on the which disease or drug key words appear in the trial title/summary. The color code is then assigned depending on the trial cluster. 

The plotting is done with the help of the visualizations script [./src/plotting.py](./src/plotting.py).
