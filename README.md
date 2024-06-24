# NeuroTrialAnalysis
Analysis of clinical trials registry data in the field of neuroscience.

# 1. Set up the environment
TODO: add conda reqs

# 2. Data

## AACT
For our project, a static copy of the AACT database was downloaded on March 07 2024. Following the [installation instructions](https://aact.ctti-clinicaltrials.org/snapshots), 
a local PostgreSQL database was populated from the database file.

The _conditions_ table was joined with our compiled disease list of neurological conditions. We kept only the trials, which had a match. 
This resulted in 40’842 unique trials related to neurological conditions (of which 35’969 were registered as interventional trials).
The official title (from table _ctgov.studies_) of each trial together with its short description (from table _ctgov.brief_summaries_) was extracted to a csv file and prepared for annotation.

# 3. Inference of condition and drug names 
## 3.1. NER annotation with a pre-trained NER model
In a previous work, we developed a dataset of annotated clinical trial registries for condition and different intervention types. We fine-tuned different models
and showed that the best-performing one for condition and drug entities was BioLinkBERT. We utilize that same model to annotate the full clinical trials dataset.

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

The final output from this notebook is in [aggregated_ner_annotations_basic_dict_mapped_19632.csv](./data/annotated_aact/ner_outputs/aggregated_ner_annotations_basic_dict_mapped_19632.csv).


# 4. Entities linking to SNOMED

## 4.1. Prediction of entities using SapBERT
We use the [SapBERT method](https://github.com/cambridgeltl/sapbert) to link the entities found by the NER system to the SNOMED ontology.
This includes the steps:
1. Download the SNOMED CT files.
2. Process them to extract concepts, synonyms, and relationships between concepts.
3. Keep only the ones relevant for the application, here disorder and substance.
4. For all remaining SNOMED concepts and synonyms generate SapBERT embeddings.
5. For each abstract-level named entity, generate a SapBERT embedding, and find the closest concepts from the SNOMED embeddings.
6. If it is a synonym, map it to its canonical form.


## 4.2. Entities aggregation to higher-level concepts

# 5. TSNE embeddings and visualisation
