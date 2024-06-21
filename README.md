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
This will save as output the annotations obtained from BioLinkBERT.




## 3.2. Entities aggregation to abstract level

# 4. Entities linking to SNOMED

## 3.1. Prediction of entities using SapBERT

## 3.2. Entities aggregation to higher-level concepts