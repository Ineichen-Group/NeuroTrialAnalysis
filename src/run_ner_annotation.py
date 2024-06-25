from ner_models import NERModel
import datetime
import pandas as pd

# OUTPUT
path_to_save_output_annotations = "./data/annotated_aact/ner_outputs/"

short_to_long_class_names_map = {
    "DRUG": "DRUG",
    "BEH": "BEHAVIOURAL",
    "SURG": "SURGICAL",
    "PHYS": "PHYSICAL",
    "RADIO": "RADIOTHERAPY",
    "OTHER": "OTHER",
    "COND": "CONDITION",
    "CTRL": "CONTROL"
}

def run_inference_hugging_face_model(input_data_path_csv, hugging_face_model_name, hugging_face_model_path, group_entities_custom = False):
    model_name_str = "bert-base-uncased"
    if "/" in hugging_face_model_name:
        model_name_str = hugging_face_model_name.split("/")[1]

    model = NERModel("huggingface", hugging_face_model_name, hugging_face_model_path, short_to_long_class_names_map, use_custom_entities_grouping = group_entities_custom)

    tmp = pd.read_csv(input_data_path_csv)
    ### ANNOTATE WITH TUPLE OUTPUT
    annotated_ds = model.annotate(input_data_path_csv, "text")
    output_path = path_to_save_output_annotations + "ner_annotations_{}_{}_{}.csv".format(model_name_str, len(tmp),
                                                                                                current_date)
    annotated_ds.to_csv(output_path, sep=",")
    print(f"Tuple annotations for {model_name_str} saved in {output_path}.")


if __name__ == '__main__':

    current_date = datetime.datetime.now().strftime("%Y%m%d")
    input_data_path = "./data/aact_for_ner/aact_texts_46376.csv"
    #### LinkBERT ####
    print("Running LinkBERT model_annotations.")
    hugging_face_model_name = "michiyasunaga/BioLinkBERT-base"
    hugging_face_model_path = "./ner_model/michiyasunaga_biolinkbert/epochs_15_data_size_100_iter_4/"
    run_inference_hugging_face_model(input_data_path, hugging_face_model_name, hugging_face_model_path)

