import sys
sys.path.append('/home/neuron/PycharmProjects/cross-lingual-phenotype-prediction')
from dataset_creation.src.utils import mimic_utils
from dataset_creation.src.utils import utils
from dataset_creation.src.utils import build_dataset_spanish_english_experiment, train_test_split_experiment
from dataset_creation.src.utils import codie_utils
import pandas as pd

if __name__== '__main__':

    SELECTOR = 'PART_2'
    task = 'codie_CCS'
    icd_10_dxccsr_paths = 'dataset_creation/input_files/DXCCSR_v2021-2.csv'
    mimic_src_path = '/home/neuron/PycharmProjects/data/mimiciii/1.4/'
    # created with GEMS (General Equivalence Mapping)
    diagnosis_icd9_icd10_mapper_path = 'dataset_creation/input_files/diagnosis_icd9_icd10.pcl'
    #LABELS OUTPUT PATH
    labels_output_path = 'dataset_creation/output_files/{}_labels.pcl'
    train_data_output_path = 'dataset_creation/output_files/{}'
    
    if SELECTOR in ['PART_1', 'ALL']:
        #load mimic, filter , map icd9 to icd 10 and get relevant sections
        mimic_df = mimic_utils.mimic_map_icd9_icd10(diagnosis_icd9_icd10_mapper_path, mimic_src_path, mimic_labels_path=None)
        mimic_df.to_csv('mimic_tmp.csv', index=False)
        
    if SELECTOR in ['PART_2', 'ALL']:

        mimic_df = pd.read_csv('mimic_tmp.csv')

        if task == 'codie_CCS':
            codie_labels = codie_utils.load_codie_labels(labels_output_path)
            mimic_df_notes = mimic_utils.map_filter_ccs(mimic_df, codie_labels, icd_10_dxccsr_paths)
            dataset_name = 'mimic_codiesp_filtered_CCS'
            labels = codie_labels

        elif task == 'achepa_diagnoses': 
            achepa_labels = achepa_utils.load_achepa_labels(labels_output_path=labels_output_path)
            achepa_diag = achepa_utils.get_diagnosis_icd_mapper(achepa_icd_diagnosis_path)
            mimic_df_notes = mimic_utils.map_icd10_achepa_diag(mimic_df=mimic_df, 
                                                            achepa_diag=achepa_diag, 
                                                            achepa_labels=achepa_labels)
            labels = achepa_labels
            dataset_name = 'mimic_achepa_filtered_diagnoses'
    
        # create sorting of codes and get position in array for each code
        label_to_pos, pos_to_label = train_test_split_experiment.label_to_pos_map(labels)
        
        #create numpy matrix for labels patinets x labels
        mimic_labels_array = train_test_split_experiment.label_to_tensor(mimic_df_notes, label_to_pos)

        # use stratified sampling to save train/dev/test
        #train_test_split_experiment.stratified_sampling_multilearn(mimic_df_notes, 
        #                                                    mimic_labels_array, 
        #                                                    train_data_output_path.format(dataset_name))
        train_test_split_experiment.load_mimic_paper_split(mimic_df_notes, train_data_output_path.format(dataset_name))

