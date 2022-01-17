import sys
sys.path.append('/home/neuron/PycharmProjects/cross-lingual-phenotype-prediction')
from dataset_creation.src.utils import codie_utils
from dataset_creation.src.utils import utils
from dataset_creation.src.utils import build_dataset_spanish_english_experiment, train_test_split_experiment


if __name__ == '__main__': 
    # keep all digits
    digits = -1
    #CODIESP PATHS
    codiesp_src_path = '/home/neuron/Documents/codiesp/final_dataset_v4_to_publish/{}/{}'
    notes_path = '/home/neuron/Documents/codiesp/final_dataset_v4_to_publish/{}/text_files/'
    #MAPPING CSVs ICD10 to  CCS Diagnosis
    icd_10_dxccsr_paths = 'dataset_creation/input_files/DXCCSR_v2021-2.csv' 
    #achepa_icd_diagnosis_path = 'src/input_files/achepa_data/ICD_Diagnosis_V4.xlsx'

    #LABELS OUTPUT PATH 
    labels_output_path = 'dataset_creation/output_files/{}_labels.pcl'
    train_data_output_path = 'dataset_creation/output_files/{}'

    task = 'codiesp_CCS'#'achepa_diagnoses_task'


    codiesp_dia, _ = codie_utils.create_codiesp_diagnosis(codiesp_src_path=codiesp_src_path)
    
    #REMOVE DOTS AND CAPITALIZE FOR ACHEPA AND CODIE 
    codie_icd10 = codie_utils.align_codie(codiesp_dia)
    #codie_icd10 = codie_utils.get_codie_icd10(codie_df,digits=digits)

    #map icd10 to CCS
    codie_ccs = utils.load_and_create_icd_dxccsr_mapping(icd10_mapper_path=icd_10_dxccsr_paths, df=codie_icd10)

    codie_utils.save_codie_labels(codie_ccs, output_path=labels_output_path)

    # create codie
    codie_df = codie_utils.get_codie_ccs_patient(codie_df=codie_ccs)

    # get all codes available in codie to determine labels
    labels = codie_utils.load_codie_labels(labels_output_path=labels_output_path)
    

    # create sorting of codes and get position in array for each code
    label_to_pos, pos_to_label = train_test_split_experiment.label_to_pos_map(labels)
    
    #create numpy matrix for labels patinets x labels
    codiesp_labels_array = train_test_split_experiment.label_to_tensor(codie_df, label_to_pos)

    # merge codiesp labels with notes
    codie_df_with_notes = codie_utils.merge_codiesp_label_notes(df=codie_df, notes_path=notes_path)

    # use stratified sampling to save train/dev/test
    dataset_name = 'codiesp_CCS'
    #train_test_split_experiment.stratified_sampling_multilearn(codie_df_with_notes, 
    #                                                        codiesp_labels_array, 
    #                                                        train_data_output_path.format(dataset_name))
    train_test_split_experiment.load_codie_paper_split(codie_df_with_notes, 
                                                      train_data_output_path.format(dataset_name))




