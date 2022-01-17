import pandas  as pd 
from dataset_creation.src.mimic import dia_groups
from dataset_creation.src.codiesp import get_codieesp_icd10_codes as codiesp 

def merge_codiesp_label_notes(label_path, notes_path): 

    labels = pd.read_csv(label_path)
    labels['notes'] = None
    path = '/home/neuron/Documents/codiesp/final_dataset_v4_to_publish/{}/text_files/'
    paths = [path.format(curr_path) for curr_path in ['train', 'test', 'dev']]
    import os
    for path in paths:
        for file in os.listdir(path): 
            patient_id = file.replace('.txt','')
            with open(path + file) as f:
                lines = f.readlines()
                note = ' '.join(lines)
            try:
                labels.loc[labels.patient_id == patient_id, 'notes'] = note
            except: 
                continue
    #labels.to_csv('src/input_files/processed_datataset/codiesp_notes_df.csv', index=False)      
    return labels 
    

def merge_mimic_label_notes(mimic_label_path, notes_path, mapper_icd9_icd10_path): 

    labels = pd.read_csv(mimic_label_path)
    mimic_dia_mapper = dia_groups.load_unique_mapping_file(mapper_icd9_icd10_path, type='diagnosis')
    notes_diagnoses_df = dia_groups.dia_icd10_mimic_notes(mimic_dir=notes_path, 
                                                        mapper_dict=mimic_dia_mapper, 
                                                        admission_only=False)
    nt = notes_diagnoses_df[['SUBJECT_ID','HADM_ID','TEXT']]
    merged = pd.merge(labels, nt, on=['SUBJECT_ID', 'HADM_ID'], how = 'left')
    #merged.to_csv('src/input_files/processed_datataset/mimic_notes_df.csv', index=False) 
    return merged


    
if __name__ == '__main__': 
    codie_label_path = 'src/input_files/processed_datataset/codie_samples_labels.csv' 
    codie_notes_path = '/home/neuron/Documents/codiesp/final_dataset_v4_to_publish/{}/{}'
    mimic_label_path = 'src/input_files/processed_datataset/mimic_samples_labels.csv' 
    mapper_icd9_icd10_path = 'src/output/{}_{}'
    mimic_notes_path = '/home/neuron/PycharmProjects/data/mimiciii/1.4/'

    labels = merge_codiesp_label_notes(label_path=codie_label_path, notes_path=codie_notes_path)
    merged = merge_mimic_label_notes(mimic_label_path=mimic_label_path, notes_path=mimic_notes_path, mapper_icd9_icd10_path=mapper_icd9_icd10_path)

    