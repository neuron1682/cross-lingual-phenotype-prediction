import sys
sys.path.append('/home/neuron/PycharmProjects/cross-lingual-phenotype-prediction')
from dataset_creation.src.codiesp import get_codieesp_icd10_codes

import pickle


def create_codiesp_diagnosis(codiesp_src_path): 
    codiesp_dia, codiesp_proc = get_codieesp_icd10_codes.get_data_codiesp(codiesp_src_path) 
    return codiesp_dia, codiesp_proc


def align_codie(codie_df): 
    'capitalize ICD10 codes and remove dots to align with MIMIC'

    codie_df['ICD10'] = codie_df.apply(lambda x: x['icd10'].strip().upper().replace('.',''), axis=1)
    codie_df = codie_df.drop(['type', 'icd10', 'garbage'], axis=1)
    return codie_df.sort_values('ICD10')


def save_codie_labels(df, output_path):
    codie_labels = list(df.rename(columns={'CCS CATEGORY DESCRIPTION': 'labels'}).labels.dropna().unique()) 

    with open(output_path.format('ccs_codie'), 'wb') as f: 
        pickle.dump(codie_labels, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_codie_labels(labels_output_path):
    with open(labels_output_path.format('ccs_codie'), 'rb') as f: 
        codie_labels = pickle.load(f)
    return codie_labels

def get_codie_ccs_patient(codie_df): 
    '''
        load codie per patient
    '''

    codie_samples = codie_df.groupby('patient_id')['CCS CATEGORY DESCRIPTION'].apply(set).apply(list)
    codie_samples = codie_samples.reset_index().rename(columns={'CCS CATEGORY DESCRIPTION': 'labels'})
    
    return codie_samples

def merge_codiesp_label_notes(df, notes_path): 
    import os
    
    df['notes'] = None
    paths = [notes_path.format(curr_path) for curr_path in ['train', 'test', 'dev']]
    
    for path in paths:
        for file in os.listdir(path): 
            patient_id = file.replace('.txt','')
            with open(path + file) as f:
                lines = f.readlines()
                note = ' '.join(lines)
            try:
                df.loc[df.patient_id == patient_id, 'notes'] = note
            except: 
                continue

    return df[['patient_id', 'notes', 'labels']]

def map_icd10_achepa_diag(achepa_diag, codie_icd10, achepa_labels):
   
    achepa_diag = achepa_diag.rename(columns={'ICD-10': 'ICD10'}) 
    codie_achepa_diag = pd.merge(codie_icd10, 
                                achepa_diag, 
                                on='ICD10', 
                                how='left').dropna()

    codie_achepa_diag = codie_achepa_diag[codie_achepa_diag['diagnosis general'].isin(achepa_labels)]
    codie_achepa_diag = codie_achepa_diag.groupby(['patient_id']).apply(lambda x: list(set(x['diagnosis general'])))
    codie_achepa_diag = codie_achepa_diag.reset_index().rename(columns={0: 'labels'})

    return codie_achepa_diag



def build_codiesp_label_english_notes(df, notes_path): 
    import os
    
    df['original_translation'] = None
    paths = [notes_path.format(curr_path) for curr_path in ['train', 'test', 'dev']]
    
    for path in paths:
        for file in os.listdir(path): 
            patient_id = file.replace('.txt','')
            with open(path + file) as f:
                lines = f.readlines()
                note = ' '.join(lines)
            try:
                df.loc[df.patient_id == patient_id, 'notes'] = note
            except: 
                continue

    return df[['patient_id', 'original_translation', 'labels']]

if __name__ == '__main__': 
    df = pd.read_csv('codie_tmp_with_labels.csv')
    notes_path = '/home/neuron/Documents/codiesp/final_dataset_v4_to_publish/{}/text_files_en/'
    build_codiesp_label_english_notes(df, notes_path)


