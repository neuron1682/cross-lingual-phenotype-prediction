import os
from typing import List
import pandas as pd
from dataset_creation.src.mimic.create_icd9_icd10_mapping import load_unique_mapping_file
import dataset_creation.src.utils.mimic_utils as mimic_utils

def dia_icd10_mimic_notes(mimic_dir: str, mapper_dict, admission_only: bool):
    """
    Extracts information needed for the task from the MIMIC dataset. Namely "TEXT" column from NOTEEVENTS.csv and
    "ICD9_CODE" from DIAGNOSES_ICD.csv. Divide all ICD9 codes' first three digits and group them per admission into
    column "SHORT_CODES".

    !Keep patient once!

    """

    # set task name
    task_name = "DIA_GROUPS"

    if admission_only:
        task_name = f"{task_name}_adm"

    # load dataframes
    mimic_diagnoses = pd.read_csv(os.path.join(mimic_dir, "DIAGNOSES_ICD.csv"))
    mimic_notes = pd.read_csv(os.path.join(mimic_dir, "NOTEEVENTS.csv"))
    mimic_admissions = pd.read_csv(os.path.join(mimic_dir, "ADMISSIONS.csv"))

    # filter notes
    mimic_notes = mimic_utils.filter_notes(mimic_notes, 
                                        mimic_admissions, 
                                        admission_text_only=admission_only)

    # only keep relevant columns
    mimic_diagnoses = mimic_diagnoses[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']]
    #mimic_diagnoses = mimic_diagnoses[['SUBJECT_ID', 'ICD9_CODE']]

    # drop all rows without diagnosis codes
    mimic_diagnoses = mimic_diagnoses.dropna(how='any', 
                                            subset=['ICD9_CODE'], 
                                            axis=0)

    # create column SHORT_CODE including first 3 digits of ICD9 code
    mimic_diagnoses["SHORT_CODE"] = mimic_diagnoses.ICD9_CODE.astype(str)
    mimic_diagnoses["SHORT_CODE"] = mimic_diagnoses.SHORT_CODE.map(mapper_dict)
    # drop all rows without diagnosis codes
    mimic_diagnoses = mimic_diagnoses.dropna(how='any', 
                                            subset=['SHORT_CODE'], 
                                            axis=0)

    # remove duplicated code groups per admission
    #mimic_diagnoses = mimic_diagnoses.drop_duplicates(["HADM_ID", "SHORT_CODE"])
    mimic_diagnoses = mimic_diagnoses.drop_duplicates(["SUBJECT_ID","SHORT_CODE"])
    
    # store all ICD codes for vectorization
    #icd10_codes = mimic_diagnoses.SHORT_CODE.unique().tolist()
    grouped_codes = mimic_diagnoses.groupby(['HADM_ID', 'SUBJECT_ID'])['SHORT_CODE'].apply(
                                                        lambda d: ",".join(d.astype(str))).reset_index()

    #grouped_codes = mimic_diagnoses.groupby(['SUBJECT_ID'])['SHORT_CODE'].apply(
    #                                                    lambda d: ",".join(d.astype(str))).reset_index()

    # rename column
    grouped_codes = grouped_codes.rename(columns={'SHORT_CODE': 'SHORT_CODES'})

    # merge discharge summaries into diagnosis table
    notes_diagnoses_df = pd.merge(
        grouped_codes[['HADM_ID', 'SHORT_CODES']], mimic_notes, how='inner', on='HADM_ID')

    #notes_diagnoses_df = pd.merge(
    #    grouped_codes[['SUBJECT_ID', 'SHORT_CODES']], mimic_notes, how='inner', on='SUBJECT_ID')
    
    return notes_diagnoses_df

def dia_icd10_names_mimic(mimic_dir: str, mapper_dict, admission_only: bool):
    """
    Extracts information needed for the task from the MIMIC dataset. Namely "TEXT" column from NOTEEVENTS.csv and
    "ICD9_CODE" from DIAGNOSES_ICD.csv. Divide all ICD9 codes' first three digits and group them per admission into
    column "SHORT_CODES".
    """

    # set task name
    task_name = "DIA_GROUPS"

    if admission_only:
        task_name = f"{task_name}_adm"

    # load dataframes
    mimic_diagnoses = pd.read_csv(os.path.join(mimic_dir, "DIAGNOSES_ICD.csv"))
    mimic_diagnoses_names = pd.read_csv(os.path.join(mimic_dir, "D_ICD_DIAGNOSES.csv"))[['ICD9_CODE', 'LONG_TITLE']]

    mimic_notes = pd.read_csv(os.path.join(mimic_dir, "NOTEEVENTS.csv"))
    mimic_admissions = pd.read_csv(os.path.join(mimic_dir, "ADMISSIONS.csv"))

    # filter notes
    mimic_notes = mimic_utils.filter_notes(mimic_notes, 
                                        mimic_admissions, 
                                        admission_text_only=admission_only)

    # only keep relevant columns
    mimic_diagnoses = mimic_diagnoses[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']]
    #mimic_diagnoses = mimic_diagnoses[['SUBJECT_ID', 'ICD9_CODE']]

    # drop all rows without diagnosis codes
    mimic_diagnoses = mimic_diagnoses.dropna(how='any', 
                                            subset=['ICD9_CODE'], 
                                            axis=0)

    # create column SHORT_CODE including first 3 digits of ICD9 code
    mimic_diagnoses["SHORT_CODE"] = mimic_diagnoses.ICD9_CODE.astype(str)
    mimic_diagnoses["SHORT_CODE"] = mimic_diagnoses.SHORT_CODE.map(mapper_dict)

    # drop all rows without diagnosis codes
    mimic_diagnoses = mimic_diagnoses.dropna(how='any', 
                                            subset=['SHORT_CODE'], 
                                            axis=0)

    # remove duplicated code groups per admissio
    #mimic_diagnoses = mimic_diagnoses.drop_duplicates(["HADM_ID", "SHORT_CODE"])
    #mimic_diagnoses = mimic_diagnoses.drop_duplicates(["SUBJECT_ID", "SHORT_CODE"])

    mimic_diagnoses = pd.merge( mimic_diagnoses[['SUBJECT_ID','HADM_ID', 'SHORT_CODE','ICD9_CODE']], 
                                mimic_notes[['SUBJECT_ID', 'HADM_ID']], 
                                how='inner', 
                                on=['SUBJECT_ID', 'HADM_ID'])

    #mimic_diagnoses = pd.merge(mimic_diagnoses[['SUBJECT_ID', 'SHORT_CODE','ICD9_CODE']], 
    #                        mimic_notes[['SUBJECT_ID']].drop_duplicates('SUBJECT_ID', keep='last'), 
    #                        how='inner', 
    #                        on=['SUBJECT_ID'])
    
    # store all ICD codes for vectorization
    #icd10_codes = mimic_diagnoses.SHORT_CODE.unique().tolist()
    
    mimic_diagnoses = pd.merge(mimic_diagnoses, 
                            mimic_diagnoses_names, 
                            on="ICD9_CODE", 
                            how='left')

    return mimic_diagnoses



if __name__ == "__main__":
    #args = mimic_utils.parse_args()
    #dia_groups_3_digits_mimic(
    #    args.mimic_dir, args.save_dir, args.seed, args.admission_only)

    mapper_path = 'src/output/{}_{}'
    path = '/home/neuron/PycharmProjects/data/mimiciii/1.4/'
    mimic_labels_path = '/home/neuron/PycharmProjects/multi-clinical-dataset-creation/src/output/{}'

    icd9_icd10_map = load_unique_mapping_file(path=mapper_path, type='diagnosis')

    notes_diagnoses_df = dia_icd10_names_mimic(mimic_dir=path, 
                                            mapper_dict=icd9_icd10_map, 
                                            admission_only=False)

    notes_diagnoses_df.to_csv(mimic_labels_path.format('mimic_labels.csv'), index=False)

    #use same groupings as provided in achepa excel