import os
import pandas as pd
from  dataset_creation.src.mimic import create_icd9_icd10_mapping
from dataset_creation.src.utils import utils

def filter_admission_brief_hospital_course_text(notes_df) -> pd.DataFrame:
    """
    Filter text information by section and only keep sections that are known on admission time + BRIEF HOSPITAL COURSE
    """
    admission_sections = {
        "CHIEF_COMPLAINT": "chief complaint:",
        "PRESENT_ILLNESS": "present illness:",
        "MEDICAL_HISTORY": "medical history:",
        "MEDICATION_ADM": "medications on admission:",
        "ALLERGIES": "allergies:",
        "PHYSICAL_EXAM": "physical exam:",
        "FAMILY_HISTORY": "family history:",
        "SOCIAL_HISTORY": "social history:", 
        "BRIEF_HOSPITAL_COURSE": "brief hospital course:"
    }

    # replace linebreak indicators
    notes_df['TEXT'] = notes_df['TEXT'].str.replace(r"\n", r"\\n")

    # extract each section by regex
    for key in admission_sections.keys():
        section = admission_sections[key]
        notes_df[key] = notes_df.TEXT.str.extract(r'(?i){}(.+?)\\n\\n[^(\\|\d|\.)]+?:'
                                                  .format(section))

        notes_df[key] = notes_df[key].str.replace(r'\\n', r' ')
        notes_df[key] = notes_df[key].str.strip()
        notes_df[key] = notes_df[key].fillna("")
        notes_df[notes_df[key].str.startswith("[]")][key] = ""

    # filter notes with missing main information
    notes_df = notes_df[(notes_df.CHIEF_COMPLAINT != "") | (notes_df.PRESENT_ILLNESS != "") |
                        (notes_df.MEDICAL_HISTORY != "")| (notes_df.BRIEF_HOSPITAL_COURSE != "")]

    # add section headers and combine into TEXT_ADMISSION
    notes_df = notes_df.assign(TEXT="CHIEF COMPLAINT: " + notes_df.CHIEF_COMPLAINT.astype(str)
                                    + '\n\n' +
                                    "PRESENT ILLNESS: " + notes_df.PRESENT_ILLNESS.astype(str)
                                    + '\n\n' +
                                    "MEDICAL HISTORY: " + notes_df.MEDICAL_HISTORY.astype(str)
                                    + '\n\n' +
                                    "MEDICATION ON ADMISSION: " + notes_df.MEDICATION_ADM.astype(str)
                                    + '\n\n' +
                                    "ALLERGIES: " + notes_df.ALLERGIES.astype(str)
                                    + '\n\n' +
                                    "PHYSICAL EXAM: " + notes_df.PHYSICAL_EXAM.astype(str)
                                    + '\n\n' +
                                    "FAMILY HISTORY: " + notes_df.FAMILY_HISTORY.astype(str)
                                    + '\n\n' +
                                    "SOCIAL HISTORY: " + notes_df.SOCIAL_HISTORY.astype(str)
                                    + '\n\n' +
                                    "BRIEF HOSPITAL COURSE: " + notes_df.BRIEF_HOSPITAL_COURSE.astype(str)
                                    )

    return notes_df

def filter_notes(notes_df: pd.DataFrame, admissions_df: pd.DataFrame, admission_text_only=False) -> pd.DataFrame:
    """
    Keep only Discharge Summaries and filter out Newborn admissions. Replace duplicates and join reports with
    their addendums. If admission_text_only is True, filter all sections that are not known at admission time.
    """
    # filter out newborns
    adm_grownups = admissions_df[admissions_df.ADMISSION_TYPE != "NEWBORN"]
    notes_df = notes_df[notes_df.HADM_ID.isin(adm_grownups.HADM_ID)]

    # remove notes with no TEXT or HADM_ID
    notes_df = notes_df.dropna(subset=["TEXT", "HADM_ID"])

    # filter discharge summaries
    notes_df = notes_df[notes_df.CATEGORY == "Discharge summary"]

    # remove duplicates and keep the later ones
    notes_df = notes_df.sort_values(by=["CHARTDATE"])
    notes_df = notes_df.drop_duplicates(subset=["TEXT"], keep="last")

    # combine text of same admissions (those are usually addendums)
    combined_adm_texts = notes_df.groupby('HADM_ID')['TEXT'].apply(lambda x: '\n\n'.join(x)).reset_index()
    notes_df = notes_df[notes_df.DESCRIPTION == "Report"]
    notes_df = notes_df[["HADM_ID", "ROW_ID", "SUBJECT_ID", "CHARTDATE"]]
    notes_df = notes_df.drop_duplicates(subset=["HADM_ID"], keep="last")
    notes_df = pd.merge(combined_adm_texts, notes_df, on="HADM_ID", how="inner")

    # strip texts from leading and trailing and white spaces
    notes_df["TEXT"] = notes_df["TEXT"].str.strip()

    # remove entries without admission id, subject id or text
    notes_df = notes_df.dropna(subset=["HADM_ID", "SUBJECT_ID", "TEXT"])


    notes_df = filter_admission_brief_hospital_course_text(notes_df)

    return notes_df

def dia_all_codes_mimic(mimic_dir, mapper_dict, admission_only):
    """
    Extracts information needed for the task from the MIMIC dataset. Namely "TEXT" column from NOTEEVENTS.csv and
    "ICD9_CODE" from DIAGNOSES_ICD.csv. Groups all ICD9 codes per admission into column "ICD9_CODES".
    Creates 70/10/20 split over patients for train/val/test sets.
    """

    # set task name
    task_name = "DIA_ALL"
    if admission_only:
        task_name = f"{task_name}_adm"

    # load dataframes
    mimic_diagnoses = pd.read_csv(os.path.join(mimic_dir, "DIAGNOSES_ICD.csv"))
    mimic_notes = pd.read_csv(os.path.join(mimic_dir, "NOTEEVENTS.csv"))
    mimic_admissions = pd.read_csv(os.path.join(mimic_dir, "ADMISSIONS.csv"))

    # filter notes
    mimic_notes = filter_notes(mimic_notes, mimic_admissions, admission_text_only=admission_only)

    # only keep relevant columns
    mimic_diagnoses = mimic_diagnoses[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']]

    # drop all rows without diagnoses codes
    mimic_diagnoses = mimic_diagnoses.dropna(how='any', subset=['ICD9_CODE', 'HADM_ID'], axis=0)
    mimic_diagnoses["ICD10"] = mimic_diagnoses.ICD9_CODE.map(mapper_dict)
    mimic_diagnoses = mimic_diagnoses.dropna(how='any', subset=['ICD10'], axis=0)

    # group by admission and join diagnoses codes into one column
    combined_diagnoses = mimic_diagnoses.groupby(['HADM_ID', 'SUBJECT_ID'])['ICD10'].apply(set).apply(list).reset_index()

    # merge discharge summaries into diagnoses table
    notes_diagnoses_df = pd.merge(combined_diagnoses[['HADM_ID', 'ICD10']], mimic_notes, how='inner', on='HADM_ID')

    return notes_diagnoses_df

def mimic_map_icd9_icd10(diagnosis_icd9_icd10_mapper_path, mimic_src_path, mimic_labels_path):
    
    icd9_icd10_map = create_icd9_icd10_mapping.load_unique_mapping_file(path=diagnosis_icd9_icd10_mapper_path, type='diagnosis')
    notes_diagnoses_df = dia_all_codes_mimic(mimic_dir=mimic_src_path, mapper_dict=icd9_icd10_map, admission_only=False)
    return notes_diagnoses_df


def map_filter_ccs(mimic_df, all_codes,icd_10_dxccsr_paths): 

    mimic_df['ICD10'] = mimic_df['ICD10'].apply(eval)
    mimic_df = mimic_df.sort_values('CHARTDATE')
    mimic_df = mimic_df.drop_duplicates(subset=["SUBJECT_ID"], keep="last")

    mimic_df = mimic_df.explode('ICD10')
    #map icd10 to CCS
    mimic_ccs = utils.load_and_create_icd_dxccsr_mapping(icd10_mapper_path=icd_10_dxccsr_paths, df=mimic_df)
    mimic_ccs_filtered = mimic_ccs[mimic_ccs['CCS CATEGORY DESCRIPTION'].isin(all_codes)]


    mimic_ccs_per_patient = mimic_ccs_filtered.groupby(['HADM_ID', 
                                    'TEXT', 
                                    'ROW_ID', 
                                    'SUBJECT_ID'])['CCS CATEGORY DESCRIPTION'].apply(set).apply(list).reset_index()
    mimic_ccs_per_patient = mimic_ccs_per_patient[mimic_ccs_per_patient['CCS CATEGORY DESCRIPTION'].str.len() > 0]
    mimic_ccs_per_patient = mimic_ccs_per_patient.rename(columns = {'CCS CATEGORY DESCRIPTION': 'labels'})
    return mimic_ccs_per_patient

def map_icd10_achepa_diag(achepa_diag, mimic_df, achepa_labels):
   
    mimic_df['ICD10'] = mimic_df['ICD10'].apply(eval)
    mimic_df = mimic_df.sort_values('CHARTDATE')
    mimic_df = mimic_df.drop_duplicates(subset=["SUBJECT_ID"], keep="last")

    mimic_df = mimic_df.explode('ICD10')

    achepa_diag = achepa_diag.rename(columns={'ICD-10': 'ICD10'}) 


    mimic_achepa_diag = pd.merge(mimic_df, 
                                achepa_diag, 
                                on='ICD10', 
                                how='left').dropna()

    mimic_achepa_diag = mimic_achepa_diag[mimic_achepa_diag['diagnosis general'].isin(achepa_labels)]


    mimic_achepa_diag_per_patient = mimic_achepa_diag.groupby(['HADM_ID', 
                                    'TEXT', 
                                    'ROW_ID', 
                                    'SUBJECT_ID'])['diagnosis general'].apply(set).apply(list).reset_index()

    mimic_achepa_diag_per_patient = mimic_achepa_diag_per_patient[mimic_achepa_diag_per_patient['diagnosis general'].str.len() > 0]
    mimic_achepa_diag_per_patient = mimic_achepa_diag_per_patient.rename(columns={'diagnosis general': 'labels'})

    return mimic_achepa_diag_per_patient


    
