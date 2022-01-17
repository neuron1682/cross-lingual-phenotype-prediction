import pandas as pd 
import numpy as np


def load_and_create_icd_dxccsr_mapping(icd10_mapper_path, df): 


    icd10_mapper = pd.read_csv(icd10_mapper_path)
    
    icd10_mapper = icd10_mapper[["'ICD-10-CM CODE'", 
                                "'ICD-10-CM CODE DESCRIPTION'",
                                "'CCSR CATEGORY 1'", 
                                "'CCSR CATEGORY 1 DESCRIPTION'"]
                                ].rename(columns={"'ICD-10-CM CODE'": 'ICD-10', 
                                                "'ICD-10-CM CODE DESCRIPTION'": 'ICD-10 Description',
                                                "'CCSR CATEGORY 1'": 'CCS CATEGORY',
                                                "'CCSR CATEGORY 1 DESCRIPTION'": 'CCS CATEGORY DESCRIPTION'
                                                })

    icd10_mapper['ICD-10'] = icd10_mapper['ICD-10'].str.strip("'")
    df = df.rename(columns={'ICD10': 'ICD-10'})
    
    df = pd.merge(df.rename(columns={'ICD10': 'ICD-10'}), 
                        icd10_mapper[['ICD-10', 'CCS CATEGORY DESCRIPTION']], 
                        on='ICD-10', 
                        how='inner')

    
    return df