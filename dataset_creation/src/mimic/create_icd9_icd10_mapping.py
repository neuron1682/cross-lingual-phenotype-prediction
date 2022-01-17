from pickle import dump
import pandas as pd 
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import logging 

logging.basicConfig(level=logging.INFO)

def get_mimic(path):
    mimic_dia_d = pd.read_csv(path.format('D_ICD_DIAGNOSES.csv'))
    mimic_dia = pd.read_csv(path.format('DIAGNOSES_ICD.csv'))
    mimic_proc_d = pd.read_csv(path.format('D_ICD_PROCEDURES.csv'))
    mimic_proc = pd.read_csv(path.format('PROCEDURES_ICD.csv'))

    return mimic_dia, mimic_proc, mimic_dia_d, mimic_proc_d


def transform_maps(dia_map): 

    print('original map size', dia_map.shape)
    dia_map = dia_map.drop_duplicates(subset=['icd9'], keep='first')[['icd9', 'icd10']]
    print('deduped map size', dia_map.shape)
    keys = dia_map.icd9.tolist()
    values = dia_map.icd10.tolist()
    dia_map_dict = dict()

    for k,v in zip(keys,values):
        try:
            val = v.replace('.','').upper()
            if val not in ['NoDx', 'NODX']:
                dia_map_dict[k.replace('.','').upper()] = val
        except: 
            print(k,v)

    return dia_map_dict

def hierarchical_approx_match(approx, exact_match):
    exact_match['cut_digit_icd10'] = exact_match.icd10
    #8 is max len of icd codes
    for dig in range(7, 2, -1): 
        approx['cut_digit_icd10'] = approx.icd10.str[:dig]
        approx = approx.drop_duplicates(['icd9', 'cut_digit_icd10' ])
        counter = approx.groupby('icd9').count()

        approx = pd.merge(approx, counter[['icd10']], left_on=approx.icd9, right_on=counter.index)
        approx = approx.drop('key_0', axis=1)
        approx.columns = ['icd9', 'icd10', 'flag', 'cut_digit_icd10', 'count']

        tmp = approx[approx['count'] == 1]
        approx = approx[approx['count'] > 1]
        
        exact_match = pd.concat([exact_match, tmp])

        print('digits', dig,
              'exact match', len(exact_match),
              'approximate match', len(approx)
            )

        approx = approx.drop('count', axis=1)
        exact_match = exact_match.drop('count', axis=1)
    return exact_match, approx

def keep_first_match(m2):
    m2['cut_digit_icd10'] =  m2.drop_duplicates(subset=['icd9'], keep='first')[['icd10']]
    return m2[['icd9', 'cut_digit_icd10']].dropna()
  
def get_map_files(path, type): 

    if type == 'diagnosis':
        map2 = pd.read_csv(path.format('dias/2018_I9gem.txt'), delimiter='\t', header=None)
    elif type == 'procedures': 
        map2 = pd.read_csv(path.format('procs/gem_i9pcs.txt'), delimiter='\t', header=None)

    map2.columns = ['row']

    #using space instead of separator here/succesfully converting to three columns
    map2 = pd.DataFrame(map2.row.str.split(' ').apply(lambda row:[r for r in row if r!='']).tolist(),
                        columns = ['icd9', 'icd10', 'flag'])

    # create unique mapping for icd9 to icd10
    counter = map2.groupby('icd9').count()
    map2 = pd.merge(map2, 
                    counter[['icd10']], 
                    left_on=map2.icd9, 
                    right_on=counter.index)

    map2 = map2.drop('key_0', axis=1)
    map2.columns = ['icd9', 'icd10', 'flag', 'count']

    # removing diagnoses that have no match
    no_match = map2[map2.icd10 == 'NoDx']
    map2 = map2[map2.icd10 != 'NoDx']
    
    #
    #### iterativetly find unique mapping by reducing code size and dropping duplicates
    # TODO check if it works, test needed
    exact_match = map2[map2['count'] == 1]
    approximate_match = map2[map2['count'] > 1]

    print('no match', len(no_match),
        'exact match', len(exact_match),
        'approximate match', len(approximate_match)
    )

    approximate_match = approximate_match.drop('count', axis=1)
    exact_match = exact_match.drop('count', axis=1)

    transformed_exact_match, approx = hierarchical_approx_match(approximate_match, exact_match)
    #### END iterativetly find unique mapping by reducing code size and dropping duplicates

    # keep first match; neglecting if more than one icd10 code suits
    first_match = keep_first_match(map2)

    logging.info([no_match.shape, 
                transformed_exact_match.shape, 
                approx.shape])

    return transformed_exact_match, approx, first_match

def construct_dict(df): 
    keys = df.icd9.tolist()
    values = df.cut_digit_icd10.tolist()

    return dict(zip(keys, values))

def mimic_map_icd9_icd10(df, dia_map): 
    res = df.ICD9_CODE.map(dia_map)
    res = res[~res.isna()]
    return res.to_list()
    
def save_unique_mapping_file(df_mapper_dict, path, type):
    import pickle 
    with open(path.format(type, 'icd9_icd10.pcl'), 'wb') as f: 
        pickle.dump(df_mapper_dict, f)
    
def load_unique_mapping_file(path, type):
    import pickle 
    with open(path.format(type, 'icd9_icd10.pcl'), 'rb') as f: 
        icd9_icd10_map = pickle.load(f)
    return icd9_icd10_map

def reduce_diag_code_size(mapper, digits): 
    for key in mapper.keys(): 
        mapper[key] = mapper[key][:digits]
    return mapper

if __name__ == '__main__':

    path = 'src/input_files/mimic_data/{}'
    output_path = 'src/output/{}_{}'

    mimic_dia, mimic_proc, mimic_dia_d, mimic_proc_d = get_mimic(path)

    ### get diagnoses
    hierarchical_match, approx, first_match = get_map_files(path, type='diagnosis')
    dia_map = construct_dict(df=first_match)
    mimic_icd10_fm = mimic_map_icd9_icd10(mimic_dia, dia_map)
    dia_map = construct_dict(df=first_match)
    save_unique_mapping_file(df_mapper_dict=dia_map, path=output_path, type='diagnosis')
    icd9_icd10_map = load_unique_mapping_file(output_path, type='diagnosis')

    ### reduce size to three digits
    reduced_map = reduce_diag_code_size(icd9_icd10_map, digits=3)
    output_path = 'src/output/3_digit_{}_{}'
    save_unique_mapping_file(df_mapper_dict=dia_map, path=output_path, type='diagnosis')


    ### get procedures 
    hierarchical_match, approx, first_match = get_map_files(path,type='procedures')
    pcs_map = construct_dict(df=first_match)
    mimic_icd10_fm = mimic_map_icd9_icd10
    save_unique_mapping_file(df_mapper_dict=pcs_map, path=output_path, type='procedures')
    icd9_icd10_map = load_unique_mapping_file(output_path, type='procedures')

    ### reduce size to three digits
    reduced_map = reduce_diag_code_size(icd9_icd10_map, digits=3)
    output_path = 'src/output/3_digit_{}_{}'
    save_unique_mapping_file(df_mapper_dict=dia_map, path=output_path, type='procedures')

    





