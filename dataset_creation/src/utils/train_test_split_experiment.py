import pandas as pd 
import numpy as np

def label_to_pos_map(all_codes):
        label_to_pos = dict([(code,pos) for code, pos in zip(sorted(all_codes),range(len(all_codes)))])
        pos_to_label = dict([(pos,code) for code, pos in zip(sorted(all_codes),range(len(all_codes)))])
        return label_to_pos, pos_to_label


def label_to_tensor(data, label_to_pos):

    tmp = np.zeros((len(data), 
                    len(label_to_pos)))

    c = 0
    for idx, row in data.iterrows():
        for code in row['labels']:
                tmp[c, label_to_pos[code]] = 1
        c += 1

    return tmp


def stratified_sampling_multilearn(df, y, train_data_output_path): 

    from skmultilearn.model_selection import iterative_train_test_split
    from skmultilearn.model_selection import IterativeStratification

    df = df.reset_index(drop=True).sample(frac=1, random_state=42)
    k_fold = IterativeStratification(n_splits=3, order=1, random_state=42)

    nfold = 1 
    for train, test in k_fold.split(df, y):
        df_train = df.iloc[train]
        y_train = y[train, :]

        df_test = df.iloc[test]
        y_test = y[test, :]
        val_tmp, y_val, df_test_tmp, y_test = iterative_train_test_split(df_test.values, y_test, test_size = 0.5,)
        df_val = pd.DataFrame(val_tmp, columns=df_test.columns)
        df_test = pd.DataFrame(df_test_tmp, columns=df_test.columns)

        df_train.to_csv(f"{train_data_output_path}_fold_{nfold}_train.csv", index=False)
        df_val.to_csv(f"{train_data_output_path}_fold_{nfold}_dev.csv", index=False)
        df_test.to_csv(f"{train_data_output_path}_fold_{nfold}_test.csv", index=False)
        
        nfold = nfold + 1

def load_mimic_paper_split(df, train_data_output_path):

    dev_patients = pd.read_csv('dataset_creation/input_files/ids_mimic_dev.csv')
    test_patients = pd.read_csv('dataset_creation/input_files/ids_mimic_test.csv')
    train_patients = pd.read_csv('dataset_creation/input_files/ids_mimic_train.csv')

    df_train = df[df.HADM_ID.isin(train_patients.HADM_ID)]
    df_test = df[df.HADM_ID.isin(test_patients.HADM_ID)]
    df_val = df[df.HADM_ID.isin(dev_patients.HADM_ID)]

    df_train.to_csv(f"{train_data_output_path}_train.csv", index=False)
    df_val.to_csv(f"{train_data_output_path}_dev.csv", index=False)
    df_test.to_csv(f"{train_data_output_path}_test.csv", index=False)


def load_codie_paper_split(df, train_data_output_path):

    dev_patients = pd.read_csv('dataset_creation/input_files/ids_codie_dev.csv')
    test_patients = pd.read_csv('dataset_creation/input_files/ids_codie_test.csv')
    train_patients = pd.read_csv('dataset_creation/input_files/ids_codie_train.csv')

    df_train = df[df.patient_id.isin(train_patients.patient_id)]
    df_test = df[df.patient_id.isin(test_patients.patient_id)]
    df_val = df[df.patient_id.isin(dev_patients.patient_id)]

    df_train.to_csv(f"{train_data_output_path}_train.csv", index=False)
    df_val.to_csv(f"{train_data_output_path}_dev.csv", index=False)
    df_test.to_csv(f"{train_data_output_path}_test.csv", index=False)
        