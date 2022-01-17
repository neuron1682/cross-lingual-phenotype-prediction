import pandas as pd 

def get_data_codiesp(path):
    'load labels of codiesp and concatenate all diagnoses with names, same for procedures'
    
    name="dev"
    codie_sp_dev = pd.read_csv(path.format(name, name + 'X.tsv'), delimiter='\t', header=None)
    name="train"
    codie_sp_train = pd.read_csv(path.format(name, name + 'X.tsv'), delimiter='\t', header=None)
    name="test"
    codie_sp_test = pd.read_csv(path.format(name, name + 'X.tsv'), delimiter='\t', header=None)

    codiesp = pd.concat([codie_sp_dev, codie_sp_train, codie_sp_test],)
    codiesp.columns = ['patient_id', 'type', 'icd10','name', 'garbage']

    codiesp_dia = codiesp[codiesp.type == 'DIAGNOSTICO']
    codiesp_proc = codiesp[codiesp.type == 'PROCEDIMIENTO']


    return codiesp_dia, codiesp_proc

if __name__ == '__main__':
    path = '/home/neuron/Documents/codiesp/final_dataset_v4_to_publish/{}/{}'
    codiesp_dia, codiesp_proc = get_data_codiesp(path)
    pass