# cross-lingual-phenotype-prediction



How to use: 

**Create the datasets for the Clinical Phenotyping Task** 

1. download datasets from the sources 
Mimic : https://physionet.org/content/mimiciii/1.4/
CodiEsp : https://zenodo.org/record/3837305#.YeVsnLzMJhF 

2. prepare dataset and map to ccsr labels 

    Run in order: 

        1)  create CodiEsp train/dev/test splits `python dataset_creation/src/codiesp/pre_process_codie.py`

        2)  create Mimic train/dev/test splits `python dataset_creation/src/mimic/pre_process_mimic.py`


**Run experiments:**

First create a Docker environment with the provided Dockerfile  
or
create a conda environment and install all required packages from  `requirements.txt`

To run the hyperparameter optimisation for the **adapters** 
run: 

`python experiments/src/xl_outcome_prediction_adapter/multilingual_adapter_hpo.py`


for the **baseline methods and other knowledge transfer methods** run: 

` python experiments/src/baselines/hpo_spanish_baseline.py`

If no GPU is available `resources_per_trial` needs to be adjusted.







    
