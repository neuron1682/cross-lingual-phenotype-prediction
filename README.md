# cross-lingual-phenotype-prediction



How to use: 

**Create the datasets for the Clinical Phenotyping Task** 

1. download datasets from the sources 

    a)  Mimic : https://physionet.org/content/mimiciii/1.4/
    
    b)  CodiEsp : https://zenodo.org/record/3837305#.YeVsnLzMJhF 

2. prepare dataset and map to ccsr labels 

    Run in order: 

        1)  create CodiEsp train/dev/test splits 
        
    `python dataset_creation/src/codiesp/pre_process_codie.py`

        2)  create Mimic train/dev/test splits 
    `python dataset_creation/src/mimic/pre_process_mimic.py`


**Run experiments:**

-   Now create a Docker environment with the provided   Dockerfile

- the created datasets will be copied into the `/pvc/output_files` folder.

Run the **hyperparameter optimisation** 

1) for the **adapters** 
 adjust paths and settings in `experiments/src/xl_outcome_prediction_adapter/multilingual_adapter_hpo.py` and execute the file.


2) for the **baseline methods and other knowledge transfer methods** adjust paths and settings in `experiments/src/baselines/hpo_spanish_baseline.py` and execute the file.









    
