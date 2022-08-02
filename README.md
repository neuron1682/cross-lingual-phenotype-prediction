# CROSS-LINGUAL KNOWLEDGE TRANSFER FOR CLINICAL PHENOTYPING

implemtation of the paper: 
http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.95.pdf


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


Citation: 
```
@InProceedings{papaioannou-EtAl:2022:LREC,
  author    = {Papaioannou, Jens-Michalis  and  Grundmann, Paul  and  van Aken, Betty  and  Samaras, Athanasios  and  Kyparissidis, Ilias  and  Giannakoulas, George  and  Gers, Felix  and  Loeser, Alexander},
  title     = {Cross-Lingual Knowledge Transfer for Clinical Phenotyping},
  booktitle      = {Proceedings of the Language Resources and Evaluation Conference},
  month          = {June},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {900--909},
  abstract  = {Clinical phenotyping enables the automatic extraction of clinical conditions from patient records, which can be beneficial to doctors and clinics worldwide. However, current state-of-the-art models are mostly applicable to clinical notes written in English. We therefore investigate cross-lingual knowledge transfer strategies to execute this task for clinics that do not use the English language and have a small amount of in-domain data available. Our results reveal two strategies that outperform the state-of-the-art: Translation-based methods in combination with domain-specific encoders and cross-lingual encoders plus adapters. We find that these strategies perform especially well for classifying rare phenotypes and we advise on which method to prefer in which situation. Our results show that using multilingual data overall improves clinical phenotyping models and can compensate for data sparseness.},
  url       = {https://aclanthology.org/2022.lrec-1.95}
}
```





    
