import sys
sys.path.append('/pvc/')
from src.xl_outcome_prediction_adapter.multilingual_adapter import *
from src.utils import utils
import pickle
import torch
import numpy as np
from sklearn.metrics import roc_auc_score as auroc


def get_data_paths(nfold, filter_set_name, eval_dataset, translator_data_selector):

        data_paths = {'train_data_path_mimic': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/mimic_codiesp_filtered_CCS_fold_{nfold}_train.csv",
                    'validation_data_path_mimic': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/mimic_codiesp_filtered_CCS_fold_{nfold}_dev.csv",
                    'test_data_path_mimic': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/mimic_codiesp_filtered_CCS_fold_{nfold}_test.csv",
                    
                    'train_data_path_achepa': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/achepa_codiesp_filtered_CCS_fold_{nfold}_train.csv",
                    'validation_data_path_achepa': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/achepa_codiesp_filtered_CCS_fold_{nfold}_dev.csv",
                    'test_data_path_achepa': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/achepa_codiesp_filtered_CCS_fold_{nfold}_test.csv",

                    'train_data_path_codie': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/codiesp_CCS_fold_{nfold}_train.csv",
                    'validation_data_path_codie': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/codiesp_CCS_fold_{nfold}_dev.csv",
                    'test_data_path_codie': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/codiesp_CCS_fold_{nfold}_test.csv", 

                    #'translation_train_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/codiesp_WITH_TRANS_fold_{nfold}_train.csv",
                    #'translation_validation_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/codiesp_WITH_TRANS_fold_{nfold}_dev.csv",
                    #'translation_test_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/codiesp_WITH_TRANS_fold_{nfold}_test.csv", 

                    #'concat_train_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/CONCAT_codiesp_WITH_TRANS_fold_{nfold}_train.csv",
                    #'concat_validation_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/CONCAT_codiesp_WITH_TRANS_fold_{nfold}_dev.csv",
                    #'concat_test_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/CONCAT_codiesp_WITH_TRANS_fold_{nfold}_test.csv", 

                    'all_labels_path': f"/pvc/tasks/codie_ccs_based_data/{filter_set_name}_labels.pcl",
                    'eval_dataset': eval_dataset,
                    'translator_data_selector': translator_data_selector,
                    #'zero_shot_ccs_path': '/pvc/tasks/codie_ccs_based_data/top10_mmc_codie_achepa.csv', 
                    'zero_shot_diagnoses_path': None,
                    'long_tail_ccs_path': '/pvc/tasks/codie_ccs_based_data/common_long_tail_ccs.csv',
                    'few_shot_ccs_path': '/pvc/tasks/codie_ccs_based_data/{}_few_shot_labels_V2.csv',
                    'zero_shot_ccs_path': '/pvc/tasks/codie_ccs_based_data/{}_zero_shot_labels.csv',

                    }

        return data_paths


def best_adapter_model_paths(): 

    best_adapters = {'achepa_original_SLA': '/pvc/raytune_ccs_codie/tune_adapter_achepa_original_SLA/_inner_4cc57928_35_first_acc_steps=2,first_attention_dropout=0.1,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.0052487,fi_2021-10-20_14-09-26/training_output_el_0.005248721818032698_0/checkpoint-198', 
                    'codie_original_SLA': '/pvc/raytune_ccs_codie/tune_adapter_codie_original_SLA/_inner_b34ab760_27_first_acc_steps=2,first_attention_dropout=0.3,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.0076105,fi_2021-10-19_14-34-52/training_output_es_0.007610478516231566_0/checkpoint-205',
                    'mimic_original_SLA':'/pvc/raytune_ccs_codie/tune_adapter_mimic_original_SLA/_inner_2c36a0d2_50_first_acc_steps=4,first_attention_dropout=0.1,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.00042751,f_2021-10-20_00-03-23/training_output_en_0.0004275118309968961_0/checkpoint-13914/',
                    'mimic_achepa_MLA': '/pvc/raytune_ccs_codie/tune_adapter_english_greek_MLA/_inner_b566d67e_34_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-10-20_17-58-50/training_output_el_0_0.0005403420575244382/checkpoint-3781', 
                    'mimic_achepa_codie_MLA': '/pvc/raytune_ccs_codie/tune_adapter_english_greek_spanish_MLA/_inner_211d3f66_45_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-10-21_08-16-29/training_output_es_0_0.0005205965952255623/checkpoint-533', 
                    'mimic_codie_MLA': '/pvc/raytune_ccs_codie/tune_adapter_english_spanish_MLA/_inner_bb08881e_30_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-10-20_10-59-24/training_output_es_0_0.002273598953256959/checkpoint-160',
                    'mimic_codie_MLA_full_ft': '/pvc/raytune_ccs_codie/tune_adapter_english_spanish_diagnosis_MLA_full_ft/_inner_7a29a87c_2_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoch_2021-11-05_16-00-20/training_output_es_0_0.004767478647497085/checkpoint-20',
                    'achepa_codie_MLA': '/pvc/raytune_ccs_codie/tune_adapter_greek_spanish_MLA/_inner_997dd5f8_15_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-10-21_09-17-08/training_output_es_0_0.0004294121677121819/checkpoint-6391', 
                    'mimic_codie_achepa_MLA': '/pvc/raytune_ccs_codie/tune_adapter_english_spanish_greek_diagnosis_MLA/_inner_2eaac7c4_37_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-11-14_17-36-21/training_output_el_0_0.001010540953761875/checkpoint-1127',
                    'codie_achepa_MLA': '/pvc/raytune_ccs_codie/tune_adapter_spanish_greek_diagnosis_MLA/_inner_2156c9c8_31_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-11-26_12-27-13/training_output_el_0_0.0033779445507932503/checkpoint-49',
                    'codie_mimic_achepa_MLA': '/pvc/raytune_ccs_codie/tune_adapter_spanish_english_greek_diagnosis_MLA/_inner_4e2aaeb8_6_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoch_2021-11-26_10-19-55/training_output_el_0_0.0018813505761500957/checkpoint-2009',
                    'achepa_mimic_codie_MLA': '/pvc/raytune_ccs_codie/tune_adapter_greek_english_spanish_diagnosis_MLA/_inner_69ca4518_1_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoch_2021-11-26_09-29-54/training_output_es_0_1e-05/checkpoint-820'
                    }

    return best_adapters


def get_best_adapter_model_paths(mname): 
    if mname:
        adapter_paths = best_adapter_model_paths()
        return adapter_paths[mname]
    else: 
        return best_adapter_model_paths()


def best_baselines_model_paths(): 

    best_baselines = {'achepa_original_greek_bert':     {'base_model_path':"nlpaueb/bert-base-greek-uncased-v1",
                                                        'best_model_path': '/pvc/raytune_ccs_codie/achepa_original_greek_bert/_inner_b1d7c612_33_acc_grads=2,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=7.3601e-05,num_training_steps=10,seed=42,w_2021-10-21_11-12-47/checkpoints/epoch=62-step=6299.ckpt'}, 
                    'achepa_original_xlmr':             {'base_model_path': "xlm-roberta-base",
                                                        'best_model_path': '/pvc/raytune_ccs_codie/achepa_original_xlmr/_inner_5c285a48_31_acc_grads=1,attention_dropout=0.1,batch_size=8,hidden_dropout=0.1,lr=2.5557e-05,num_training_steps=10,seed=42,w_2021-10-21_15-48-41/checkpoints/epoch=24-step=4974.ckpt'},
                    'codie_original_spanish_bert':      {'base_model_path': 'dccuchile/bert-base-spanish-wwm-cased',
                                                        'best_model_path': '/pvc/raytune_ccs_codie/codie_original_spanish_bert/_inner_4af36f40_36_acc_grads=16,attention_dropout=0.8,batch_size=8,hidden_dropout=0.8,lr=2.2708e-05,num_training_steps=10,seed=42,_2021-10-19_09-46-00/checkpoints/epoch=1-step=11.ckpt'},
                    'codie_original_spanish_bert_uncased': {'base_model_path': 'dccuchile/bert-base-spanish-wwm-uncased',
                                                        'best_model_path':'/pvc/raytune_ccs_codie/codie_original_spanish_bert_uncased/_inner_66947bf2_20_acc_grads=1,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=5.7102e-05,num_training_steps=10,seed=42,w_2021-12-03_15-05-21/checkpoints/epoch=50-step=4181.ckpt'},
                    'codie_original_xlmr':              {'base_model_path': "xlm-roberta-base",
                                                        'best_model_path': '/pvc/raytune_ccs_codie/codie_original_xlmr/_inner_c4c1ef8a_38_acc_grads=1,attention_dropout=0.1,batch_size=8,hidden_dropout=0.5,lr=7.3718e-05,num_training_steps=10,seed=42,w_2021-10-21_14-26-04/checkpoints/epoch=5-step=497.ckpt'}, 
                    'codie_original_spanish_biobert':   {'base_model_path': 'fvillena/bio-bert-base-spanish-wwm-uncased',
                                                        'best_model_path': '/pvc/raytune_ccs_codie/codie_original_spanish_biobert/_inner_aad2664a_37_acc_grads=16,attention_dropout=0.8,batch_size=8,hidden_dropout=0.8,lr=5.0452e-06,num_training_steps=10,seed=42,_2021-10-18_13-41-00/checkpoints/epoch=2-step=17.ckpt'}, 
                    'codie_spanish_clinical_bert':      {'base_model_path': 'BSC-TeMU/roberta-base-biomedical-clinical-es',
                                                        'best_model_path':'/pvc/raytune_ccs_codie/clinical_spanish_V3_None_spanish_biobert_uncased/_inner_4eb5e400_43_acc_grads=2,attention_dropout=0.3,batch_size=8,hidden_dropout=0.1,lr=7.4687e-05,seed=42,warmup_steps=100_2022-01-05_16-12-55/checkpoints/epoch=93-step=3853.ckpt'},
                    'mimic_original_pubmedBert':        {'base_model_path': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                                        'best_model_path': '/pvc/raytune_ccs_codie/mimic_original_pubmedBert/_inner_f4d2b538_36_acc_grads=2,attention_dropout=0.3,batch_size=8,hidden_dropout=0.1,lr=4.6682e-05,num_training_steps=10,seed=42,w_2021-10-19_02-56-20/checkpoints/epoch=11-step=18575.ckpt'},
                    'achepa_opus_pubmed':               {'base_model_path': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                                        'best_model_path': '/pvc/raytune_ccs_codie/translation_models/achepa_Opus_el_en_pubmedBert/_inner_db81fb5e_20_acc_grads=1,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=5.7102e-05,num_training_steps=10,seed=42,w_2021-11-03_10-36-43/checkpoints/epoch=25-step=5173.ckpt'},
                    'codie_off_pubmed':                 {'base_model_path': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                                        'best_model_path': '/pvc/raytune_ccs_codie/translation_models/spanish_V3_official_translation_pubmedBert/_inner_bc3d04d0_21_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.3,lr=5.0918e-05,seed=42,warmup_steps=0_2021-12-08_19-05-29/checkpoints/epoch=78-step=6477.ckpt'},
                                                        #'best_model_path': '/pvc/raytune_ccs_codie/translation_models/codie_official_translation_pubmedBert/_inner_592a24c0_43_acc_grads=2,attention_dropout=0.5,batch_size=8,hidden_dropout=0.3,lr=7.6748e-05,num_training_steps=10,seed=42,w_2021-11-02_17-07-26/checkpoints/epoch=78-step=3238.ckpt'}, 
                    'mimic_achepa_opus_pubmed':         {'base_model_path': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                                        'best_model_path': '/pvc/raytune_ccs_codie/translation_models/english_greek_V2_Opus_el_en_pubmedBert/_inner_9671bc12_39_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.1,lr=5.5209e-05,num_training_steps=10,seed=42,w_2021-12-07_18-51-06/checkpoints/epoch=10-step=2188.ckpt'},
                                                        #'best_model_path': '/pvc/raytune_ccs_codie/translation_models/english_greek_Opus_el_en_pubmedBert/_inner_9e728b18_32_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.1,lr=7.8774e-05,num_training_steps=10,seed=42,w_2021-11-04_10-47-59/checkpoints/epoch=20-step=50514.ckpt'},
                    'mimic_achepa_codie_off_pubmed':    {'base_model_path': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                                         'best_model_path': '/pvc/raytune_ccs_codie/translation_models/english_greek_spanish_V2_official_translation_pubmedBert/_inner_5c5b2794_37_acc_grads=2,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=7.3709e-05,num_training_steps=10,seed=42,w_2021-12-07_23-29-40/checkpoints/epoch=30-step=1270.ckpt'},       
                                                        #'best_model_path': '/pvc/raytune_ccs_codie/translation_models/english_greek_spanish_official_translation_pubmedBert/_inner_59e8ec34_45_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.1,lr=6.909e-05,num_training_steps=10,seed=42,wa_2021-11-04_12-40-35/checkpoints/epoch=39-step=52072.ckpt'},
                    
                    'mimic_codie_off_pubmed':           {'base_model_path': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                                         'best_model_path': '/pvc/raytune_ccs_codie/translation_models/english_spanish_V2_official_translation_pubmedBert/_inner_4e621ce6_34_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.1,lr=7.6222e-05,num_training_steps=10,seed=42,w_2021-12-08_11-31-49/checkpoints/epoch=22-step=1885.ckpt'},
                                                        #'best_model_path': '/pvc/raytune_ccs_codie/translation_models/english_spanish_official_translation_pubmedBert/_inner_06e72e24_31_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.3,lr=7.5526e-05,num_training_steps=10,seed=42,w_2021-11-03_17-32-53/checkpoints/epoch=54-step=52717.ckpt'},
                    
                    'achepa_codie_off_pubmed':          {'base_model_path': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                                        #'best_model_path': '/pvc/raytune_ccs_codie/translation_models/greek_spanish_official_translation_pubmedBert/_inner_02c9c514_40_acc_grads=2,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=2.1012e-05,num_training_steps=10,seed=42,w_2021-11-04_10-07-42/checkpoints/epoch=30-step=5378.ckpt',
                                                        'best_model_path': '/pvc/raytune_ccs_codie/translation_models/greek_spanish_V2_official_translation_pubmedBert/_inner_d5d0588c_31_acc_grads=1,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=4.1921e-05,num_training_steps=10,seed=42,w_2021-12-07_17-05-43/checkpoints/epoch=62-step=5165.ckpt'
                                                        },
                    
                    'achepa_mimic_pubmed': {'base_model_path': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                            'best_model_path':'/pvc/raytune_ccs_codie/greek_english_None_pubmedBert/_inner_3725a1a0_11_acc_grads=2,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=4.074e-05,num_training_steps=10,seed=42,wa_2021-11-18_12-45-54/checkpoints/epoch=41-step=29941.ckpt'},
                    
                    'codie_mimic_pubmed': {'base_model_path': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                            'best_model_path':'/pvc/raytune_ccs_codie/spanish_english_V2_None_pubmedBert/_inner_65f99556_39_acc_grads=4,attention_dropout=0.3,batch_size=8,hidden_dropout=0.3,lr=5.6108e-05,seed=42,warmup_steps=100_2021-12-11_03-50-15/checkpoints/epoch=8-step=6965.ckpt'},

                    'mimic_codie_achepa_pubmed': {'base_model_path': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
                                                  #"best_model_path": "/pvc/raytune_ccs_codie/translation_models/english_spanish_greek_Opus_el_en_pubmedBert/_inner_e7b2080a_20_acc_grads=1,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=5.7102e-05,num_training_steps=10,seed=42,w_2021-11-25_16-37-46/checkpoints/epoch=59-step=53712.ckpt"
                                                #"best_model_path": "/pvc/raytune_ccs_codie/translation_models/english_spanish_greek_V2_Opus_el_en_pubmedBert/_inner_c491d5b8_32_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.3,lr=7.8774e-05,num_training_steps=10,seed=42,w_2021-12-08_00-57-27/checkpoints/epoch=15-step=3183.ckpt"
                                                "best_model_path": "/pvc/raytune_ccs_codie/translation_models/english_spanish_greek_V21_Opus_el_en_pubmedBert/_inner_32b9a500_4_acc_grads=1,attention_dropout=0.5,batch_size=8,hidden_dropout=0.3,lr=5.2831e-05,num_training_steps=10,seed=42,wa_2021-12-08_11-59-38/checkpoints/epoch=21-step=4377.ckpt"
                                                }, 

                    "achepa_mimic_codie_off_pubmed": {'base_model_path': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
                                                    "best_model_path": "/pvc/raytune_ccs_codie/translation_models/greek_english_spanish_official_translation_pubmedBert/_inner_06d9ba72_38_acc_grads=2,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=2.0692e-05,num_training_steps=10,seed=42,w_2021-11-25_18-11-31/checkpoints/epoch=63-step=30843.ckpt"
                                                    },
                    "codie_achepa_opus_pubmed": {'base_model_path': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
                                                'best_model_path': '/pvc/raytune_ccs_codie/translation_models/spanish_greek_V2_Opus_el_en_pubmedBert/_inner_a9c65e26_20_acc_grads=1,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=5.7102e-05,num_training_steps=10,seed=42,w_2021-12-08_09-46-18/checkpoints/epoch=20-step=4178.ckpt'},
                                                #"best_model_path": "/pvc/raytune_ccs_codie/translation_models/spanish_greek_Opus_el_en_pubmedBert/_inner_22384462_48_acc_grads=2,attention_dropout=0.3,batch_size=8,hidden_dropout=0.1,lr=7.9123e-05,num_training_steps=10,seed=42,w_2021-11-26_14-07-52/checkpoints/epoch=83-step=3738.ckpt"},
                    
                    #"codie_achepa_opus_pubmed": {'base_model_path': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
                    #                            "best_model_path": "None"},
                    "codie_mimic_achepa_opus_pubmed":{'base_model_path': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
                                                    "best_model_path":"/pvc/raytune_ccs_codie/translation_models/spanish_english_greek_V2_Opus_el_en_pubmedBert/_inner_fa3d8cb6_21_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.3,lr=5.0918e-05,seed=42,warmup_steps=0_2021-12-11_13-05-51/checkpoints/epoch=17-step=3581.ckpt"},                      
                    
                    "xlmr_mimic_achepa":{"base_model_path": "xlm-roberta-base",
                                        'best_model_path':"/pvc/raytune_ccs_codie/english_greek_V2_None_xlmr/_inner_b8de7b1e_9_acc_grads=4,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=6.3152e-05,num_training_steps=10,seed=42,wa_2021-12-08_15-46-20/checkpoints/epoch=32-step=1649.ckpt"
                                        #"best_model_path": '/pvc/raytune_ccs_codie/english_greek_None_xlmr/_inner_30191ee6_47_acc_grads=4,attention_dropout=0.1,batch_size=8,hidden_dropout=0.1,lr=4.384e-05,num_training_steps=10,seed=42,wa_2021-12-02_15-27-16/checkpoints/epoch=44-step=15729.ckpt'
                                        }, 
                                        
                    "xlmr_achepa_codie":{"base_model_path": "xlm-roberta-base",
                                        #"best_model_path": "/pvc/raytune_ccs_codie/greek_spanish_None_xlmr/_inner_1abf2876_48_acc_grads=2,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=6.8469e-05,num_training_steps=10,seed=42,w_2021-12-04_17-54-28/checkpoints/epoch=29-step=5179.ckpt"
                                        "best_model_path": "/pvc/raytune_ccs_codie/greek_spanish_V2_None_xlmr/_inner_08c16e90_36_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.1,lr=7.4554e-05,seed=42,warmup_steps=500_2021-12-10_08-43-05/checkpoints/epoch=35-step=2951.ckpt"
                                        },

                    "xlmr_codie_achepa":{"base_model_path": "xlm-roberta-base",
                                        #"best_model_path": "/pvc/raytune_ccs_codie/spanish2_greek_None_xlmr/_inner_24fc92a6_43_acc_grads=2,attention_dropout=0.8,batch_size=8,hidden_dropout=0.5,lr=3.9203e-05,num_training_steps=10,seed=42,w_2021-12-05_14-12-32/checkpoints/epoch=10-step=997.ckpt"
                                        "best_model_path": "/pvc/raytune_ccs_codie/spanish_greek_V2_None_xlmr/_inner_fa289040_38_acc_grads=4,attention_dropout=0.1,batch_size=8,hidden_dropout=0.1,lr=4.6144e-05,seed=42,warmup_steps=500_2021-12-10_00-07-04/checkpoints/epoch=50-step=2549.ckpt"
                                        },  
                    "xlmr_mimic":{'base_model_path': "xlm-roberta-base",
                                'best_model_path': '/pvc/raytune_ccs_codie/mimic_original_xlmr/_inner_7bb9f1aa_6_acc_grads=8,attention_dropout=0.5,batch_size=8,hidden_dropout=0.3,lr=6.4523e-05,num_training_steps=10,seed=42,wa_2021-11-26_15-47-00/checkpoints/epoch=39-step=15479.ckpt'},
                    
                    "xlmr_mimic_codie":{"base_model_path": "xlm-roberta-base",
                                        "best_model_path": "/pvc/raytune_ccs_codie/english_spanish_V2_None_xlmr/_inner_bccb5784_21_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.3,lr=5.0918e-05,num_training_steps=10,seed=42,w_2021-12-08_17-25-22/checkpoints/epoch=18-step=1557.ckpt",
                                        #"best_model_path": "/pvc/raytune_ccs_codie/english_spanish_None_xlmr/_inner_2a9d035a_11_acc_grads=2,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=4.074e-05,num_training_steps=10,seed=42,wa_2021-12-02_12-35-00/checkpoints/epoch=61-step=16381.ckpt"
                                        },  
                    "xlmr_mimic_achepa_codie":{"base_model_path": "xlm-roberta-base",
                                        "best_model_path": "/pvc/raytune_ccs_codie/english_greek_spanish_V2_None_xlmr/_inner_0b923c5a_38_acc_grads=2,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=7.3697e-05,seed=42,warmup_steps=100_2021-12-10_12-39-05/checkpoints/epoch=31-step=1311.ckpt"
                                        },  
                    "xlmr_mimic_codie_achepa":{"base_model_path": "xlm-roberta-base",
                                        #"best_model_path": "/pvc/raytune_ccs_codie/english_spanish_greek_None_xlmr/_inner_073f4f40_15_acc_grads=16,attention_dropout=0.5,batch_size=8,hidden_dropout=0.3,lr=6.517e-05,num_training_steps=10,seed=42,w_2021-12-02_21-23-18/checkpoints/epoch=64-step=16420.ckpt"
                                        "best_model_path": "/pvc/raytune_ccs_codie/english_spanish_greek_V2_None_xlmr/_inner_7e7b8fb4_33_acc_grads=4,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=4.3004e-05,seed=42,warmup_steps=100_2021-12-10_13-54-50/checkpoints/epoch=40-step=2049.ckpt"
                                        },  
                    "xlmr_achepa_mimic":{"base_model_path": "xlm-roberta-base",
                                        "best_model_path": "/pvc/raytune_ccs_codie/greek_english_None_xlmr/_inner_87a70776_49_acc_grads=4,attention_dropout=0.3,batch_size=8,hidden_dropout=0.1,lr=3.8818e-05,num_training_steps=10,seed=42,w_2021-11-30_07-39-02/checkpoints/epoch=50-step=25098.ckpt"
                                        }, 
                    "xlmr_codie_mimic":{"base_model_path": "xlm-roberta-base",
                                        "best_model_path":  '/pvc/raytune_ccs_codie/spanish_english_None_xlmr/_inner_50a35616_1_acc_grads=4,attention_dropout=0.1,batch_size=8,hidden_dropout=0.1,lr=1e-05,num_training_steps=10,seed=42,warmup__2021-12-04_20-11-07/checkpoints/epoch=24-step=15203.ckpt'
                                        }, 
                     

                    "xlmr_achepa_mimic_codie":{"base_model_path": "xlm-roberta-base",
                                        "best_model_path": "/pvc/raytune_ccs_codie/greek_english_spanish_None_xlmr/_inner_18632f98_39_acc_grads=1,attention_dropout=0.1,batch_size=8,hidden_dropout=0.1,lr=8.4941e-06,num_training_steps=10,seed=42,w_2021-12-03_11-50-28/checkpoints/epoch=70-step=26738.ckpt"
                                        }, 
                    "xlmr_codie_mimic_achepa":{"base_model_path": "xlm-roberta-base",
                                            #"best_model_path": "/pvc/raytune_ccs_codie/spanish_english_greek_None_xlmr/_inner_07c3f41a_1_acc_grads=4,attention_dropout=0.1,batch_size=8,hidden_dropout=0.1,lr=1e-05,num_training_steps=10,seed=42,warmup__2021-12-05_14-31-27/checkpoints/epoch=29-step=15453.ckpt"
                                            "best_model_path": '/pvc/raytune_ccs_codie/spanish_english_greek_V2_None_xlmr/_inner_e0ce45d0_40_acc_grads=2,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=4.4892e-05,seed=42,warmup_steps=10_2021-12-10_17-32-08/checkpoints/epoch=28-step=2899.ckpt'
                                            }, 
                    }

    return best_baselines


def get_best_baselines_model_paths(mname): 
    if mname:
        model_paths = best_baselines_model_paths()
        return model_paths[mname]
    else: 
        return best_baselines_model_paths()



#load data for adapters 
def evaluate_adapter(language, model_path, model_name, labels, dataset, dataset_name):

    language_codes = {'english':'en',
                      'spanish': 'es', 
                      'greek': 'el'
                    }

    task_adapter_name = 'codiesp_diagnosis_v4'
    is_first = False
    languages = ['english', 'greek', 'spanish']

    dummy_config = {"first_lr": 1e-4, #hp.loguniform("first_lr", 2e-5, 1e-2),
                "second_lr": 1e-5, #hp.loguniform("second_lr", 2e-5, 1e-2),
                'first_batch_size': 8,
                'second_batch_size': 8,
                'per_device_eval_batch_size': 8,
                "first_acc_steps": 2,
                "second_acc_steps": 2,
                "first_warmup_steps":0,
                "second_warmup_steps":0,
                'first_weight_decay': 0,
                'second_weight_decay': 0,
                'first_num_epochs':  100,
                'seed': 42,
                'second_num_epochs': 100,
                'first_attention_dropout': 0.1,
                'first_hidden_dropout': 0.1,
                'second_attention_dropout': 0.1,
                'second_hidden_dropout': 0.1,
                }

   
    if dataset_name == 'codie' and language == 'ft':
        from transformers import XLMRobertaModel, PretrainedConfig
        config = PretrainedConfig.from_json_file(f"{model_path}/config.json")
        model_config = XLMRobertaModel(config)
        model = model_config.from_pretrained(model_path)
        task_adapter_path = model_path + '/codiesp_diagnosis_v4'
        #model.load_head(task_adapter_path)
        model.load_adapter(task_adapter_path)
        task_adapter_path = model_path + '/en'
        model.load_adapter(task_adapter_path, AdapterType.text_lang, )
        task_adapter_path = model_path + '/es'
        model.load_adapter(task_adapter_path, AdapterType.text_lang,)

    else: 
        codieSP_adapter = AdapterSetup(task_adapter_path=model_path,
                                        num_labels=len(labels),
                                        languages=languages, 
                                        task_adapter_name=task_adapter_name, 
                                        is_first=is_first,
                                        model_name=model_name,
                                        config=dummy_config)
        model = codieSP_adapter.final_adapter_model

    ########################### TRAINING FIRST LANGUAGE ########################### 
    adapter_trainer = AdapterTrainer(task_adapter_name, 
                                     model=model)
    
    eval_lm_model_code = language_codes[language]


    metrics = adapter_trainer.evaluate_adapter(eval_dataset=dataset, 
                                                lm_model_code=eval_lm_model_code,
                                                num_labels=len(labels),
                                                is_trained=True,)


    return metrics


def save_metric(metric, metric_path):
        with open(metric_path, 'wb') as f: 
            pickle.dump(metric, f)


def load_labels(data_paths): 
    with open(data_paths['all_labels_path'], 'rb') as f: 
                labels = pickle.load(f)
    return labels


def get_label_mappings(data_paths): 

    labels = load_labels(data_paths)
    label_to_pos, pos_to_label = utils.label_to_pos_map(labels)

    return label_to_pos, pos_to_label


def get_cols_for_spec_eval(eval_type, metrics, data_paths, label_to_pos, dataset_name, nsamples=None): 
    
    valid_cols = metrics['eval_cols']  
    if eval_type == 'average': 
        cols = list(map(lambda x: x.item(), valid_cols))

    elif eval_type == 'zero_shot_ccs':
        zero_shot_df = pd.read_csv(data_paths['zero_shot_ccs_path'].format(dataset_name))
        cols = [label_to_pos[c] for c in zero_shot_df['ccs'].to_list()]
        #assert set(valid_cols).intersection(set(cols)) == set(cols)

    elif eval_type == 'few_shot_ccs': 
            long_tail_df = pd.read_csv(data_paths['few_shot_ccs_path'].format(dataset_name))
            long_tail_df = long_tail_df.rename(columns={'labels': 'ccs'})
            long_tail_df = long_tail_df[long_tail_df["count"].between(nsamples['min'], nsamples['max'])] 
            cols = [label_to_pos[c] for c in long_tail_df['ccs'].to_list()]

    return cols


def combine_name_n_metrics(auc_scores, pos_to_label, samples_per_label, pr_aucs, cols): 
    result = dict()
    col_names = [pos_to_label[c] for c in cols]
    for c_name, auc_score, pr_auc, sample in zip(col_names, auc_scores, pr_aucs, samples_per_label):
        result[c_name] = dict() 
        result[c_name]['auc'] = auc_score
        result[c_name]['#samples'] = sample.item()
        result[c_name]['pr_auc'] = pr_auc
    return result


def compute_metrics(metrics, data_paths, eval_type, dataset_name, nsamples):

        y_true = metrics['eval_y_true']
        y_pred = metrics['eval_y_pred']
        samples_per_label = metrics['eval_samples_per_label']
        y_true = y_true.type_as(y_pred).view(len(y_true), -1)
        y_pred = y_pred.view(len(y_pred), -1)
        
        label_to_pos, pos_to_label = get_label_mappings(data_paths=data_paths)

        cols = get_cols_for_spec_eval(eval_type, 
                                    metrics=metrics, 
                                    data_paths=data_paths, 
                                    label_to_pos=label_to_pos,
                                    dataset_name=dataset_name, 
                                    nsamples=nsamples
                                    )

        auc_score = auroc(y_true[:,cols].cpu(),
                          y_pred[:,cols].cpu(),
                          average= None)
        
        #0.9151873767258383

        micro_auc_score = auroc(y_true[:,cols].cpu(),
                                y_pred[:,cols].cpu(),
                                average='micro')

        avg_auc_score = np.mean(auc_score)
        
        pr_aucs = list()
        precision_recall_dict = dict()
        
        for col in cols:
            lr_precision, lr_recall, _ = precision_recall_curve(y_true[:, col].cpu(), y_pred[:, col].cpu())
            if col not in precision_recall_dict: 
                precision_recall_dict[col] = dict() 
            precision_recall_dict[col]['recall'] = lr_recall
            precision_recall_dict[col]['precision'] = lr_precision
            pr_auc = auc(lr_recall, lr_precision)
            pr_aucs.append(pr_auc)

        result = combine_name_n_metrics(auc_score, pos_to_label, samples_per_label[cols], pr_aucs, cols, )

        avg_pr_auc = np.mean(pr_aucs)
        
        print("AUC score:", avg_auc_score, "PR AUC score", avg_pr_auc, 'micro_auc', micro_auc_score)
        
        result['avg_auc'] = avg_auc_score
        result['avg_micro_auc'] = micro_auc_score
        result['avg_pr_auc'] = avg_pr_auc

        return result

#### get_data for bert evaluation add parameter do_eval=TRUE