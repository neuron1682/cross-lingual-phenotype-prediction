import argparse
import json
import os
import pickle
import pytorch_lightning as pl
import ray
import torch.utils.data
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import dataset
from transformers import BertTokenizerFast
from spanish_bert import *
import sys
sys.path.append('/pvc/')
#from src.utils.utils_v2 import *
from src.utils import utils
from src.baselines.trainer_extended import Trainer as Trainer_Extended

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from hyperopt import hp
import math

def tune_spanish_bert(config, 
                    model_name,
                    task, 
                    data_paths,
                    language,
                    pretrained_model_path,
                    ):

    utils.set_seeds(seed=config['seed'])
    # batch size is taken from config instead of defaults dict
    train_dataloader, validation_dataloader, _, labels = utils.get_data(model_name,
                                                                        data_paths, 
                                                                        language,
                                                                        data_paths['eval_dataset'],
                                                                        config['batch_size'], 
                                                                        task)

     
    #train_dataloader, validation_dataloader, test_dataloader, num_labels = get_datav2(data_paths,
    # len(tr)  
    #                                                                                train_lng=language)
    num_update_steps_per_epoch = len(train_dataloader) // config['acc_grads']
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    #epochs are set in ray async scheduler max_t
    num_train_epochs = 100
    max_steps = math.ceil(num_train_epochs * num_update_steps_per_epoch)
    
    spanish_bert = SpanishBertBaseline(config,
                                    num_labels=len(labels), 
                                    model_name=model_name,
                                    num_training_steps=max_steps
                                    )

                                    
    trainer = Trainer_Extended(#precision=16,
                        gpus=1,
                        max_steps=1e6,
                        #min_epochs=4, 
                        #max_epochs=config['max_epochs'], 
                        accumulate_grad_batches=config["acc_grads"],
                        fast_dev_run=False,
                        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(),
                                                name="", 
                                                version="."),
                        callbacks=[
                                    TuneReportCallback({
                                                        "train_loss": 'train_loss',
                                                        "val_loss": "val_loss",
                                                        'val_auc': 'val_auc',
                                                        'val_pr_auc': 'val_pr_auc'
                                                        #'ncols_30_25': 'ncols_30_25',
                                                        #'ncols_30_50': 'ncols_30_50',
                                                        #'ncols_30_75': 'ncols_30_75',
                                                        #'ncols_30_95': 'ncols_30_95',
                                                        #'ncols_50_25': 'ncols_50_25',
                                                        #'ncols_50_50': 'ncols_50_50',
                                                        #'ncols_50_75': 'ncols_50_75',
                                                        #'ncols_50_95': 'ncols_50_95',
                                                        #'ncols_70_25': 'ncols_70_25',
                                                        #'ncols_70_50': 'ncols_70_50',
                                                        #'ncols_70_75': 'ncols_70_75',
                                                        #'ncols_70_95': 'ncols_70_95',
                                                        #'ncols_80_25': 'ncols_80_25',
                                                        #'ncols_80_50': 'ncols_80_50',
                                                        #'ncols_80_75': 'ncols_80_75',
                                                        #'ncols_80_95': 'ncols_80_95',
                                                        #'ncols_90_25': 'ncols_90_25',
                                                        #'ncols_90_50': 'ncols_90_50',
                                                        #'ncols_90_75': 'ncols_90_75',
                                                        #'ncols_90_95': 'ncols_90_95',
                                                        #"val_selected_auc": "val_selected_auc",
                                                        #'val_f1': 'val_f1',            
                                                        #"val_recall": 'val_recall',
                                                        #"val_precision": 'val_precision',
                                                        #"val_selected_f1": 'val_selected_f1',
                                                        #"val_selected_precision": 'val_selected_precision',
                                                        #"val_selected_recall": 'val_selected_recall',
                                                        }, 
                                                    on="validation_end"), 
                                    EarlyStopping(monitor='val_auc', patience=5, mode='max'), 
                                    ModelCheckpoint(monitor='val_auc', mode='max')
                                ],
                    resume_from_checkpoint=pretrained_model_path,
                    
                    )

    trainer.fit(spanish_bert, train_dataloader, validation_dataloader)
    #metrics = trainer.test(spanish_bert, test_dataloaders=validation_dataloader)
    #metrics = spanish_bert.test_results
    """
    validation_score = trainer.validate(val_dataloaders=spanish_dev_dataset)
    test_score_val = trainer.validate(val_dataloaders=spanish_test_dataset)
    test_score_test = trainer.test(test_dataloaders=spanish_test_dataset)

    test_results = pd.DataFrame(columns=['validation_score','test_score_val', 'test_score_test'], 
                                data=[[validation_score,test_score_val, test_score_test]])
    test_results['lr'] = config['lr']

    test_results['batch_size'] = config['batch_size']
    test_results['acc_grads'] = config['acc_grads']
    test_results['warmup_steps'] = config['warmup_steps']
    test_results.to_csv(f'tasks/english_spanish_task_data/hpo_result/results_test_set/{tune.get_trial_id()}.csv', index=False)
    """
    return trainer




if __name__  == "__main__":

    cluster = True

    if not cluster:
        print('starting ray cluster locally for debugging')
        ray.init(local_mode=True)
        
    else:
        print('starting ray  with the ray service on the CLUSTER')
        ray.init(address=os.environ["RAY_HEAD_SERVICE_HOST"] + ":6379")
    
    config = {"lr": hp.uniform("lr", 5e-6, 8e-5),
              "batch_size": 8,
              "acc_grads": hp.choice("acc_grads", [1, 2, 4, 8, 16]),
              "warmup_steps": hp.choice("warmup_steps", [0, 10, 100, 250, 500, 750]),
              #"num_training_steps":10, 
              'seed': 42,
              'hidden_dropout': hp.choice('hidden_dropout', [0.1, 0.3, 0.5, 0.8]),
              'attention_dropout': hp.choice('attention_dropout', [0.1, 0.3, 0.5, 0.8]),
            }


    utils.set_seeds(seed=config['seed'])


    defaults = [{"lr": 1e-5,
              "batch_size": 8,
              "acc_grads": 4,
              "warmup_steps": 0,
              #"num_training_steps":10, 
              'hidden_dropout': 0.1,
              'attention_dropout': 0.1
            }]


    search = HyperOptSearch(
                            config,
                            metric="val_auc",
                            mode="max",
                            points_to_evaluate=defaults,
                            n_initial_points=30, 
                            random_state_seed=config['seed']
                        )
   

    scheduler = AsyncHyperBandScheduler(
                                        #metric="val_auc",
                                        #mode="max",
                                        brackets=1,
                                        grace_period=2,
                                        reduction_factor=2,
                                        max_t=100
                                    )

    
    model_names = {"spanish_bert_cased":'dccuchile/bert-base-spanish-wwm-cased',
                "spanish_bert_uncased":'dccuchile/bert-base-spanish-wwm-uncased',
                #"spanish_biobert_uncased": 'BSC-TeMU/roberta-base-biomedical-es',
                "spanish_biobert_uncased":  'BSC-TeMU/roberta-base-biomedical-clinical-es',#'fvillena/bio-bert-base-spanish-wwm-uncased',
                #"english_bert": 'bert-base-uncased', 
                #"mbert": 'bert-base-multilingual-uncased', 
                "xlmr": "xlm-roberta-base",
                "pubmedBert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
                "greek_bert": "nlpaueb/bert-base-greek-uncased-v1"
                }

    
    task = 'diagnosis'  #'zero_shot_diag_ccs', 'diagnosis', zero_shot_diag_achepa
    
    #run settings
    # language is trained on eval_dataset in the "language" language

    language = 'clinical_spanish_V3'
    eval_dataset = 'codie'
    mname = 'spanish_biobert_uncased'
    model_name = model_names[mname]

    # if translator is set, the english tranlation of the dataset above is used elsewise the original note
    translator_data_selector = None #'official_translation' #'Opus_el_en' #'official_translation' #None #'Opus_es_en_concat_notes'
    filter_set_name = 'ccs_codie'
    
    mimic_pretrained_model_path = '/pvc/raytune_ccs_codie/mimic_original_pubmedBert/_inner_3fb5fe66_31_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.3,lr=4.1921e-05,num_training_steps=10,seed=42,w_2021-10-19_01-02-58/checkpoints/epoch=15-step=49519.ckpt'
    achepa_pretrained_model_path = '/pvc/raytune_ccs_codie/translation_models/achepa_Opus_el_en_pubmedBert/_inner_db81fb5e_20_acc_grads=1,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=5.7102e-05,num_training_steps=10,seed=42,w_2021-11-03_10-36-43/checkpoints/epoch=25-step=5173.ckpt'
    mimic_achepa_model_path = '/pvc/raytune_ccs_codie/translation_models/english_greek_V2_Opus_el_en_pubmedBert/_inner_9671bc12_39_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.1,lr=5.5209e-05,num_training_steps=10,seed=42,w_2021-12-07_18-51-06/checkpoints/epoch=10-step=2188.ckpt'
    codie_pretrained_model_path = '/pvc/raytune_ccs_codie/translation_models/codie_official_translation_pubmedBert/_inner_592a24c0_43_acc_grads=2,attention_dropout=0.5,batch_size=8,hidden_dropout=0.3,lr=7.6748e-05,num_training_steps=10,seed=42,w_2021-11-02_17-07-26/checkpoints/epoch=78-step=3238.ckpt'
    #codie_pretrained_model_path = '/pvc/raytune_ccs_codie/translation_models/spanish_V3_official_translation_pubmedBert/_inner_bc3d04d0_21_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.3,lr=5.0918e-05,seed=42,warmup_steps=0_2021-12-08_19-05-29/checkpoints/epoch=78-step=6477.ckpt'
    #mimic_codie_pretrained_model_path = '/pvc/raytune_ccs_codie/translation_models/english_spanish_official_translation_pubmedBert/_inner_ac93afac_21_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.3,lr=5.0918e-05,num_training_steps=10,seed=42,w_2021-11-03_17-22-52/checkpoints/epoch=54-step=52717.ckpt'
    mimic_codie_pretrained_model_path =  '/pvc/raytune_ccs_codie/translation_models/english_spanish_V2_official_translation_pubmedBert/_inner_4e621ce6_34_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.1,lr=7.6222e-05,num_training_steps=10,seed=42,w_2021-12-08_11-31-49/checkpoints/epoch=22-step=1885.ckpt'
    achepa_mimic_pretrained_model_path = '/pvc/raytune_ccs_codie/greek_english_None_pubmedBert/_inner_3725a1a0_11_acc_grads=2,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=4.074e-05,num_training_steps=10,seed=42,wa_2021-11-18_12-45-54/checkpoints/epoch=41-step=29941.ckpt'
    #codie_mimic_pretrained_model_path =  '/pvc/raytune_ccs_codie/spanish_english_None_pubmedBert/_inner_610b6f16_11_acc_grads=2,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=4.074e-05,num_training_steps=10,seed=42,wa_2021-11-19_18-14-26/checkpoints/epoch=83-step=10978.ckpt'
    codie_mimic_pretrained_model_path = '/pvc/raytune_ccs_codie/spanish_english_V2_None_pubmedBert/_inner_65f99556_39_acc_grads=4,attention_dropout=0.3,batch_size=8,hidden_dropout=0.3,lr=5.6108e-05,seed=42,warmup_steps=100_2021-12-11_03-50-15/checkpoints/epoch=8-step=6965.ckpt'
    #######XLMR MODELS################################
    achepa_xlmr_pretrained_path = '/pvc/raytune_ccs_codie/achepa_original_xlmr/_inner_5c285a48_31_acc_grads=1,attention_dropout=0.1,batch_size=8,hidden_dropout=0.1,lr=2.5557e-05,num_training_steps=10,seed=42,w_2021-10-21_15-48-41/checkpoints/epoch=24-step=4974.ckpt'
    codie_xlmr_pretrained_path = '/pvc/raytune_ccs_codie/codie_original_xlmr/_inner_c4c1ef8a_38_acc_grads=1,attention_dropout=0.1,batch_size=8,hidden_dropout=0.5,lr=7.3718e-05,num_training_steps=10,seed=42,w_2021-10-21_14-26-04/checkpoints/epoch=5-step=497.ckpt'
    mimic_xlmr_pretrained_path = '/pvc/raytune_ccs_codie/mimic_original_xlmr/_inner_7bb9f1aa_6_acc_grads=8,attention_dropout=0.5,batch_size=8,hidden_dropout=0.3,lr=6.4523e-05,num_training_steps=10,seed=42,wa_2021-11-26_15-47-00/checkpoints/epoch=39-step=15479.ckpt'
    
    #mimic_achepa_xlmr_pretrained_path = '/pvc/raytune_ccs_codie/english_greek_None_xlmr/_inner_0289b9e0_1_acc_grads=4,attention_dropout=0.1,batch_size=8,hidden_dropout=0.1,lr=1e-05,num_training_steps=10,seed=42,warmup__2021-12-02_14-49-11/checkpoints/epoch=44-step=15729.ckpt'
    mimic_achepa_xlmr_pretrained_path = "/pvc/raytune_ccs_codie/english_greek_V2_None_xlmr/_inner_b8de7b1e_9_acc_grads=4,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=6.3152e-05,num_training_steps=10,seed=42,wa_2021-12-08_15-46-20/checkpoints/epoch=32-step=1649.ckpt"
    #mimic_codie_xlmr_pretrained_path = '/pvc/raytune_ccs_codie/english_spanish_None_xlmr/_inner_58e6ad9a_33_acc_grads=2,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=7.3601e-05,num_training_steps=10,seed=42,w_2021-12-02_12-50-39/checkpoints/epoch=61-step=16381.ckpt'
    mimic_codie_xlmr_pretrained_path = "/pvc/raytune_ccs_codie/english_spanish_V2_None_xlmr/_inner_bccb5784_21_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.3,lr=5.0918e-05,num_training_steps=10,seed=42,w_2021-12-08_17-25-22/checkpoints/epoch=18-step=1557.ckpt"
    achepa_mimic_xlmr_pretrained_model_path = '/pvc/raytune_ccs_codie/greek_english_None_xlmr/_inner_87a70776_49_acc_grads=4,attention_dropout=0.3,batch_size=8,hidden_dropout=0.1,lr=3.8818e-05,num_training_steps=10,seed=42,w_2021-11-30_07-39-02/checkpoints/epoch=50-step=25098.ckpt'
    codie_mimic_xlmr_pretrained_model_path = '/pvc/raytune_ccs_codie/spanish_english_None_xlmr/_inner_50a35616_1_acc_grads=4,attention_dropout=0.1,batch_size=8,hidden_dropout=0.1,lr=1e-05,num_training_steps=10,seed=42,warmup__2021-12-04_20-11-07/checkpoints/epoch=24-step=15203.ckpt'

    mimic_V3_pubmed_path = '/pvc/raytune_ccs_codie/english_V3_None_pubmedBert/_inner_8d8f4fe6_8_acc_grads=4,attention_dropout=0.5,batch_size=8,hidden_dropout=0.3,lr=6.9677e-05,seed=42,warmup_steps=500_2021-12-09_05-11-41/checkpoints/epoch=22-step=17801.ckpt'
    
    pretrained_model_path = None #codie_mimic_pretrained_model_path

    if language == 'spanish':
        if translator_data_selector is not None:
            dataset_name = f"codie_{translator_data_selector}"
        else:
             dataset_name = f"codie_original"
        experiment_name = f"{dataset_name}_{mname}"

    elif language == 'english':
        if translator_data_selector is not None:
            dataset_name = f"mimic_{translator_data_selector}"
        else:
             dataset_name = f"mimic_original"
        experiment_name = f"{dataset_name}_{mname}"
    elif language == 'greek':
        if translator_data_selector is not None:
            dataset_name = f"achepa_{translator_data_selector}"
        else:
             dataset_name = f"achepa_original"
        experiment_name = f"{dataset_name}_{mname}"

    else: 
        dataset_name = f"{language}_{translator_data_selector}"
        experiment_name = f"{dataset_name}_{mname}"
    
    if translator_data_selector:
        local_dir = f"/pvc/raytune_{filter_set_name}/translation_models"
    else:
         local_dir = f"/pvc/raytune_{filter_set_name}/"
    
    if task == 'zero_shot_diag_ccs': 
        experiment_name = f"{experiment_name}_{task}"


    nfold = 3
    

    #data_paths = {'train_data_path': f"/pvc/tasks/english_spanish_task_data/processed_data/fold_{nfold}/{dataset_name}_train.csv",
    #            'validation_data_path': f"/pvc/tasks/english_spanish_task_data/processed_data/fold_{nfold}/{dataset_name}_dev.csv",
    #            'test_data_path': f"/pvc/tasks/english_spanish_task_data/processed_data/fold_{nfold}/{dataset_name}_test.csv", 
    #            'all_labels_path': f"/pvc/tasks/{filter_set_name}_labels.pcl", 
    #            }

    #data_paths = {'train_data_path': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/{dataset_name}_fold_{nfold}_train.csv",
    #            'validation_data_path': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/{dataset_name}_fold_{nfold}_dev.csv",
    #            'test_data_path': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/{dataset_name}_fold_{nfold}_test.csv", 
    #            'all_labels_path': f"/pvc/tasks/{filter_set_name}_labels.pcl", 
    #            }
    

    data_paths = {'train_data_path_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/mimic_fold_{nfold}_train.csv",
        'validation_data_path_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/mimic_fold_{nfold}_dev.csv",
        'test_data_path_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/mimic_fold_{nfold}_test.csv",

        'train_data_path_es': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/codiesp_fold_{nfold}_train.csv",
        'validation_data_path_es': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/codiesp_fold_{nfold}_dev.csv",
        'test_data_path_es': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/codiesp_fold_{nfold}_test.csv", 

        'translation_train_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/codiesp_WITH_TRANS_fold_{nfold}_train.csv",
        'translation_validation_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/codiesp_WITH_TRANS_fold_{nfold}_dev.csv",
        'translation_test_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/codiesp_WITH_TRANS_fold_{nfold}_test.csv", 

        'concat_train_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/CONCAT_codiesp_WITH_TRANS_fold_{nfold}_train.csv",
        'concat_validation_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/CONCAT_codiesp_WITH_TRANS_fold_{nfold}_dev.csv",
        'concat_test_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/CONCAT_codiesp_WITH_TRANS_fold_{nfold}_test.csv", 

        'all_labels_path': f"/pvc/tasks/{filter_set_name}_labels.pcl", 
        'zero_shot_ccs_path': '/pvc/tasks/codie_ccs_based_data/top10_mmc_codie_achepa.csv', 
        'zero_shot_diagnoses_path': None,
        'translator_data_selector': translator_data_selector}
        

    data_paths = {'train_data_path_mimic': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/mimic_codiesp_filtered_CCS_fold_{nfold}_train.csv",
                'validation_data_path_mimic': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/mimic_codiesp_filtered_CCS_fold_{nfold}_dev.csv",
                'test_data_path_mimic': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/mimic_codiesp_filtered_CCS_fold_{nfold}_test.csv",

                'train_data_path_codie': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/codiesp_CCS_fold_{nfold}_train.csv",
                'validation_data_path_codie': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/codiesp_CCS_fold_{nfold}_dev.csv",
                'test_data_path_codie': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/codiesp_CCS_fold_{nfold}_test.csv", 

                'train_data_path_achepa': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/achepa_codiesp_filtered_CCS_fold_{nfold}_train.csv",
                'validation_data_path_achepa': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/achepa_codiesp_filtered_CCS_fold_{nfold}_dev.csv",
                'test_data_path_achepa': f"/pvc/tasks/codie_ccs_based_data/k_folds/fold_{nfold}/achepa_codiesp_filtered_CCS_fold_{nfold}_test.csv",
                #'translation_train_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/codiesp_WITH_TRANS_fold_{nfold}_train.csv",
                #'translation_validation_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/codiesp_WITH_TRANS_fold_{nfold}_dev.csv",
                #'translation_test_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/codiesp_WITH_TRANS_fold_{nfold}_test.csv", 

                #'concat_train_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/CONCAT_codiesp_WITH_TRANS_fold_{nfold}_train.csv",
                #'concat_validation_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/CONCAT_codiesp_WITH_TRANS_fold_{nfold}_dev.csv",
                #'concat_test_data_path_es_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/CONCAT_codiesp_WITH_TRANS_fold_{nfold}_test.csv", 
                'zero_shot_ccs_path': '/pvc/tasks/codie_ccs_based_data/top10_mmc_codie_achepa.csv', 
                'zero_shot_diagnoses_path': None,
                'all_labels_path': f"/pvc/tasks/codie_ccs_based_data/{filter_set_name}_labels.pcl",
                'eval_dataset': eval_dataset,
                'translator_data_selector': translator_data_selector}


    assert data_paths['translator_data_selector'] in ['official_translation', 'Opus_el_en', 
                                                      'Opus_es_to_en', 'GT_es_to_en', 
                                                      'Opus_es_en_concat_notes', 'GT_es_en_concat_notes', 
                                                      None]
    print(50 * '+')
    print('WARNING, Translator is set to: ', data_paths['translator_data_selector'])
    print(50 * '+')


    reporter = CLIReporter(
                        parameter_columns=["lr",
                                        "batch_size",
                                        "acc_grads", 
                                        "warmup_steps"
                                        ],

                        metric_columns=["loss",
                                        "val_loss",
                                        'val_auc',
                                        'val_pr_auc',
                                        #'ncols_30_25',
                                        #'ncols_30_50',
                                        #'ncols_30_75',
                                        #'ncols_30_95',
                                        ##'ncols_50_25',
                                        #'ncols_50_50',
                                        #'ncols_50_75',
                                        #'ncols_50_95',
                                        #'ncols_70_25',
                                        #'ncols_70_50',
                                        #'ncols_70_75',
                                        #'ncols_70_95',
                                        #'ncols_80_25',
                                        #'ncols_80_50',
                                        #'ncols_80_75',
                                        #'ncols_80_95',
                                        #'ncols_90_25',
                                        #'ncols_90_50',
                                        #'ncols_90_75',
                                        #'ncols_90_95',
                                        ]
                                        
                                        #"val_selected_auc",
                                        #'val_f1',            
                                        #"val_recall",
                                        #"val_precision",
                                        #"val_selected_f1",
                                        #"val_selected_precision",
                                        #"val_selected_recall", ]
                            )


    trainer = tune.run(tune.with_parameters(tune_spanish_bert, 
                                            model_name=model_name,
                                            task=task, 
                                            data_paths=data_paths,
                                            language=language,
                                            pretrained_model_path=pretrained_model_path,
                                            ),    
                        local_dir=local_dir,
                        resources_per_trial={'cpu': 8, "gpu":1},
                        metric="val_auc",
                        mode="max",
                        config=config,
                        num_samples=50,
                        scheduler=scheduler,
                        search_alg=search,
                        progress_reporter=reporter,
                        name=experiment_name, )
                        #checkpoint_at_end=True)
    
    
    best_config = trainer.get_best_config()
    with open(f"/pvc/tasks/codie_ccs_based_data/{experiment_name}_best_config_{filter_set_name}_fold_{nfold}.pcl",'wb') as f: 
        pickle.dump(best_config, f)

    trainer.best_result_df.to_csv(f"/pvc/tasks/codie_ccs_based_data/{experiment_name}__best_result_{filter_set_name}_fold_{nfold}.csv", index=False)
    trainer.dataframe().to_csv(f"/pvc/tasks/codie_ccs_based_data/{experiment_name}__best_result_{filter_set_name}_fold_{nfold}.csv", index=False)
    
