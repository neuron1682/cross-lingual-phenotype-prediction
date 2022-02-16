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
                    resources_per_trial,
                    ):

    utils.set_seeds(seed=config['seed'])
    # batch size is taken from config instead of defaults dict
    train_dataloader, validation_dataloader, _, labels = utils.get_data(model_name,
                                                                        data_paths, 
                                                                        language,
                                                                        data_paths['eval_dataset'],
                                                                        config['batch_size'], 
                                                                        task)

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

       
    trainer = Trainer_Extended(
                        gpus=resources_per_trial['gpu'],
                        max_steps=1e6,
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
                                                        }, 
                                                    on="validation_end"), 
                                    EarlyStopping(monitor='val_auc', patience=5, mode='max'), 
                                    ModelCheckpoint(monitor='val_auc', mode='max')
                                ],
                    resume_from_checkpoint=pretrained_model_path,
                    )

    trainer.fit(spanish_bert, train_dataloader, validation_dataloader)
    return trainer




if __name__  == "__main__":

    # start ray on cluster or debug locally
    cluster = False

    if not cluster:
        print('starting ray cluster locally for debugging')
        ray.init(local_mode=True)
        
    else:
        print('starting ray  with the ray service on the CLUSTER')
        ray.init(address=os.environ["RAY_HEAD_SERVICE_HOST"] + ":6379")
    

    model_names = {"spanish_bert_uncased":'dccuchile/bert-base-spanish-wwm-uncased',
                #"spanish_biobert_uncased": 'BSC-TeMU/roberta-base-biomedical-es',
                "spanish_biobert_uncased":  'BSC-TeMU/roberta-base-biomedical-clinical-es',
                "xlmr": "xlm-roberta-base",
                "pubmedBert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
                "greek_bert": "nlpaueb/bert-base-greek-uncased-v1"
                }

    # only diagnosis task is implemented
    task = 'diagnosis'
    
    #run settings
    #the variable is used for experiment naming
    language = 'clinical_spanish_V3'
    
    # dataset to train or evaluate with
    eval_dataset = 'codie' #'mimic'

    # model naming
    mname = 'spanish_biobert_uncased'

    # model base to train with
    model_name = model_names[mname]

    # column name of the text with English Translation
    ''' choose `None` if you want the notes original language '''
    translator_data_selector = None #'official_translation' #'Opus_el_en'
    
    # filename to load labels created by `dataset_creation/pre-process_codie.py`
    filter_set_name = 'ccs_codie'
    
    # resources to execute the hpo
    resources_per_trial = {'cpu': 8, "gpu":1}

    # paths to datasets labels and column (translation or original)
    data_paths = {'train_data_path_mimic': f"/pvc/output_files/mimic_codiesp_filtered_CCS_train.csv",
                'validation_data_path_mimic': f"/pvc/output_files/mimic_codiesp_filtered_CCS_dev.csv",
                'test_data_path_mimic': f"/pvc/output_files/mimic_codiesp_filtered_CCS_test.csv",

                'train_data_path_achepa': f"/pvc/output_files/train.csv",
                'validation_data_path_achepa': f"/pvc/output_files/dev.csv",
                'test_data_path_achepa': f"/pvc/output_files/test.csv",

                'train_data_path_codie': f"/pvc/output_files/codiesp_CCS_train.csv",
                'validation_data_path_codie': f"/pvc/output_files/codiesp_CCS_dev.csv",
                'test_data_path_codie': f"/pvc/output_files/codiesp_CCS_test.csv",

                'all_labels_path': f"/pvc/output_files/{filter_set_name}_labels.pcl",
                'eval_dataset': eval_dataset,
                'translator_data_selector': translator_data_selector,
                }

    assert data_paths['translator_data_selector'] in ['official_translation', 'Opus_el_en', None]
    print(50 * '+')
    print('WARNING, Translator is set to: ', data_paths['translator_data_selector'])
    print(50 * '+')

    # Paths to best models to continue training
    ######## PUBMEDBERT MODELS #######################
    mimic_pretrained_model_path = '/pvc/raytune_ccs_codie/mimic_original_pubmedBert/_inner_3fb5fe66_31_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.3,lr=4.1921e-05,num_training_steps=10,seed=42,w_2021-10-19_01-02-58/checkpoints/epoch=15-step=49519.ckpt'
    achepa_pretrained_model_path = '/pvc/raytune_ccs_codie/translation_models/achepa_Opus_el_en_pubmedBert/_inner_db81fb5e_20_acc_grads=1,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=5.7102e-05,num_training_steps=10,seed=42,w_2021-11-03_10-36-43/checkpoints/epoch=25-step=5173.ckpt'
    mimic_achepa_model_path = '/pvc/raytune_ccs_codie/translation_models/english_greek_V2_Opus_el_en_pubmedBert/_inner_9671bc12_39_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.1,lr=5.5209e-05,num_training_steps=10,seed=42,w_2021-12-07_18-51-06/checkpoints/epoch=10-step=2188.ckpt'
    codie_pretrained_model_path = '/pvc/raytune_ccs_codie/translation_models/codie_official_translation_pubmedBert/_inner_592a24c0_43_acc_grads=2,attention_dropout=0.5,batch_size=8,hidden_dropout=0.3,lr=7.6748e-05,num_training_steps=10,seed=42,w_2021-11-02_17-07-26/checkpoints/epoch=78-step=3238.ckpt'
    mimic_codie_pretrained_model_path =  '/pvc/raytune_ccs_codie/translation_models/english_spanish_V2_official_translation_pubmedBert/_inner_4e621ce6_34_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.1,lr=7.6222e-05,num_training_steps=10,seed=42,w_2021-12-08_11-31-49/checkpoints/epoch=22-step=1885.ckpt'
    achepa_mimic_pretrained_model_path = '/pvc/raytune_ccs_codie/greek_english_None_pubmedBert/_inner_3725a1a0_11_acc_grads=2,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=4.074e-05,num_training_steps=10,seed=42,wa_2021-11-18_12-45-54/checkpoints/epoch=41-step=29941.ckpt'
    codie_mimic_pretrained_model_path = '/pvc/raytune_ccs_codie/spanish_english_V2_None_pubmedBert/_inner_65f99556_39_acc_grads=4,attention_dropout=0.3,batch_size=8,hidden_dropout=0.3,lr=5.6108e-05,seed=42,warmup_steps=100_2021-12-11_03-50-15/checkpoints/epoch=8-step=6965.ckpt'
    
    #######XLMR MODELS################################
    achepa_xlmr_pretrained_path = '/pvc/raytune_ccs_codie/achepa_original_xlmr/_inner_5c285a48_31_acc_grads=1,attention_dropout=0.1,batch_size=8,hidden_dropout=0.1,lr=2.5557e-05,num_training_steps=10,seed=42,w_2021-10-21_15-48-41/checkpoints/epoch=24-step=4974.ckpt'
    codie_xlmr_pretrained_path = '/pvc/raytune_ccs_codie/codie_original_xlmr/_inner_c4c1ef8a_38_acc_grads=1,attention_dropout=0.1,batch_size=8,hidden_dropout=0.5,lr=7.3718e-05,num_training_steps=10,seed=42,w_2021-10-21_14-26-04/checkpoints/epoch=5-step=497.ckpt'
    mimic_xlmr_pretrained_path = '/pvc/raytune_ccs_codie/mimic_original_xlmr/_inner_7bb9f1aa_6_acc_grads=8,attention_dropout=0.5,batch_size=8,hidden_dropout=0.3,lr=6.4523e-05,num_training_steps=10,seed=42,wa_2021-11-26_15-47-00/checkpoints/epoch=39-step=15479.ckpt'
    mimic_achepa_xlmr_pretrained_path = "/pvc/raytune_ccs_codie/english_greek_V2_None_xlmr/_inner_b8de7b1e_9_acc_grads=4,attention_dropout=0.5,batch_size=8,hidden_dropout=0.1,lr=6.3152e-05,num_training_steps=10,seed=42,wa_2021-12-08_15-46-20/checkpoints/epoch=32-step=1649.ckpt"
    mimic_codie_xlmr_pretrained_path = "/pvc/raytune_ccs_codie/english_spanish_V2_None_xlmr/_inner_bccb5784_21_acc_grads=1,attention_dropout=0.3,batch_size=8,hidden_dropout=0.3,lr=5.0918e-05,num_training_steps=10,seed=42,w_2021-12-08_17-25-22/checkpoints/epoch=18-step=1557.ckpt"
    achepa_mimic_xlmr_pretrained_model_path = '/pvc/raytune_ccs_codie/greek_english_None_xlmr/_inner_87a70776_49_acc_grads=4,attention_dropout=0.3,batch_size=8,hidden_dropout=0.1,lr=3.8818e-05,num_training_steps=10,seed=42,w_2021-11-30_07-39-02/checkpoints/epoch=50-step=25098.ckpt'
    codie_mimic_xlmr_pretrained_model_path = '/pvc/raytune_ccs_codie/spanish_english_None_xlmr/_inner_50a35616_1_acc_grads=4,attention_dropout=0.1,batch_size=8,hidden_dropout=0.1,lr=1e-05,num_training_steps=10,seed=42,warmup__2021-12-04_20-11-07/checkpoints/epoch=24-step=15203.ckpt'
    
    '''
    # if it is the first training keep `None` 
    # else choose path to best model to continue training 
    # from model_path
    '''
    pretrained_model_path = None 


    '''
        Naming of the experiments
    '''
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
    
    

    '''
                settings for hyperparameter tuning
    '''    
    config = {"lr": hp.uniform("lr", 5e-6, 8e-5),
              "batch_size": 8,
              "acc_grads": hp.choice("acc_grads", [1, 2, 4, 8, 16]),
              "warmup_steps": hp.choice("warmup_steps", [0, 10, 100, 250, 500, 750]),
              'seed': 42,
              'hidden_dropout': hp.choice('hidden_dropout', [0.1, 0.3, 0.5, 0.8]),
              'attention_dropout': hp.choice('attention_dropout', [0.1, 0.3, 0.5, 0.8]),
            }


    utils.set_seeds(seed=config['seed'])


    defaults = [{"lr": 1e-5,
              "batch_size": 8,
              "acc_grads": 4,
              "warmup_steps": 0,
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
                                        brackets=1,
                                        grace_period=2,
                                        reduction_factor=2,
                                        max_t=100
                                    )

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
                                        ]
                            )


    trainer = tune.run(tune.with_parameters(tune_spanish_bert, 
                                            model_name=model_name,
                                            task=task, 
                                            data_paths=data_paths,
                                            language=language,
                                            pretrained_model_path=pretrained_model_path,
                                            resources_per_trial=resources_per_trial,
                                            ),    
                        local_dir=local_dir,
                        resources_per_trial=resources_per_trial,
                        metric="val_auc",
                        mode="max",
                        config=config,
                        num_samples=50,
                        scheduler=scheduler,
                        search_alg=search,
                        progress_reporter=reporter,
                        name=experiment_name, )
    
    
    best_config = trainer.get_best_config()
    with open(f"/pvc/tasks/codie_ccs_based_data/{experiment_name}_best_config_{filter_set_name}.pcl",'wb') as f: 
        pickle.dump(best_config, f)

    trainer.best_result_df.to_csv(f"/pvc/tasks/codie_ccs_based_data/{experiment_name}__best_result_{filter_set_name}.csv", index=False)
    trainer.dataframe().to_csv(f"/pvc/tasks/codie_ccs_based_data/{experiment_name}__best_result_{filter_set_name}.csv", index=False)
    
