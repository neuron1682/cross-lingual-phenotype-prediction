from multilingual_adapter import *

from transformers import AutoConfig, AutoModelWithHeads
from transformers import TrainingArguments, Trainer
from datasets import concatenate_datasets
from transformers import AdapterType, AdapterConfig
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import roc_auc_score as auroc

import argparse
import json
import os

import pytorch_lightning as pl
import ray
import torch.utils.data
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from transformers import BertTokenizerFast
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from hyperopt import hp
import sys

from src.utils.codiespDataset import codiespDataset
sys.path.append('/pvc/')
from src.utils import utils
from src.utils.trainer_callback import EarlyStoppingCallback
import pickle

def tune_adapter(config,
                model_name, 
                task, 
                language,
                data_paths,
                is_first,
                dataset_name,
                task_adapter_path
                ):
                              
        language_codes = {'spanish':'es',
                        'english':'en', 
                        'greek':'el'
                        }

        utils.set_seeds(seed=config['seed'])
        ########################### SETUP ADAPTER ########################### 
        #first_train_language = language
        #train_data_path = f'/pvc/codiesp/xl-training-data/v4/{task}_task/rebalanced/{first_train_language}_clinical_text'
        train_dataset, dev_dataset, _, labels = utils.get_datav2(data_paths,
                                                                dataset_name=dataset_name)

        codie_cond = task == 'zero_shot_diag_ccs' and dataset_name == 'codie'
        achepa_cond = task == 'zero_shot_diag_achepa' and dataset_name == 'achepa'
                                                  
        if codie_cond or achepa_cond: 
                # TODO load selected labels
                ccs = utils.load_zero_shot_ccs_codes(data_paths)
                data_class = codiespDataset(data_paths, dataset_name)
                label_to_pos, _ = data_class.label_to_pos(all_codes=labels)
                selected_labels = utils.get_zero_shot_ccs_idx(label_to_pos, ccs)
                train_dataset = train_dataset.map(lambda x: manipulate_zero_shot_diagnosis(x, selected_labels))
                dev_dataset = dev_dataset.map(lambda x: manipulate_zero_shot_diagnosis(x, selected_labels))

        elif task == 'zero_shot_diag_css' or task == 'zero_shot_diag_achepa': 
                logger.warning('zero_shot_diag is only valid for codiesp for ccs and for achepa for achepa diagnoses')
                raise

        task_adapter_name = f'codiesp_diagnosis_v4'

        codieSP_adapter = AdapterSetup(task_adapter_path=task_adapter_path,
                                        num_labels=len(labels),
                                        languages=languages, 
                                        task_adapter_name=task_adapter_name, 
                                        is_first=is_first,
                                        model_name=model_name,
                                        config=config)

        ########################### TRAINING FIRST LANGUAGE ########################### 
        adapter_trainer = AdapterTrainer(task_adapter_name, 
                                         model=codieSP_adapter.final_adapter_model)

        adapter_trainer.train_adapter(is_first=is_first,
                                lm_model_code=language_codes[language], 
                                train_dataset=train_dataset,
                                eval_dataset= dev_dataset,
                                config=config,
                                dataset_name=dataset_name, 
                                )
        
if __name__ == "__main__":
        
        # start ray on cluster or debug locally
        cluster = False

        if not cluster:
                print('starting ray cluster locally for debugging')
                ray.init(local_mode=True)
                
        else:
                print('starting ray  with the ray service on the CLUSTER')
                ray.init(address=os.environ["RAY_HEAD_SERVICE_HOST"] + ":6379")
        
        # Is it the first training of the task adapter
        is_first = True

        # model name SLA(single language)
        mname = 'SLA' 

        # base model where adapters are intergrated
        model_name = 'xlm-roberta-base'

        # column name of the text with English Translation
        ''' None if you want the original language '''
        translator_data_selector = None #'Opus_es_en_concat_notes'
        
        # filename to load labels
        filter_set_name = 'ccs_codie'

        # name of the dataset to train with
        eval_dataset = 'codie'
        
        '''
         if it is not the first run include the other 
         language adapters name in the list  
         e.g. if it is pretrained with mimic and 
         second training is with CodiEsp 
         languages = ['english', 'spanish']
        '''
        languages = ['spanish']
        
        #language of the current dataset to continue training and evaluation
        language = 'spanish'
        
        # just a variable for the naming of the experiments
        mla_order = '_'.join(languages)
        
        # only diagnosis task is implemented
        task = 'diagnosis' 
        
        # is it a test run
        test = True

        # resources to execute the hpo
        resources_per_trial = {'cpu': 8, "gpu":1}

        # paths to datasets labels and column (translation or original)
        data_paths = {'train_data_path_mimic': f"dataset_creation/output_files/mimic_codiesp_filtered_CCS_train.csv",
                'validation_data_path_mimic': f"dataset_creation/output_files/mimic_codiesp_filtered_CCS_dev.csv",
                'test_data_path_mimic': f"dataset_creation/output_files/mimic_codiesp_filtered_CCS_test.csv",
                
                'train_data_path_achepa': f"dataset_creation/output_files/train.csv",
                'validation_data_path_achepa': f"dataset_creation/output_files/dev.csv",
                'test_data_path_achepa': f"dataset_creation/output_files/test.csv",

                'train_data_path_codie': f"dataset_creation/output_files/codiesp_CCS_train.csv",
                'validation_data_path_codie': f"dataset_creation/output_files/codiesp_CCS_dev.csv",
                'test_data_path_codie': f"dataset_creation/output_files/codiesp_CCS_test.csv",

                'all_labels_path': f"dataset_creation/output_files/{filter_set_name}_labels.pcl",
                'eval_dataset': eval_dataset,
                'translator_data_selector': translator_data_selector,
                }

        # Paths to best models to continue training
        task_adapter_mimic_sla_path = '/pvc/raytune_ccs_codie/tune_adapter_mimic_original_SLA/_inner_2c36a0d2_50_first_acc_steps=4,first_attention_dropout=0.1,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.00042751,f_2021-10-20_00-03-23/training_output_en_0.0004275118309968961_0/checkpoint-13914/'
        task_adapter_achepa_sla_path = '/pvc/raytune_ccs_codie/tune_adapter_achepa_original_SLA/_inner_4cc57928_35_first_acc_steps=2,first_attention_dropout=0.1,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.0052487,fi_2021-10-20_14-09-26/training_output_el_0.005248721818032698_0/checkpoint-198'
        task_adapter_mimic_achepa_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_greek_MLA/_inner_b566d67e_34_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-10-20_17-58-50/training_output_el_0_0.0005403420575244382/checkpoint-3781'
        task_adapter_mimic_codie_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_spanish_MLA/_inner_b697bfc6_12_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-10-20_10-52-50/training_output_es_0_0.0011030338137158105/checkpoint-160'
        task_adapter_codie_sla_path = '/pvc/raytune_ccs_codie/tune_adapter_codie_original_SLA/_inner_b34ab760_27_first_acc_steps=2,first_attention_dropout=0.3,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.0076105,fi_2021-10-19_14-34-52/training_output_es_0.007610478516231566_0/checkpoint-205'
        task_adapter_achepa_mimic_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_greek_english_diagnosis_MLA/_inner_3a0ea5a2_41_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-11-16_03-20-21/training_output_en_0_0.0017506470138346506/checkpoint-6176'
        task_adapter_codie_mimic_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_spanish_english_diagnosis_MLA/_inner_7ceb71c6_34_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-11-15_14-43-55/training_output_en_0_0.0008006564657455058/checkpoint-6957'
        
        if first:
                # first training
                task_adapter_path = None
        else:
                # select path of best model to continue training from 
                task_adapter_path = task_adapter_mimic_sla_path



        '''
                Naming of the experiments
        '''
        if language == 'spanish':
                if translator_data_selector is not None:
                        dataset_name = f"codie_{translator_data_selector}_{task}"
                else:
                        if mname == 'SLA':
                                dataset_name = f"codie_original"
                        elif mname == 'MLA': 
                                dataset_name = f"{mla_order}_{task}"
                        experiment_name = f"{dataset_name}_{mname}"


        elif language == 'english':
                if translator_data_selector is not None:
                        dataset_name = f"mimic_{translator_data_selector}"
                else:
                        if mname == 'SLA':
                                dataset_name = f"mimic_original"
                        elif mname == 'MLA': 
                                dataset_name = f"{mla_order}_{task}"
                        experiment_name = f"{dataset_name}_{mname}"

        elif language == 'greek':
                if translator_data_selector is not None:
                        dataset_name = f"achepa_{translator_data_selector}"
                else:
                        if mname == 'SLA':
                                dataset_name = f"achepa_original"
                        elif mname == 'MLA': 
                                dataset_name = f"{mla_order}_{task}"
                        experiment_name = f"{dataset_name}_{mname}"
        else: 
                dataset_name = f"{translator_data_selector}"
                experiment_name = f"{dataset_name}_{mname}_{task}"

        experiment_name = f"{experiment_name}" 

        if test:

                experiment_name = experiment_name + "_TEST"           


        
        '''
                settings for hyperparameter tuning
        '''
        if is_first:
                config = {"first_lr": hp.uniform("first_lr", 1e-5, 1e-2),
                        "second_lr": 0,#hp.uniform("second_lr", 1e-5, 1e-2),
                        'first_batch_size': 8,
                        'second_batch_size': 0,
                        'per_device_eval_batch_size': 8,
                        "first_acc_steps": hp.choice("first_acc_steps", [1, 2, 4, 8, 16, 32]),
                        "second_acc_steps": 0,#hp.choice("second_acc_steps", [1, 2, 4, 8, 16]),
                        "first_warmup_steps": hp.choice("first_warmup_steps", [0, 10, 250, 500, 750]),
                        "second_warmup_steps": 0, #hp.choice("second_warmup_steps", [0, 10, 250, 500, 750]),
                        'first_weight_decay': 0,
                        'second_weight_decay': 0,
                        'first_num_epochs':  100, #hp.choice("first_num_epochs", [10, 30, 50, 80, 100, 150]), 
                        'second_num_epochs': 1,#hp.choice("second_num_epochs", [10, 30, 50, 80, 100, 150]),
                        'seed': 42, 
                        'first_hidden_dropout': hp.choice('first_hidden_dropout', [0.1, 0.3, 0.5, 0.8]),
                        'first_attention_dropout': hp.choice('first_attention_dropout', [0.1, 0.3, 0.5, 0.8]),
                        'second_hidden_dropout': 0,
                        'second_attention_dropout': 0,
                        }
        else:
                config = {"first_lr": 0,#hp.uniform("first_lr", 1e-5, 1e-2),
                        "second_lr": hp.uniform("second_lr", 1e-5, 1e-2),
                        'first_batch_size': 8,
                        'second_batch_size': 8,
                        'per_device_eval_batch_size': 8,
                        "first_acc_steps": 0,#hp.choice("first_acc_steps", [1, 2, 4, 8, 16, 32]),
                        "second_acc_steps": hp.choice("second_acc_steps", [1, 2, 4, 8, 16]),
                        "first_warmup_steps": 0,#hp.choice("first_warmup_steps", [0, 10, 250, 500, 750]),
                        "second_warmup_steps": hp.choice("second_warmup_steps", [0, 10, 250, 500, 750]),
                        'first_weight_decay': 0,
                        'second_weight_decay': 0,
                        'first_num_epochs':  100,#hp.choice("first_num_epochs", [10, 30, 50, 80, 100, 150]), 
                        'second_num_epochs': 100,#hp.choice("second_num_epochs", [10, 30, 50, 80, 100, 150]),
                        'seed': 42,
                        'second_hidden_dropout': hp.choice('second_hidden_dropout', [0.1, 0.3, 0.5, 0.8]),
                        'second_attention_dropout': hp.choice('second_attention_dropout', [0.1, 0.3, 0.5, 0.8]),
                        'first_hidden_dropout': 0,
                        'first_attention_dropout': 0,
                        }

        defaults = [{"first_lr": 1e-4, #hp.loguniform("first_lr", 2e-5, 1e-2),
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
                }]

        utils.set_seeds(seed=config['seed'])


        search = HyperOptSearch(
                            config,
                            metric="eval_val_auc",
                            mode="max",
                            points_to_evaluate=defaults,
                            n_initial_points=30)


        scheduler = AsyncHyperBandScheduler(
                                                #metric="eval_val_auc_avg",
                                                #mode="max",
                                                brackets=1,
                                                grace_period=2,
                                                reduction_factor=4,
                                                max_t=100
                                        )


        reporter = CLIReporter(
                        parameter_columns=["first_lr",
                                        "first_batch_size",
                                        "first_acc_steps", 
                                        "first_warmup_steps",
                                        "second_lr",
                                        "second_batch_size",
                                        "second_acc_steps", 
                                        "second_warmup_steps"
                                        ],

                        metric_columns=["eval_val_pr_auc",
                                        #'val_f1',            
                                       #"val_recall",
                                       #"val_precision",
                                       #"val_selected_f1",
                                       #"val_selected_precision",
                                       #"val_selected_recall", 
                                        ]
                            )


        analysis = tune.run(tune.with_parameters(tune_adapter,
                                                model_name=model_name,
                                                task=task, 
                                                language=language,
                                                data_paths=data_paths,
                                                is_first=is_first,
                                                dataset_name=eval_dataset,
                                                task_adapter_path=task_adapter_path
                                                ),    
                        local_dir= f"/pvc/raytune_{filter_set_name}/",
                        resources_per_trial=resources_per_trial,
                        metric="eval_val_auc",
                        mode="max",
                        config=config,
                        num_samples=50,
                        scheduler=scheduler,
                        search_alg=search,
                        progress_reporter=reporter,
                        name=f"tune_adapter_{experiment_name}", 
                        checkpoint_at_end=True,)

        best_config = analysis.get_best_config()
        with open(f"/pvc/tasks/codie_ccs_based_data/{experiment_name}_best_config_{filter_set_name}_fold_{nfold}.pcl",'wb') as f: 
                pickle.dump(best_config, f)

        analysis.best_result_df.to_csv(f"/pvc/tasks/codie_ccs_based_data/{experiment_name}_best_result_{filter_set_name}_fold_{nfold}.csv", index=False)
        analysis.dataframe().to_csv(f"/pvc/tasks/codie_ccs_based_data/{experiment_name}_best_result_{filter_set_name}_fold_{nfold}.csv", index=False)
                

        