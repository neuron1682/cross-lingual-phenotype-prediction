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

"""
def get_datav2(data_path, train_lng): 
  
        data_class = codiespDataset(data_path, train_lng)
        codiesp_dataset, all_labels = data_class.load_codiesp_dataset()
        #codiesp_dataset, all_labels = data_class.load_codiesp_mixed_lng_dataset(train_lng)
        codiesp_dataset = data_class.tokenize_dataset(codiesp_dataset)
        codiesp_dataset = data_class.transform_labels(codiesp_dataset, 
                                                     all_labels=all_labels)
        num_labels = len(all_labels)
        print(codiesp_dataset.num_rows)
        return codiesp_dataset["train"], codiesp_dataset["validation"], codiesp_dataset["test"], num_labels
"""

def manipulate_zero_shot_diagnosis(x, idx): 
        x['labels'][idx] = 0
        return x 

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


def tune_multilingual_adapter(config,
                              model_name, 
                              task, 
                              languages,
                              data_paths,
                              ):
                              
        language_codes = {'spanish':'es',
                        'english':'en'
                        }

        #utils.set_seeds(seed=config['seed'])

        ########################### SETUP ADAPTER ########################### 
        first_train_language = languages[0]
        #train_data_path = f'/pvc/codiesp/xl-training-data/v4/{task}_task/rebalanced/{first_train_language}_clinical_text'
        first_train_dataset, first_dev_dataset, first_test_dataset, labels = utils.get_datav2(data_paths,
                                                                                                first_train_language)
        task_adapter_name = f'codiesp_{task}_v4'

        codieSP_adapter = AdapterSetup(len(labels),
                                        languages, 
                                        task_adapter_name, 
                                        model_name=model_name)

        ########################### TRAINING FIRST LANGUAGE ########################### 
        adapter_trainer = AdapterTrainer(task_adapter_name, 
                                        model=codieSP_adapter.final_adapter_model)

        adapter_trainer.train_adapter(is_first=True,
                                lm_model_code=language_codes[first_train_language], 
                                train_dataset=first_train_dataset,
                                eval_dataset= first_dev_dataset,
                                config=config,
                                #learning_rate=1e-3,
                                )

        ########################### EVALUATING IN THE FIRST OF THE LANGUAGES ########################### 
        """
        eval_lng = first_train_language
        eval_lm_model_code = language_codes[first_train_language]
        eval_data_path = f'/pvc/codiesp/xl-training-data/v4/diagnosis_task/rebalanced/{eval_lng}_clinical_text'

        eval_dataset = first_dev_dataset
        adapter_trainer.evaluate_adapter(eval_dataset=eval_dataset, 
                                        lm_model_code=eval_lm_model_code,
                                        num_labels=num_labels,)

        eval_lm_model_code = language_codes[first_train_language]
        metrics_es = adapter_trainer.evaluate_adapter(eval_dataset=first_test_dataset, 
                                                        lm_model_code=eval_lm_model_code,
                                                       num_labels=num_labels,)
        """
        ########################### TRAINING SECOND LANGUAGE ########################### 
        second_train_language = languages[1]
        second_train_dataset, second_dev_dataset, second_test_dataset, labels = utils.get_datav2(data_paths,
                                                                                                second_train_language,)

        adapter_trainer = AdapterTrainer(task_adapter_name, 
                                        model=codieSP_adapter.final_adapter_model,)

        adapter_trainer.train_adapter(is_first=False,
                                lm_model_code=language_codes[second_train_language], 
                                train_dataset=second_train_dataset,
                                eval_dataset= second_dev_dataset,
                                #learning_rate=1e-4,
                                config=config,
                                #learning_rate=config['second_lr'],
                                #num_train_epochs=config['second_num_epochs'],
                                )

        #tune.report(val_auc=0.99)
        ########################### EVALUATING IN BOTH OF THE LANGUAGES ########################### 
        
        eval_lm_model_code = language_codes[first_train_language]
        metrics = adapter_trainer.evaluate_adapter(eval_dataset=first_dev_dataset, 
                                                lm_model_code=eval_lm_model_code,
                                                num_labels=len(labels),
                                                is_trained=True,
                                                )
        if eval_lm_model_code == 'es':
                metrics_es = metrics
        else: 
                metrics_en = metrics

        eval_lm_model_code = language_codes[second_train_language]
        metrics = adapter_trainer.evaluate_adapter(eval_dataset=second_dev_dataset, 
                                                lm_model_code=eval_lm_model_code,
                                                num_labels=len(labels),
                                                is_trained=True,)

        if eval_lm_model_code == 'es':
                metrics_es = metrics
        else: 
                metrics_en = metrics

        print(metrics_es)
        print(metrics_en)

        avg_all = (metrics_es['eval_complete_val_auc'] + metrics_en['eval_complete_val_auc'])/2.

        eval_val_selected_auc_avg = (metrics_es['eval_complete_val_selected_auc'] + metrics_en['eval_complete_val_selected_auc'])/2.
        
        ########################### EVALUATING TEST IN BOTH OF THE LANGUAGES ########################### 
        eval_lm_model_code = language_codes[first_train_language]
        test_metrics_es = adapter_trainer.evaluate_adapter(eval_dataset=first_test_dataset, 
                                                        lm_model_code=eval_lm_model_code,
                                                        num_labels=len(labels),)

        eval_lm_model_code = language_codes[second_train_language]
        test_metrics_en = adapter_trainer.evaluate_adapter(eval_dataset=second_test_dataset, 
                                                lm_model_code=eval_lm_model_code,
                                                num_labels=len(labels),)


        tune.report(
                en_val_loss=metrics_en['eval_complete_val_loss'],
                es_val_loss=metrics_es['eval_complete_val_loss'],
                en_val_auc=metrics_en['eval_complete_val_auc'], 
                es_val_auc=metrics_es['eval_complete_val_auc'], 
                en_val_selected_auc=metrics_en['eval_complete_val_selected_auc'],
                es_val_selected_auc=metrics_es['eval_complete_val_selected_auc'],
                eval_val_auc_avg=avg_all,
                eval_val_selected_auc_avg=eval_val_selected_auc_avg
               )

        
if __name__ == "__main__":

        #ray.init(local_mode=True)
        cluster = False

        if not cluster:
                print('starting ray cluster locally for debugging')
                ray.init(local_mode=True)
                
        else:
                print('starting ray  with the ray service on the CLUSTER')
                ray.init(address=os.environ["RAY_HEAD_SERVICE_HOST"] + ":6379")

        is_first = True
        mname = 'SLA'
        model_name = 'xlm-roberta-base'
        translator_data_selector = None #'Opus_es_en_concat_notes'
        filter_set_name = 'ccs_codie'
        eval_dataset = 'codie'
        languages = ['spanish']
        language = 'spanish'
        mla_order = '_'.join(languages)
        task = 'diagnosis' #'diagnosis', zero_shot_diag_ccs
        nfold = 3

        test = True


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

                experiment_name = experiment_name+"_TEST"           
        #Mimic only - best adapter model
        #task_adapter_path = "/pvc/MLA_raytune_adapter_spanish_english/tune_adapter_hpo_adapters/_inner_fab17cbc_17_first_acc_steps=4,first_batch_size=8,first_lr=0.0015909,first_num_epochs=100,first_warmup_steps=750,first_weigh_2021-07-15_09-38-26/training_output_en_0.0015908617123582417_0/checkpoint-13141/"
        #task_adapter_path = "/pvc/raytune_greek_spanish_english/tune_adapter_mimic_SLA/_inner_aa242fb2_25_first_acc_steps=2,first_attention_dropout=0.3,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.00083617,f_2021-08-28_02-38-59/training_output_en_0.0008361731926701554_0/checkpoint-30072"
        #task_adapter_path = '/pvc/raytune_greek_spanish_english/tune_adapter_mimic_SLA/_inner_aa242fb2_25_first_acc_steps=2,first_attention_dropout=0.3,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.00083617,f_2021-08-28_02-38-59/training_output_en_0.0008361731926701554_0/checkpoint-31504'
        task_adapter_mimic_sla_path = '/pvc/raytune_ccs_codie/tune_adapter_mimic_original_SLA/_inner_2c36a0d2_50_first_acc_steps=4,first_attention_dropout=0.1,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.00042751,f_2021-10-20_00-03-23/training_output_en_0.0004275118309968961_0/checkpoint-13914/'
        task_adapter_achepa_sla_path = '/pvc/raytune_ccs_codie/tune_adapter_achepa_original_SLA/_inner_4cc57928_35_first_acc_steps=2,first_attention_dropout=0.1,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.0052487,fi_2021-10-20_14-09-26/training_output_el_0.005248721818032698_0/checkpoint-198'
        task_adapter_mimic_achepa_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_greek_MLA/_inner_b566d67e_34_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-10-20_17-58-50/training_output_el_0_0.0005403420575244382/checkpoint-3781'
        task_adapter_mimic_codie_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_spanish_MLA/_inner_b697bfc6_12_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-10-20_10-52-50/training_output_es_0_0.0011030338137158105/checkpoint-160'
        task_adapter_codie_sla_path = '/pvc/raytune_ccs_codie/tune_adapter_codie_original_SLA/_inner_b34ab760_27_first_acc_steps=2,first_attention_dropout=0.3,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.0076105,fi_2021-10-19_14-34-52/training_output_es_0.007610478516231566_0/checkpoint-205'
        task_adapter_achepa_mimic_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_greek_english_diagnosis_MLA/_inner_3a0ea5a2_41_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-11-16_03-20-21/training_output_en_0_0.0017506470138346506/checkpoint-6176'
        task_adapter_codie_mimic_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_spanish_english_diagnosis_MLA/_inner_7ceb71c6_34_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-11-15_14-43-55/training_output_en_0_0.0008006564657455058/checkpoint-6957'
        
        task_adapter_path = None #task_adapter_codie_sla_path

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

        #run settings
        # determines which dataset is loaded according to language from data_paths
        #language = 'spanish'
        
        #languages = ['english', 'spanish']
        #filter_set_name = 'greek_spanish_english'
        #mname = 'MLA'
        #model_name = 'xlm-roberta-base'
        #dataset_name = 'mimic_codie'
        #experiment_name = f"{dataset_name}_{mname}_test"
        
        #model_name = 'xlm-roberta-base'
        #task = 'diagnosis'
        #order matters
        #languages = ['english', 'spanish']
        #language = 'spanish'
        
        #nfold = 3
        #filter_set_name = 'spanish_english'
        #experiment_name = 'mimic_codie_hpo_adapters'

        data_paths = {'train_data_path_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/mimic_fold_{nfold}_train.csv",
                'validation_data_path_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/mimic_fold_{nfold}_dev.csv",
                'test_data_path_en': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/mimic_fold_{nfold}_test.csv",
                'train_data_path_es': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/codiesp_fold_{nfold}_train.csv",
                'validation_data_path_es': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/codiesp_fold_{nfold}_dev.csv",
                'test_data_path_es': f"/pvc/tasks/english_spanish_task_data/k_folds/fold_{nfold}/codiesp_fold_{nfold}_test.csv", 
                'all_labels_path': f"/pvc/tasks/{filter_set_name}_labels.pcl", 
                }



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
                'zero_shot_ccs_path': '/pvc/tasks/codie_ccs_based_data/top10_mmc_codie_achepa.csv', 
                'zero_shot_diagnoses_path': None,

                }


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


        #reporter = CLIReporter(
        #                parameter_columns=["first_lr",
        #                                "first_batch_size",
        #                                "first_acc_steps", 
        #                                "first_warmup_steps",
        #                                "second_lr",
        #                                "second_batch_size",
        #                                "second_acc_steps", 
        #                                "second_warmup_steps"
        #                                ],#

        #                metric_columns=["en_loss",
        #                                "es_loss"
        #                                'en_val_loss',
        #                                'es_val_loss',
        #                                "en_val_auc",
        #                                "es_val_auc",
        #                                "en_val_selected_auc",
        #                                "es_val_selected_auc",
        #                                "eval_val_auc",
        #                                "eval_complete_val_auc",
        #                                #'val_f1',            
                                        #"val_recall",
                                        #"val_precision",
                                        #"val_selected_f1",
                                        #"val_selected_precision",
                                        #"val_selected_recall", 
        #                                ]
        #                    )


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
                        resources_per_trial={'cpu': 8, "gpu":1},
                        metric="eval_val_auc",
                        mode="max",
                        config=config,
                        num_samples=50,
                        #metric="loss",
                        #mode='min',
                        scheduler=scheduler,
                        search_alg=search,
                        progress_reporter=reporter,
                        name=f"tune_adapter_{experiment_name}", 
                        checkpoint_at_end=True,)
                        #src/baselines/hpo_spanish_baseline.py


        #print("best config: ", analysis.get_best_config(metric="eval_val_auc_avg", mode="max"))
        #print("best trial: ", analysis.get_best_trial(metric="eval_val_auc_avg", mode="max"))

        
        #best_config = analysis.get_best_config()
        #with open(f"/pvc/tasks/english_spanish_task_data/hpo_result/MLA_{language}_best_config_{filter_set_name}_fold_{nfold}.pcl",'wb') as f: 
        #        pickle.dump(best_config, f)

        #analysis.best_result_df.to_csv(f"/pvc/tasks/english_spanish_task_data/hpo_result/MLA_{language}_best_result_{filter_set_name}_fold_{nfold}.csv", index=False)
        #analysis.dataframe().to_csv(f"/pvc/tasks/english_spanish_task_data/hpo_result/MLA_{language}_best_result_{filter_set_name}_fold_{nfold}.csv", index=False)
    

        best_config = analysis.get_best_config()
        with open(f"/pvc/tasks/codie_ccs_based_data/{experiment_name}_best_config_{filter_set_name}_fold_{nfold}.pcl",'wb') as f: 
                pickle.dump(best_config, f)

        analysis.best_result_df.to_csv(f"/pvc/tasks/codie_ccs_based_data/{experiment_name}_best_result_{filter_set_name}_fold_{nfold}.csv", index=False)
        analysis.dataframe().to_csv(f"/pvc/tasks/codie_ccs_based_data/{experiment_name}_best_result_{filter_set_name}_fold_{nfold}.csv", index=False)
                

        