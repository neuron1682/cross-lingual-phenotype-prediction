import sys
sys.path.append('/pvc/')
from src.utils import utils
from src.utils.trainer_callback import EarlyStoppingCallback
import torch
from transformers import AutoConfig, AutoModelWithHeads
from transformers import TrainingArguments#, Trainer
from src.xl_outcome_prediction_adapter.ExtendedTrainer import *
from datasets import concatenate_datasets
from transformers import AdapterType, AdapterConfig
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import roc_auc_score as auroc
from ray import tune
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import ndcg_score as ndcg
from transformers.integrations import MLflowCallback
from src.utils.trainer_callback import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import pandas as pd 

LANGUAGE_ADAPTERS = {'english': "en/wiki@ukp", 
                    'spanish': "es/wiki@ukp", 
                    'greek':   "el/wiki@ukp"
                    }

class AdapterSetup():
    def __init__(self, num_labels, languages, task_adapter_name, is_first, task_adapter_path, config, model_name="xlm-roberta-base"):
        '''
            language and task specify path of the data
            model_name: choose multilingual model
        '''

        self.languages = languages
        self.task_adapter_name = task_adapter_name

        base_model = self.setup_base_model(model_name=model_name, config=config, is_first=is_first)
    
        adapter_language_model = self.setup_language_adapter(languages=languages,
                                                            base_model=base_model,)
        if is_first:
            self.final_adapter_model = self.add_task_adapter(task_adapter_path=None,
                                                            adapter_language_model=adapter_language_model,
                                                            num_labels=num_labels,
                                                            is_first=is_first,
                                                            multilabel=True,
                                                            )
        else:
            self.final_adapter_model = self.add_task_adapter(task_adapter_path=task_adapter_path,
                                                            adapter_language_model=adapter_language_model,
                                                            num_labels=num_labels,
                                                            is_first=is_first,
                                                            multilabel=True,
                                                            )


    def setup_base_model(self, 
                        model_name,
                        config, 
                        is_first
                        ):
        base_model_config = AutoConfig.from_pretrained(model_name,)
        if is_first:
            base_model_config.attention_probs_dropout_prob = config['first_attention_dropout']
            base_model_config.hidden_dropout_prob = config['first_hidden_dropout']
        else:
            base_model_config.attention_probs_dropout_prob = config['second_attention_dropout']
            base_model_config.hidden_dropout_prob = config['second_hidden_dropout']
        base_model = AutoModelWithHeads.from_pretrained(model_name,
                                                        config=base_model_config,)
        return base_model

    def setup_language_adapter(self,
                            languages, 
                            base_model): 
        lang_adapter_config = AdapterConfig.load("pfeiffer", 
                                                reduction_factor=2)
        for language in languages:
             base_model.load_adapter(LANGUAGE_ADAPTERS[language], 
                                    AdapterType.text_lang, 
                                    config=lang_adapter_config,)
        return base_model

    def add_task_adapter(self,
                        task_adapter_path, 
                        adapter_language_model,
                        num_labels,
                        is_first,
                        multilabel=True,):

        if is_first: 
            adapter_language_model.add_adapter(self.task_adapter_name, AdapterType.text_task)
            # Add a classification head for our target task
            adapter_language_model.add_classification_head(self.task_adapter_name, num_labels=num_labels, multilabel=multilabel,)
        else:
            task_adapter_path = task_adapter_path + '/' + self.task_adapter_name
            adapter_language_model.load_head(task_adapter_path)
            adapter_language_model.load_adapter(task_adapter_path)
        
        return adapter_language_model


class AdapterTrainer(): 
    def __init__(self, 
                task_adapter_name, 
                model,):

        self.task_adapter_name = task_adapter_name
        self.model = model

    def prepare_task_adapter_training(self,
                                    adapter_task_lng_model, 
                                    lm_model_code, 
                                    dataset_name,
                                    ):

        # Unfreeze and activate fusion setup
        adapter_setup = [
                            [lm_model_code],
                            [self.task_adapter_name]
                        ]

        adapter_task_lng_model.set_active_adapters(adapter_setup)
        adapter_task_lng_model.train_adapter([self.task_adapter_name])
        if dataset_name == 'codie':
            adapter_task_lng_model.freeze_model(False)
        return adapter_task_lng_model

    def get_score_every_class(self, outputs): 
        import pickle

        auc = [x["auc"] for x in outputs]
        pr_auc = [x["pr_auc"] for x in outputs]
        cols = [x["cols"] for x in outputs]
        
        with open('tasks/spanish_english_labels.pcl', 'rb') as f:
           all_codes = pickle.load(f)

        pos_to_label = dict([(pos, code) for code, pos in zip(sorted(all_codes), range(len(all_codes)))])
        
        auc_all = np.zeros((len(auc), len(all_codes)))
        pr_auc_all = np.zeros((len(auc), len(all_codes)))

        for idx, auc_idx in enumerate(auc): 
            cols_idx = cols[idx]
            pr_auc_idx = pr_auc[idx]
            for ii, col in enumerate(cols_idx): 
                auc_all[idx, col] = auc_idx[ii]
                pr_auc_all[idx, col] = pr_auc_idx[ii]

        auc_all_df = pd.DataFrame(columns=pos_to_label.values(),
                                 data= auc_all)
        auc_all_df = auc_all_df.replace(0, np.NaN)

        pr_auc_all_df = pd.DataFrame(columns=pos_to_label.values(),
                                 data=pr_auc_all)
        pr_auc_all_df = pr_auc_all_df.replace(0, np.NaN)
        
        return auc_all_df.mean(axis=0), pr_auc_all_df.mean(axis=0), pr_auc_all_df.count(axis=0)

            

    def compute_metrics(self, is_first, p: EvalPrediction):
            label_ids_tmp = torch.tensor(p.label_ids)
            predictions_tmp = torch.tensor(p.predictions)

            cols, selected_cols = utils.get_nonzero_cols_n_rows(label_ids_tmp)


            y_true = label_ids_tmp.type_as(predictions_tmp).view(len(label_ids_tmp), -1)
            y_pred = predictions_tmp.view(len(predictions_tmp), -1)

            loss_func = BCEWithLogitsLoss(reduce="mean")
            # important! before sigmoid
            val_loss = loss_func(y_pred, y_true).item()

            y_pred = torch.sigmoid(y_pred)

            auc_score = auroc(y_true[:, cols].cpu(),
                              y_pred[:, cols].cpu(),
                              average= None)

            micro_auc_score = auroc(y_true[:, cols].cpu(),
                                    y_pred[:, cols].cpu(),
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
            avg_pr_auc = np.mean(pr_aucs)
            pr_auc = np.array(pr_aucs)


            if selected_cols:
                selected_auc = np.mean(auc_score[selected_cols])
            else: 
                selected_auc = 0


            val_ndcg_k = ndcg(y_true[:,cols], torch.sigmoid(y_pred[:,cols]), k=None)
            val_ndcg_5 = ndcg(y_true[:,cols], torch.sigmoid(y_pred[:,cols]), k=min(5, len(cols)))
            
            print("AUC score:", avg_auc_score, "PR AUC score", avg_pr_auc, 'micro_auc', micro_auc_score)
            return {
                        "pr_auc": pr_auc, 
                        "auc": auc_score,
                        "cols": torch.tensor(cols),
                        "precision_recall_dict": precision_recall_dict, 
                        "val_loss": val_loss,
                        "val_auc": avg_auc_score, 
                        "val_pr_auc": avg_pr_auc,
                        'samples_per_label': y_true.sum(axis=0),
                        'micro_auc': micro_auc_score,
                        'y_true': y_true, 
                        'y_pred': y_pred
                    }

    def train_adapter(self,
                    is_first,
                    lm_model_code, 
                    train_dataset, 
                    eval_dataset,
                    config,
                    dataset_name,
                    ): 

        model = self.prepare_task_adapter_training(adapter_task_lng_model=self.model, 
                                                   lm_model_code=lm_model_code,
                                                   dataset_name=dataset_name,)

        if dataset_name == 'codie': 
            do_save_full_model = False
            do_save_adapters = True
            
        else: 
            do_save_full_model = False
            do_save_adapters = True
            
        #second step
        logging_dir = f"./adapter_logs_{lm_model_code}_{config['first_lr']}_{config['second_lr']}"
        output_dir = f"./training_output_{lm_model_code}_{config['first_lr']}_{config['second_lr']}"
        logging_steps = 50
        # if evaluate_during_training is True then evaluation strategy is set to STEPS
        evaluate_during_training = False
        evaluation_strategy = "epoch"
        overwrite_output_dir = True
        do_eval = True
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns = False
        per_device_eval_batch_size = 32
        seed = 42
        fp16 = False
        dataloader_num_workers = 4
        load_best_model_at_end = True
        #max_steps = 1000

        if is_first:
            best_model_path = None
            training_args = TrainingArguments(
                                            learning_rate=config['first_lr'],
                                            num_train_epochs=config['first_num_epochs'],
                                            per_device_train_batch_size=config['first_batch_size'],
                                            per_device_eval_batch_size=per_device_eval_batch_size,
                                            gradient_accumulation_steps=config['first_acc_steps'],
                                            warmup_steps=config['first_warmup_steps'],
                                            weight_decay=config['first_weight_decay'],
                                            logging_steps=logging_steps,
                                            evaluate_during_training=evaluate_during_training,
                                            evaluation_strategy=evaluation_strategy,
                                            logging_dir=logging_dir,
                                            output_dir=output_dir,
                                            overwrite_output_dir=overwrite_output_dir,
                                            do_eval=do_eval,
                                            # The next line is important to ensure the dataset labels are properly passed to the model
                                            remove_unused_columns=remove_unused_columns,
                                            seed=seed,
                                            fp16=fp16,
                                            dataloader_num_workers=dataloader_num_workers,
                                            load_best_model_at_end=load_best_model_at_end,
                                            metric_for_best_model="val_auc",
                                            greater_is_better=True,
                                            save_total_limit=1,
                                        )

        else: 
            
            training_args = TrainingArguments(
                                                learning_rate=config['second_lr'],
                                                num_train_epochs=config['second_num_epochs'],
                                                #max_steps=max_steps,
                                                per_device_train_batch_size=config['second_batch_size'],
                                                per_device_eval_batch_size=per_device_eval_batch_size,
                                                gradient_accumulation_steps=config['second_acc_steps'],
                                                warmup_steps=config['second_warmup_steps'],
                                                weight_decay=config['second_weight_decay'],
                                                logging_steps=logging_steps,
                                                evaluate_during_training=evaluate_during_training,
                                                evaluation_strategy=evaluation_strategy,
                                                logging_dir=logging_dir,
                                                output_dir=output_dir,
                                                overwrite_output_dir=overwrite_output_dir,
                                                do_eval=do_eval,
                                                # The next line is important to ensure the dataset labels are properly passed to the model
                                                remove_unused_columns=remove_unused_columns,
                                                seed=seed,
                                                fp16=fp16,
                                                dataloader_num_workers=dataloader_num_workers,
                                                load_best_model_at_end=load_best_model_at_end,
                                                metric_for_best_model="val_auc",
                                                greater_is_better=True,
                                                save_total_limit=1,
                                            )
            

        trainer = ExtendedTrainer(
                        is_first=is_first,
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset, 
                        compute_metrics= lambda x: self.compute_metrics(is_first=is_first, p=x),
                        do_save_full_model=do_save_full_model,
                        do_save_adapters=do_save_adapters,
                        callbacks =[DefaultFlowCallback, EarlyStoppingCallback(early_stopping_patience=5, greater_is_better=True)]
                    )
                        
        trainer.train(model_path=None)

        #self.model = model

        
    
    def prepare_task_adapter_evaluation(self,  
                                        lm_model_code, 
                                        ):

        adapter_setup = [
                         [lm_model_code],
                         [self.task_adapter_name]
                        ]

        self.model.set_active_adapters(adapter_setup)
        return adapter_setup

   

    def evaluate_adapter(self, is_trained, eval_dataset, lm_model_code, num_labels):

        adapter_setup = self.prepare_task_adapter_evaluation(lm_model_code)

        eval_trainer = ExtendedTrainer(
                            model=self.model,
                            args=TrainingArguments(remove_unused_columns=False,
                                                   output_dir="./eval_output2"),
                            eval_dataset=eval_dataset,
                            compute_metrics=lambda x:self.compute_metrics(is_first=False, p=x),
                            adapter_names=adapter_setup,
                            is_first=None)   
                            
        eval_trainer.callback_handler.remove_callback(MLflowCallback)
        metrics = eval_trainer.evaluate(is_trained=is_trained)

        return metrics


"""
if __name__ == "__main__":
    ########################### SETUP ADAPTER ########################### 
    train_lng = 'spanish'
    train_data_path = f'/pvc/codiesp/xl-training-data/v4/diagnosis_task/rebalanced/{train_lng}_clinical_text'
    #spanish_train_dataset, num_labels = get_data(train_data_path, is_train=True)
    spanish_train_dataset, spanish_dev_dataset, spanish_test_dataset, num_labels = utils.get_datav2(train_data_path, train_lng)

    languages = ['english', 'spanish']
    task_adapter_name = 'codiesp_diagnosis_v4'

    codieSP_diagnosis_adapter = AdapterSetup(
                                            num_labels,
                                            languages, 
                                            task_adapter_name, 
                                            model_name="xlm-roberta-base")

    ########################### TRAINING FIRST LANGUAGE ########################### 
    adapter_trainer = AdapterTrainer(task_adapter_name, 
                                     model=codieSP_diagnosis_adapter.final_adapter_model)

    adapter_trainer.train_adapter(lm_model_code='es', 
                                  train_dataset=spanish_train_dataset,
                                  eval_dataset= spanish_dev_dataset,
                                  #learning_rate=1e-3,
                                  num_train_epochs=1,
                                  )

    ########################### EVALUATING IN ONE OF THE LANGUAGES ########################### 
    eval_lng = 'spanish'
    eval_lm_model_code = 'es'
    eval_data_path = f'/pvc/codiesp/xl-training-data/v4/diagnosis_task/rebalanced/{eval_lng}_clinical_text'
    #eval_dataset, _ = get_data(train_data_path, 
    #                           is_train=False)

    eval_dataset = spanish_dev_dataset
    adapter_trainer.evaluate_adapter(eval_dataset=eval_dataset, 
                                    lm_model_code=eval_lm_model_code,
                                    num_labels=num_labels,)

    ########################### TRAINING SECOND LANGUAGE ########################### 
    train_lng = 'english'
    train_data_path = f'/pvc/codiesp/xl-training-data/v4/diagnosis_task/rebalanced/{train_lng}_clinical_text'
    #english_train_dataset, _ = get_data(train_data_path, is_train=True)
    english_train_dataset, english_dev_dataset, english_test_dataset, num_labels = utils.get_datav2(train_data_path, train_lng)
    adapter_trainer = AdapterTrainer(task_adapter_name, 
                                     model=codieSP_diagnosis_adapter.final_adapter_model)

    adapter_trainer.train_adapter(lm_model_code='en', 
                                  train_dataset=english_train_dataset,
                                  eval_dataset= eval_dataset,
                                  learning_rate=1e-4,
                                  num_train_epochs=1,
                                  )

    ########################### EVALUATING IN BOTH OF THE LANGUAGES ########################### 

    metrics_es = adapter_trainer.evaluate_adapter(eval_dataset=spanish_dev_dataset, 
                                                 lm_model_code=eval_lm_model_code,
                                                 num_labels=num_labels,)
    eval_lm_model_code = 'en'
    metrics_en = adapter_trainer.evaluate_adapter(eval_dataset=english_dev_dataset, 
                                                 lm_model_code=eval_lm_model_code,
                                                 num_labels=num_labels,)
    """