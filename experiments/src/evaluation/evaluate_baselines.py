import sys 
sys.path.append('/pvc/')
import src.evaluation_utils as evaluation_utils
import src.utils.utils as utils
import datasets
from src.baselines.spanish_bert import SpanishBertBaseline
import pytorch_lightning as pl
import os 
import torch
import pandas as pd
import numpy as np

trainer = pl.Trainer(#precision=16,
                        gpus=1, 
                        min_epochs=98,
                        max_epochs=98,
                        fast_dev_run=False,
                        accumulate_grad_batches=1
                        )


def prepare_bert_model(num_labels, pretrained_model_name, eval_model_path): 
        # IMPORT FROM UTILS

    dummy_config = {"lr": 2e-5,
              "batch_size": 8,
              "acc_grads": 1,
              "warmup_steps": 0,
              "num_training_steps":10, 
              'hidden_dropout':0,
              'attention_dropout':0,
            }

    model_base = SpanishBertBaseline(dummy_config, num_labels=num_labels, model_name=pretrained_model_name, num_training_steps=None)
    model_checkpoint = torch.load(eval_model_path)
    print('LOAD MODEL FROM STATE DICT')
    model_base.load_state_dict(model_checkpoint['state_dict'])
    device = 'cuda'
    model = model_base.to(device)

    return  model

def evaluate_berts(data_paths, 
                language, 
                eval_dataset, 
                eval_type, 
                eval_model_path, 
                pretrained_model_name, 
                output_path, 
                nsamples=None):

    train_dataloader, dev_dataloader, test_dataloader, labels = utils.get_data(model_name=pretrained_model_name,
                                                                                data_paths=data_paths, 
                                                                                language=language,
                                                                                eval_dataset=eval_dataset,
                                                                                task=None,
                                                                                batch_size=32, 
                                                                                do_eval=True)
    
    model = prepare_bert_model(len(labels), pretrained_model_name, eval_model_path)
    

    if eval_type == 'zero_shot_ccs': 
        #dataset = torch.utils.data.ConcatDataset([train_dataloader, dev_dataloader, test_dataloader])
        dataset = test_dataloader
    elif eval_type in ['few_shot_ccs', 'top_codes_ccs', 'average']:
        dataset = test_dataloader

    trainer.test(model, test_dataloaders=dataset)
    metrics = model.test_results
    
    if eval_type != 'average':
        result = evaluation_utils.compute_metrics(metrics, 
                                                eval_type=eval_type, 
                                                data_paths=data_paths, 
                                                dataset_name=eval_dataset, 
                                                nsamples=nsamples)
        
    else: 
        result = dict()
        result['avg_pr_auc'] = metrics['eval_val_pr_auc']
        result['avg_micro_auc'] = metrics['eval_micro_auc']
        result['avg_auc'] = metrics['eval_val_auc']

    evaluation_utils.save_metric(metric=result,
                                metric_path=output_path.format(f"{language}_{eval_dataset}_{eval_type}_metrics"))

    return result


def get_model_names(): 
    model_names = {"spanish_bert":'dccuchile/bert-base-spanish-wwm-cased',
                "spanish_biobert": 'fvillena/bio-bert-base-spanish-wwm-uncased',
                "english_bert": 'bert-base-cased', 
                "pubmedBert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
                "greek_bert": "nlpaueb/bert-base-greek-uncased-v1",
                "multilingual_bert": 'bert-base-multilingual-cased', 
                "xlmr": "xlm-roberta-base",
                }
    
    return model_names

if __name__ == '__main__': 
    
    eval_models = evaluation_utils.get_best_baselines_model_paths(mname=None)
    output_path = "/pvc/tasks/codie_ccs_based_data/evaluation_metrics/{}.pcl"

    
    # SETTINGS
    nfold = 3
    filter_set_name = 'ccs_codie'
    translator_data_selector = 'Opus_el_en'  #'official_translation' #'Opus_el_en'
    eval_dataset = 'achepa'
    language = 'spanish'
    mname = 'mimic_achepa_opus_pubmed'
    eval_type = 'few_shot_ccs' #'zero_shot_ccs' #average #long_tail_ccs, top_codes few_shot_ccs
    eval_model = eval_models[mname]
    data_paths = evaluation_utils.get_data_paths(nfold, filter_set_name, eval_dataset, translator_data_selector)

    '''  
    dict_keys(['achepa_original_greek_bert', 
    'achepa_original_xlmr', 
    'codie_original_spanish_bert', 
    'codie_original_xlmr', 
    'codie_original_spanish_biobert', 
    'mimic_original_pubmedBert', 
    'achepa_opus_pubmed', 
    'codie_off_pubmed', 
    'mimic_achepa_opus_pubmed', 
    'mimic_achepa_codie_off_pubmed', 
    'mimic_codie_off_pubmed', 
    'achepa_codie_off_pubmed'])
    '''
    
    if eval_type == 'few_shot_ccs':
        #{'min':1, 'max':5}, 
        groups = [{'min':0, 'max':10}, {'min': 11, 'max':50}, 
                  {'min': 51, 'max':100}, {'min': 101, 'max':1e4}]

        if eval_dataset == 'codie': 
            #groups = [{'min': 0, 'max':0}] + groups
            plot_df = pd.DataFrame(np.zeros((4)))
        else: 
            plot_df = pd.DataFrame(np.zeros((4)))

        for idx,nsamples in enumerate(groups):
                result = evaluate_berts(data_paths, 
                                        language=language, 
                                        eval_dataset=eval_dataset, 
                                        pretrained_model_name=eval_model['base_model_path'],
                                        eval_type=eval_type,
                                        eval_model_path=eval_model['best_model_path'],
                                        output_path=output_path, 
                                        nsamples=nsamples)
                    
                print('evaluation for {}'.format(nsamples))
                import numpy as np
                avg_auc = np.round(result['avg_auc'] * 100, 2)
                avg_micro_auc = np.round(result['avg_micro_auc'] * 100, 2)
                avg_pr_auc = np.round(result['avg_pr_auc'] * 100, 2)
                print('avg_auc', avg_auc)
                print('avg_micro_auc', avg_micro_auc)
                print('avg_pr_auc', avg_pr_auc)
                plot_df.iloc[idx] = avg_auc
                #plot_df.loc[idx, 'avg_auc'] = avg_auc
                #plot_df.loc[idx, 'avg_pr_auc'] = avg_pr_auc
                result = dict([(k,result[k]) for k in result if k not in ['avg_auc', 'avg_micro_auc', 'avg_pr_auc']])
                pd.DataFrame.from_dict(result).to_csv(f"{mname}_{nsamples['max']}_auc_scores", index=False)
        plot_df.to_csv(f'/pvc/auc_per_train_sample_{mname}.csv', index=False)

        
    else: 
        result = evaluate_berts(data_paths, 
                                    language=language, 
                                    eval_dataset=eval_dataset, 
                                    pretrained_model_name=eval_model['base_model_path'],
                                    eval_type=eval_type,
                                    eval_model_path=eval_model['best_model_path'],
                                    output_path=output_path, 
                                    nsamples=None)
                    
        import numpy as np
        avg_auc = np.round(result['avg_auc'] * 100, 2)
        avg_micro_auc = np.round(result['avg_micro_auc'] * 100, 2)
        avg_pr_auc = np.round(result['avg_pr_auc'] * 100, 2)
        print('avg_auc', avg_auc)
        print('avg_micro_auc', avg_micro_auc)
        print('avg_pr_auc', avg_pr_auc)
    
    



