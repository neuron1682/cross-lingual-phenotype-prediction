import sys 
sys.path.append('/pvc/')
import src.evaluation_utils as evaluation_utils
import utils.utils as utils
import datasets
import pandas as pd
import numpy as np 

def save_adapter_metrics(data_paths, language, eval_dataset_name, eval_type, eval_model_path, output_path, nsamples):

    train_dataset, dev_dataset, test_dataset, labels = utils.get_datav2(data_paths,
                                                                        dataset_name=eval_dataset_name)


    #if eval_type == 'zero_shot_ccs' and eval_dataset_name == 'codie': 
        #dataset = datasets.concatenate_datasets([train_dataset, dev_dataset, test_dataset])

    if eval_type in ['few_shot_ccs', 'zero_shot_ccs', 'top_codes', 'average']:
        dataset = test_dataset

    metrics = evaluation_utils.evaluate_adapter(language=language, 
                                                model_path=eval_model_path, 
                                                model_name="xlm-roberta-base", 
                                                labels=labels, 
                                                dataset=dataset, 
                                                dataset_name=eval_dataset_name)

    if eval_type != 'average':
        result = evaluation_utils.compute_metrics(metrics, 
                                                    eval_type=eval_type, 
                                                    data_paths=data_paths, 
                                                    dataset_name=eval_dataset_name, 
                                                    nsamples=nsamples)
    else: 
        result = dict()
        result['avg_pr_auc'] = metrics['eval_val_pr_auc']
        result['avg_micro_auc'] = metrics['eval_micro_auc']
        result['avg_auc'] = metrics['eval_val_auc']

    import numpy as np
    print('avg_auc', np.round(result['avg_auc']*100, 2))
    print('avg_micro_auc', np.round(result['avg_micro_auc']*100,2))
    print('avg_pr_auc', np.round(result['avg_pr_auc']*100,2))
    

    #evaluation_utils.save_metric(metric=result,
     #                           metric_path=output_path.format(f"{language}_{eval_dataset_name}_{eval_type}_metrics"))

    return result


if __name__ == '__main__': 
    
    eval_models = evaluation_utils.get_best_adapter_model_paths(mname=None)
    output_path = "/pvc/tasks/codie_ccs_based_data/evaluation_metrics/{}.pcl"


    # SETTINGS
    nfold = 3
    filter_set_name = 'ccs_codie'
    translator_data_selector=None #'Opus_es_to_en'
    eval_dataset_name = 'codie'
    language = 'spanish'
    mname = 'mimic_achepa_codie_MLA'
    eval_type = 'few_shot_ccs' #'zero_shot_ccs' #average #few_shot_ccs, top_codes
    eval_model_path = eval_models[mname]
    data_paths = evaluation_utils.get_data_paths(nfold, filter_set_name, eval_dataset_name, translator_data_selector)

    'achepa_original_SLA'      
    'codie_original_SLA'       
    'mimic_original_SLA'       
    'mimic_achepa_MLA'         
    'mimic_achepa_codie_MLA'
    'mimic_codie_MLA'          
    'mimic_codie_MLA_full_ft'
    'achepa_codie_MLA'         

    if eval_type == 'few_shot_ccs':
        #{'min':1, 'max':5}, 
        groups = [{'min':0, 'max':10}, {'min': 11, 'max':50}, 
                  {'min': 51, 'max':100}, {'min': 101, 'max':1e4}]

        if eval_dataset_name == 'codie': 
            #groups = [{'min': 0, 'max':0}] + groups
            plot_df = pd.DataFrame(np.zeros((4)))
        else: 
            plot_df = pd.DataFrame(np.zeros((4)))

        for idx,nsamples in enumerate(groups):
            
            result = save_adapter_metrics(data_paths, 
                                language=language, 
                                eval_dataset_name=eval_dataset_name, 
                                eval_type=eval_type,
                                eval_model_path=eval_model_path,
                                output_path=output_path, 
                                nsamples=nsamples,)
            plot_df.iloc[idx] = result['avg_auc']
            result = dict([(k,result[k]) for k in result if k not in ['avg_auc', 'avg_micro_auc', 'avg_pr_auc']])
            pd.DataFrame.from_dict(result).to_csv(f"{mname}_{nsamples['max']}_auc_scores", index=False)
        plot_df.to_csv(f'/pvc/auc_per_train_sample_{mname}_new.csv', index=False)
    else: 
        save_adapter_metrics(data_paths, 
                                language=language, 
                                eval_dataset_name=eval_dataset_name, 
                                eval_type=eval_type,
                                eval_model_path=eval_model_path,
                                output_path=output_path, 
                                nsamples=None,)
                                
    
    
