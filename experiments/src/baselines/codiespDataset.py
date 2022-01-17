from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer
import logging
import ast
import json
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

class codiespDataset(): 

  def __init__(self, data_path):
    self.data_path = data_path

  def load_codiesp_dataset(self):
    
    dataset = load_dataset('csv', data_files={'train': f'{self.data_path}/train_data_cutoff.csv',
                                            'validation': f'{self.data_path}//dev_data_cutoff.csv'})
    logging.warning(f' number of samples: {dataset.num_rows}')

    with open(f'{self.data_path}/all_codes_cutoff.json') as f: 
            all_labels = json.load(f)['all_D_codes']

    return dataset, all_labels
  
  def load_codiesp_mixed_lng_dataset(self, train_lng):
    
    dataset = load_dataset('csv', data_files={'train': f'{self.data_path}/{train_lng}_half_train_data_cutoff.csv',
                                            'validation': f'{self.data_path}//dev_data_cutoff.csv'})
    logging.warning(f' number of samples: {dataset.num_rows}')

    with open(f'{self.data_path}/all_codes_cutoff.json') as f: 
            all_labels = json.load(f)['all_D_codes']

    return dataset, all_labels

  def encode_batch(self, batch):
    """Encodes a batch of input data using the model tokenizer."""
    return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

  def tokenize_dataset(self, dataset): 
    # Encode the input data
    dataset = dataset.map(self.encode_batch, batched=True)
    dataset.rename_column_("label", "labels")
    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    return dataset

  def labels_to_tensor(self, curr_labels, label_to_pos_map):
    tmp = np.zeros((len(label_to_pos_map)))
    for l in curr_labels:
      tmp[label_to_pos_map[l]] = 1 
    return torch.tensor(tmp)

  def map_labels(self, item, label_to_pos_map):  
    curr_labels = ast.literal_eval(item['labels'])
    label_arr = self.labels_to_tensor(curr_labels, label_to_pos_map)
    return {'attention_mask': item["attention_mask"], 'input_ids': item['input_ids'], 'labels': label_arr}
  

  def label_to_pos(self, all_codes):
        label_to_pos_map = dict([(code, pos) for code, pos in zip(sorted(all_codes),range(len(all_codes)))])
        pos_to_label_map = dict([(pos, code) for code, pos in zip(sorted(all_codes),range(len(all_codes)))])
        return label_to_pos_map, pos_to_label_map

  def transform_labels(self, dataset, all_labels):
    label_to_pos_map, _ = self.label_to_pos(all_labels)
    dataset = dataset.map(lambda x: self.map_labels(x, label_to_pos_map), batched=False)
    dataset.set_format(type='torch')
    self.dataset = dataset
    return dataset


#### create label array/ create utils file
### create github possibility 
## update baseline github 
### train with zero shot prediction and measure auc 
### incorporate multilingual knowledge


if __name__ == "__main__":

  path = "/pvc/codiesp/xl-training-data/v2/diagnosis_task/spanish_clinical_text"
  data_class = codiespDataset(path)
  codiesp_dataset, all_labels = data_class.load_codiesp_dataset()
  codiesp_dataset = data_class.tokenize_dataset(codiesp_dataset)
  codiesp_dataset = data_class.transform_labels(codiesp_dataset, all_labels=all_labels)
  print(codiesp_dataset.num_rows)



