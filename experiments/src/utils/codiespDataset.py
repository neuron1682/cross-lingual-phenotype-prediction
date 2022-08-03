from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer
import logging
import ast
import pickle
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

class codiespDataset(): 

  def __init__(self, data_paths, dataset_name):
    self.dataset_name = dataset_name
    self.data_paths = data_paths
   
  def load_codiesp_dataset(self):

    if not self.data_paths['translator_data_selector']:

      dataset = load_dataset('csv', data_files={'train': self.data_paths[f"train_data_path_{self.dataset_name}"],
                                                  'validation': self.data_paths[f"validation_data_path_{self.dataset_name}"],
                                                  'test': self.data_paths[f"test_data_path_{self.dataset_name}"],
                                                })

    elif self.data_paths['translator_data_selector'] in ['Opus_es_to_en', 'GT_es_to_en']:
            
            dataset = load_dataset('csv', data_files={'train': self.data_paths[f"translation_train_data_path_codie_en"],
                                                  'validation': self.data_paths[f"translation_validation_data_path_codie_en"],
                                                  'test': self.data_paths[f"translation_test_data_path_codie_en"],
                                  })

    elif self.data_paths['translator_data_selector'] in ['Opus_es_en_concat_notes', 'GT_es_en_concat_notes']:
            dataset = load_dataset('csv', data_files={'train': self.data_paths[f"concat_train_data_path_mimic_codie_en"],
                                                  'validation': self.data_paths[f"concat_validation_data_path_mimic_codie_en"],
                                                  'test': self.data_paths[f"concat_test_data_path_mimic_codie_en"],
                                  })

    logging.warning(f' number of samples: {dataset.num_rows}')
    #f'{self.data_path}/all_codes_cutoff.json'
    with open(self.data_paths['all_labels_path'], 'rb') as f: 
      #all_labels = json.load(f)['all_D_codes']
      all_labels = pickle.load(f)
    return dataset, all_labels

  def encode_batch(self, batch):
    """Encodes a batch of input data using the model tokenizer."""
    if not self.data_paths['translator_data_selector']:
      return tokenizer(batch["notes"], max_length=512, truncation=True, padding="max_length")
    elif self.data_paths['translator_data_selector'] in ['Opus_es_to_en', 'GT_es_to_en']: 
      return tokenizer(batch[self.data_paths['translator_data_selector']], max_length=512, truncation=True, padding="max_length")

  def tokenize_dataset(self, dataset): 
    # Encode the input data
    dataset = dataset.map(self.encode_batch, batched=True)
    #dataset = dataset.rename_column("ICD10", "labels")
    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    return dataset

  def labels_to_tensor(self, curr_labels, label_to_pos_map):
    tmp = np.zeros((len(label_to_pos_map)))
    for l in curr_labels:
      try:
        tmp[label_to_pos_map[l]] = 1
      except:
        pass
    if tmp.sum() > 0:
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
    dataset = dataset.filter(lambda x: x['labels'] is not None)
    dataset.set_format(type='torch')
    self.dataset = dataset
    return dataset


if __name__ == "__main__":


  data_class = codiespDataset(path)
  codiesp_dataset, all_labels = data_class.load_codiesp_dataset()
  codiesp_dataset = data_class.tokenize_dataset(codiesp_dataset)
  codiesp_dataset = data_class.transform_labels(codiesp_dataset, all_labels=all_labels)
  print(codiesp_dataset.num_rows)



