from transformers import BertForSequenceClassification, XLMRobertaForSequenceClassification, RobertaForSequenceClassification #get_constant_schedule_with_warmup
from torch.nn import functional as F
import torch
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
#from pytorch_lightning.metrics.functional.classification import auroc
from sklearn.metrics import roc_auc_score as auroc
from pytorch_lightning.metrics.classification import F1
import sys
sys.path.append('/pvc/')
import  src.utils.utils as utils
from ray import tune
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import BCEWithLogitsLoss
import logging
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import numpy as np



class SpanishBertBaseline(pl.LightningModule):
    def __init__(self, 
                config,
                num_labels, 
                model_name,
                num_training_steps
                ):

        super(SpanishBertBaseline, self).__init__()
        
        if model_name != "xlm-roberta-base" and not 'BSC' in model_name:
            
            self.spanish_bert = BertForSequenceClassification.from_pretrained(model_name, 
                                                                            num_labels=num_labels,
                                                                            output_attentions=False,
                                                                            output_hidden_states=False,
                                                                            hidden_dropout_prob=config['hidden_dropout'],
                                                                            attention_probs_dropout_prob=config['attention_dropout'],
                                                                            )
        elif model_name == "xlm-roberta-base": 

            self.spanish_bert = XLMRobertaForSequenceClassification.from_pretrained(model_name, 
                                                                                    num_labels=num_labels,
                                                                                    output_attentions=False,
                                                                                    output_hidden_states=False,
                                                                                    hidden_dropout_prob=config['hidden_dropout'],
                                                                                    attention_probs_dropout_prob=config['attention_dropout'],
                                                                                    )
        elif 'BSC' in model_name :

            self.spanish_bert = RobertaForSequenceClassification.from_pretrained(model_name, 
                                                                                    num_labels=num_labels,
                                                                                    output_attentions=False,
                                                                                    output_hidden_states=False,
                                                                                    hidden_dropout_prob=config['hidden_dropout'],
                                                                                    attention_probs_dropout_prob=config['attention_dropout'],
                                                                                    )


        self.lr = config['lr'] 
        self.num_training_steps = num_training_steps
        self.batch_size = config['batch_size']
        self.num_labels = num_labels
        self.model_name = model_name
        self.save_hyperparameters(config)
        self.recall_func = recall_score
        self.precision_func = precision_score
        self.dummy_f1_func = f1_score
        self.auc_func = auroc
        #Accepts logits from a model output
        self.f1_func = F1(num_classes=self.num_labels,
                          average='macro',
                          multilabel=True,
                        )

        self.loss_func = BCEWithLogitsLoss(reduce="mean")
        self.num_warmup_steps = config['warmup_steps']


    def compute_label_thresholds(self, metric_func, y_true, y_pred):
        thresh = torch.tensor(np.round(np.arange(0, 1, 0.1), 2))
        thresh_dict= {}
        num_labels = y_true.shape[1]
        for l in range(num_labels):
            best_score = 0
            t_best = 0
            for t in thresh:
                score = metric_func(y_true=y_true[:, l], y_pred=y_pred[:, l] >= t)
                if score > best_score: 
                    best_score = score
                    t_best = t
            thresh_dict[l] = (t_best, best_score)
        thresh_vec = torch.tensor(list(map(lambda x: x[0], thresh_dict.values())))
        thresh_scores = torch.tensor(list(map(lambda x: x[1], thresh_dict.values())))

        return thresh_vec, thresh_scores

    def compute_metric_scores(self, metric_func, thresh_vec, y_pred, y_true):
            y_pred_new = y_pred >= thresh_vec.view(-1, y_pred.shape[1])
            return metric_func(y_pred=y_pred_new, y_true=y_true, average=None)

    def forward(self, input_ids, attention_mask, token_type_ids=None):

        logits = self.spanish_bert(input_ids, 
                                token_type_ids=None,
                                   attention_mask=attention_mask)[0]
        return logits

    def training_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
        
        logits = self.spanish_bert(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)[0]
    
        logits = logits.view(-1, self.num_labels)
        b_labels = b_labels.type_as(logits).view(-1, self.num_labels)

        train_loss = self.loss_func(logits, b_labels)

        tensorboard_logs = {'train_loss': train_loss}
        self.log('train_loss', train_loss, on_step=True, on_epoch=False)
        return {'loss': train_loss, 'log': tensorboard_logs}

    #def training_epoch_end(self, outputs: List[Any]) -> None:
    #    return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        logits = self.spanish_bert(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask)[0]

        
        pred_logits = logits.view(-1, self.num_labels)
        y_true = b_labels.type_as(logits).view(-1, self.num_labels)
        val_loss = self.loss_func(pred_logits, y_true)

        self.log('step_val_loss', val_loss, on_step=False, on_epoch=True)        

        return {'val_loss': val_loss, 'y_pred': torch.sigmoid(pred_logits), 'y_true': y_true}


    def validation_epoch_end(self, outputs):

        val_loss = torch.tensor([x['val_loss'].item() for x in outputs]).mean()
        y_true = torch.cat([x['y_true'] for x in outputs])
        y_pred = torch.cat([x['y_pred'] for x in outputs])

        selected_cols = set([248, 267, 246, 93, 259])
        cols, selected_cols = utils.get_nonzero_cols_n_rows(y_true, 
                                                     selected_cols=selected_cols)
        
        auc_score = self.auc_func(y_true[:, cols].cpu(),
                                  y_pred[:, cols].cpu(),
                                  average= None)
        
        pr_aucs = list()
        
        
        for col in cols:
            lr_precision, lr_recall, _ = precision_recall_curve(y_true[:, col].cpu(), 
                                                                y_pred[:, col].cpu())
            pr_auc = auc(lr_recall, lr_precision)
            pr_aucs.append(pr_auc)
        pr_auc = np.array(pr_aucs)

        avg_pr_auc = np.mean(pr_aucs)
        avg_auc = np.mean(auc_score)

        self.log("val_loss", val_loss)
        self.log("val_auc", avg_auc)
        self.log("val_pr_auc", avg_pr_auc)
 


    def test_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        logits = self.spanish_bert(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask)[0]

        
        pred_logits = logits.view(-1, self.num_labels)
        y_true = b_labels.type_as(logits).view(-1, self.num_labels)
        test_loss = self.loss_func(pred_logits, y_true)
        
        return {'test_loss': test_loss, 'y_pred': torch.sigmoid(pred_logits), 'y_true': y_true}


    def test_epoch_end(self, outputs):

        test_loss = torch.tensor([x['test_loss'].item() for x in outputs]).mean()
        y_true = torch.cat([x['y_true'] for x in outputs])
        y_pred = torch.cat([x['y_pred'] for x in outputs])

        selected_cols = set([248, 267, 246, 93, 259])
        cols, selected_cols = utils.get_nonzero_cols_n_rows(y_true, 
                                                     selected_cols=selected_cols)
        
        auc_score = self.auc_func(y_true[:, cols].cpu(),
                                  y_pred[:, cols].cpu(),
                                  average= None)

        
        micro_auc = self.auc_func(y_true[:, cols].cpu(),
                                  y_pred[:, cols].cpu(),
                                  average='micro')

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
        pr_auc = np.array(pr_aucs)


        avg_pr_auc = np.mean(pr_aucs)
        avg_auc = np.mean(auc_score)

        results = {"eval_auc": auc_score, 
                "eval_pr_auc ": pr_auc,
                "eval_val_auc": avg_auc,
                "eval_val_pr_auc": avg_pr_auc,
                'eval_test_loss': test_loss,
                'eval_samples_per_label': y_true.sum(axis=0),
                'eval_precision_recall_dict': precision_recall_dict,
                'eval_cols': cols,
                'eval_micro_auc': micro_auc, 
                'eval_y_true': y_true, 
                'eval_y_pred': y_pred,
                }
                
        self.test_results = results 
    
        return results

   

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                       num_warmup_steps=self.num_warmup_steps, 
                                                      num_training_steps=self.num_training_steps,
                                                    )
        lr_scheduler = {
                    'scheduler': lr_scheduler,  # The LR scheduler instance (required)
                    'interval': 'step',  # The unit of the scheduler's step size
                    'frequency': 1  # The frequency of the scheduler
                    }
        return [optimizer], [lr_scheduler]
       

def train_spanish_bert(model,
                       train_data, 
                       eval_data,
                       ):

    trainer = pl.Trainer(#precision=16,
                        gpus=1, 
                        min_epochs=4, 
                        max_epochs=1, 
                        fast_dev_run=False,
                        #callbacks=
                        ) 
                        #callbacks=[early_stop_callback])

    trainer.fit(model, 
                train_dataloader=train_data, 
                val_dataloaders=[eval_data])
    torch.save(model.state_dict(),'raytune/model.pth')     
    return trainer














    
"""
def eval_spanish_bert():
    trainer = pl.Trainer(precision=16,
                        gpus=1, 
                        min_epochs=4, 
                        max_epochs=400, 
                        fast_dev_run=False,
                        #accumulate_grad_batches=8
                        ) 
    trainer.test()

if __name__ == "__main__":

    config = {"lr": 2.26e-5,
            "batch_size": 32,    
            "num_training_steps": 60,}

    config = {"lr":  2e-5,
              "batch_size": 8,
              "acc_grads": 4,
              "warmup_steps": 0,
              "num_training_steps":10, 
              #"max_epochs":4
            }

    model_names = {"spanish_bert":'dccuchile/bert-base-spanish-wwm-cased',
                "english_bert": 'bert-base-cased', 
                "multilingual_bert": 'bert-base-multilingual-cased', 
                "xlmr": '"xlm-roberta-base"'
                }

    language ='spanish'
    task = 'diagnosis'
    model_name = 'spanish_bert'

    spanish_train_dataset, spanish_dev_dataset, spanish_test_dataset, num_labels = get_data(model_names[model_name],
                                                                                            task, 
                                                                                            language,
                                                                                            batch_size=config["batch_size"]
                                                                                            )

    model = SpanishBertBaseline(config, num_labels=num_labels, model_name=model_names[model_name])

    trainer = train_spanish_bert(
                    model=model,
                    train_data= spanish_train_dataset,
                    eval_data=spanish_dev_dataset,)
    
    print(trainer.validate(val_dataloaders=spanish_dev_dataset))
    print(trainer.validate(val_dataloaders=spanish_test_dataset))
    print(trainer.test(test_dataloaders=spanish_test_dataset))

    
    #[('r52', 145), ('r69', 141), ('r50.9', 139), ('i10', 99), ('r60.9', 95)]

#how to change optimisation goal to max auc and not loss.
"""


class ExtensionSpanishBertBaseline(SpanishBertBaseline):
    def __init__(self, 
                config,
                num_labels, 
                model_name,
                pretrained_model_path
                ):

        super(ExtensionSpanishBertBaseline, self).__init__(config,
                                                            num_labels, 
                                                            model_name,
                                                            )

        model_checkpoint = torch.load(pretrained_model_path)
        self.load_state_dict(model_checkpoint['state_dict'])
        device = 'cuda'
        self.spanish_bert = self.to(device)