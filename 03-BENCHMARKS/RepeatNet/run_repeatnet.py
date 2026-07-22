# 03-BENCHMARKS\RepeatNet\run_repeatnet.py
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Contact: francesca.stefano@unipr.it
# =============================================================================
"""RepeatNet B2B Recommendation Experiment Script

This script defines and executes a complete training and evaluation pipeline for the
RepeatNet (Repeat Aware Neural Recommendation Machine) model using the RecBole library.
It is configured to run on a pre-processed B2B dataset, which has been split
into training, validation, and test sets by the `create_splits.py` script.

The entire experiment, including dataset paths, model hyperparameters, training
settings, and evaluation metrics, is defined within a single configuration
dictionary, making it easy to reproduce and modify.

Workflow:
---------
1.  **Configuration Definition**: All experiment parameters are specified in the
    `config_dict`. This includes dataset details, model hyperparameters for RepeatNet,
    training settings (like learning rate and epochs), and evaluation metrics.
2.  **Environment Initialization**: The script initializes the RecBole environment,
    setting the random seed for reproducibility and configuring the logger.
3.  **Data Loading**: RecBole's data utilities are used to load the specified
    training, validation, and test files into its internal data structures.
4.  **Model Instantiation**: The RepeatNet model is created with the hyperparameters
    defined in the configuration.
5.  **Training and Validation**: A `Trainer` object manages the training loop. The
    model is trained on the training set, and its performance is periodically
    evaluated on the validation set. The best model checkpoint is saved based on
    the `valid_metric` (NDCG@20).
6.  **Final Evaluation**: After training is complete, the best saved model is
    loaded and its final performance is measured on the held-out test set.
7.  **Result Logging**: The best validation score and the final test results are
    logged for analysis.

Usage:
------
Run the script directly from the command line:
    python .\03-BENCHMARKS\RepeatNet\run_repeatnet.py

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- The dataset must be pre-processed and located in the `dataset/b2b_data/`
  directory. The files must be named `b2b_data.train.inter`,
  `b2b_data.validation.inter`, and `b2b_data.test.inter` as specified in the
  configuration.

Dependencies:
-------------
- Python 3.7+
- RecBole
- PyTorch
- logging
"""
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import RepeatNet
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging
import os 
import torch


_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

if __name__ == '__main__':

    config_dict = {

    'dataset': 'b2b_data',
    'data_path': os.path.join(os.path.dirname(__file__), '../../dataset/'),
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'TIME_FIELD': 'timestamp',
    'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
    'field_separator': "\t",
    'train_file': 'b2b_data.train.inter',
    'valid_file': 'b2b_data.valid.inter',
    'test_file': 'b2b_data.test.inter',
    'metrics': ["Recall", "NDCG"],
    'topk': [20],
    'valid_metric': 'NDCG@20',
    'model': 'RepeatNet',
    'embedding_size': 32,
    'hidden_size': 32,
    'joint_train': False,
    'dropout_prob': 0.2,
    'loss_type': 'CE',
    'train_neg_sample_args': None,
    'epochs': 1,
    'train_batch_size': 512,
    'eval_batch_size': 1024,
    'learning_rate': 0.01,
    'stopping_step': 1,
    'checkpoint_dir': os.path.join(os.path.dirname(__file__), 'saved_models/'),
    'reproducibility': True,
    'seed': 2020,
    'use_gpu': False,
    'use_tensorboard': False,
    'tensorboard_dir': os.path.join(os.path.dirname(__file__), 'tb_logs/'),
    'log_wandb': False,
}

config = Config(config_dict=config_dict)

init_seed(config['seed'], config['reproducibility'])
init_logger(config)
logger = logging.getLogger()
logger.info(config)

dataset = create_dataset(config)
logger.info(dataset)

train_data, valid_data, test_data = data_preparation(config, dataset)

config.final_config_dict['model_class'] = RepeatNet
model_class = config.final_config_dict['model_class']
model = model_class(config, train_data.dataset).to(config['device'])
logger.info(model)

trainer = Trainer(config, model)
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
test_result = trainer.evaluate(test_data)

logger.info('best valid result: {}'.format(best_valid_result))
logger.info('test result: {}'.format(test_result))