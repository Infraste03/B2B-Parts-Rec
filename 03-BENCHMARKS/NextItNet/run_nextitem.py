#03-BENCHMARKS\NextItNet\run_nextitem.py

# =============================================================================
# AUTHOR INFORMATION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Contact: francesca.stefano@unipr.it
#
# Description:
# This script runs a quick local test for the NextItNet model on a small
# sample of the B2B dataset to ensure that the data loading and model
# instantiation work correctly before launching a full hyperparameter search.
# =============================================================================
"""NextItNet Recommendation Quick Test Script

This script defines and executes a quick, lightweight test run for the NextItNet
model using the RecBole library. It is configured with minimal hyperparameters and
runs for only a few epochs on a small data sample to verify the correctness of the
data loading pipeline and model implementation before committing to a full-scale
training or hyperparameter search.

Key Characteristics of NextItNet:
--------------------------------
- **Convolutional Neural Network (CNN) Based**: Unlike Transformer or RNN models,
  NextItNet uses a stack of dilated causal convolutional layers to capture
  sequential patterns.
- **Dilated Convolutions**: This technique allows the model to have a very large
  receptive field (i.e., see far back into the user's history) with a relatively
  small number of layers, making it computationally efficient.
- **Causal Convolutions**: Ensures that the prediction for a given timestep can only
  depend on previous timesteps, maintaining the auto-regressive nature required for
  next-item prediction.
- **Non-linearities**: Typically uses residual connections and layer normalization,
  similar to deep learning architectures like ResNet.

Usage:
------
Run the script directly from the command line for a quick sanity check:
    python .\03-BENCHMARKS\FDSA\run_nextitnet.py

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- A sample dataset (`b2b_data_sample`) must be available in the configured
  `data_path` to ensure a fast and lightweight test run.
"""

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import NextItNet
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging

if __name__ == '__main__':


    config_dict = {

        'dataset': 'b2b_data',
        'data_path': '../../dataset/',
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
        'model': 'NextItNet',
        'embedding_size': 16,     # Very small for a quick test.
        'n_layers': 2,            # Use 2 convolutional layers.
        'kernel_size': 3,         # The size of the convolutional filter.
        'dilations': [1, 2],      # The length of the dilations list must match n_layers.
        'dropout_prob': 0.5,
        'loss_type': 'CE',
        'train_neg_sample_args': None,
        'epochs': 2,          # Run for only 2 epochs to ensure a quick test.
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'learning_rate': 0.001,
        'stopping_step': 1,
        'checkpoint_dir': f'saved_models_debug/NextItNet/', # 'checkpoint_dir': f'saved_models_debug/{dataset_name}/NextItNet/',
        'reproducibility': True,
        'seed': 2020,
        'use_gpu': False,  # Run on CPU for local testing.
    }

    config = Config(config_dict=config_dict)

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = logging.getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = NextItNet(config, train_data.dataset).to(config['device'])
    logger.info(model)

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    test_result = trainer.evaluate(test_data)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))