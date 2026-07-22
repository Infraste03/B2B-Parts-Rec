# 03-BENCHMARKS\NARM\run_narm.py

# =============================================================================
# AUTHOR INFORMATION 
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Email: francesca.stefano@unipr.it
# =============================================================================
"""NARM Recommendation Experiment Script

This script defines and executes a complete training and evaluation pipeline for the
NARM (Neural Attentive Session-based Recommendation Model) using the RecBole library.
NARM is an RNN-based sequential recommendation model that uses a GRU to encode the
sequence of user interactions and an attention mechanism to emphasize the most
important items in the user's history.

Key Characteristics of NARM:
---------------------------
- **RNN-Based Architecture**: It employs a Gated Recurrent Unit (GRU) to capture
  the sequential dependencies in user behavior.
- **Attention Mechanism**: After encoding the sequence, it uses an attention network
  to create a summary of the user's main purpose by assigning different weights to
  items in the sequence. This helps in capturing the most relevant historical
  items for the next prediction.
- **Session-based Focus**: Although its name suggests session-based recommendation,
  its architecture is highly effective for general sequential recommendation tasks
  where capturing user intent from recent actions is important.
- **Cross-Entropy Loss**: Typically trained with Cross-Entropy (CE) loss to predict
  the next item in the sequence.

Usage:
------
Run the script directly from the command line:
    python .\03-BENCHMARKS\FDSA\run_narm.py

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- The dataset must be pre-processed and located in the `dataset/b2b_data/`
  directory, with filenames correctly specified in the configuration.
"""

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import NARM
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging

if __name__ == '__main__':
    # A single dictionary to hold all experiment configurations for clarity and ease of modification.
    config_dict = {
        'data_path': '../../dataset/',
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        'field_separator': "\t",
        'train_file': 'b2b_data.train.inter',
        'valid_file': 'b2b_data.validation.inter',
        'test_file': 'b2b_data.test.inter',
        'metrics': ["Recall", "NDCG"],
        'topk': [20],
        'valid_metric': 'NDCG@20',
        'model': 'NARM',
        'embedding_size': 64,   # The dimensionality of the item embeddings.
        'hidden_size': 128,      # The dimension of the GRU hidden state.
        'n_layers': 1,           # The number of GRU layers.
        'dropout_prob': 0.5,     # The dropout rate for regularization.
        'loss_type': 'CE',       # Cross-Entropy is the standard loss function for NARM.
        'train_neg_sample_args': None,
        'epochs': 200,
        'train_batch_size': 2048,
        'eval_batch_size': 4096,
        'learning_rate': 0.001,
        'stopping_step': 10,
        'checkpoint_dir': 'saved_models/NARM/',
        'reproducibility': True,
        'seed': 2020,
        'use_gpu': True,
        'use_tensorboard': True,
        'tensor_dir': 'tb_logs/',
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

    model = NARM(config, train_data.dataset).to(config['device'])
    logger.info(model)

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    test_result = trainer.evaluate(test_data)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))