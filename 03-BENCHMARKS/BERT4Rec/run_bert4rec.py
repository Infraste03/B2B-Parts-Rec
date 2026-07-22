# 03-BENCHMARKS\BERT4Rec\run_bert4rec.py
#fast local test

# =============================================================================
# AUTHOR INFORMATION AND SCRIPT DESCRIPTION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Contact: francesca.stefano@unipr.it
# =============================================================================

"""BERT4Rec B2B Recommendation Experiment Script

This script defines and executes a complete training and evaluation pipeline for the
BERT4Rec (Bidirectional Encoder Representations from Transformers for Sequential Recommendation)
model using the RecBole library.

It is configured to run on the same pre-processed B2B dataset as the other
benchmark models. The entire experiment configuration is self-contained within this script,
ensuring clarity and reproducibility.

Key Differences from SASRec:
---------------------------
- **Bidirectional Training**: Unlike the auto-regressive (left-to-right) nature of
  SASRec, BERT4Rec learns from the entire sequence context (both past and future items)
  by masking some items and training the model to predict them based on the surrounding items.
  This is analogous to the Masked Language Model (MLM) task in the original BERT paper.
- **Masking**: A specific hyperparameter, `mask_ratio`, defines the percentage
  of items to randomly mask in each sequence during the training process.
- **Loss Function**: BERT4Rec typically uses Cross-Entropy (CE) loss to predict the
  identity of the masked items from the entire item catalog, rather than the pairwise
  BPR loss often used in SASRec.

Usage:
------
Run the script directly from the command line:
    python .\03-BENCHMARKS\BERT4Rec\run_bert4rec.py

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- The dataset must be pre-processed and located in the `dataset/b2b_data/`
  directory, with filenames correctly specified in the configuration.
"""
# =============================================================================
# AUTHOR INFORMATION
# =============================================================================
__author__ = "Francesca Stefano"
__affiliation__ = "PhD Student in Information Technology, University of Parma"
__email__ = "francesca.stefano@unipr.it"
# =============================================================================

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import BERT4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging


if __name__ == '__main__':
    # A single dictionary to hold all experiment configurations.
    config_dict = {
        # --- 1. Dataset Parameters (Consistent across benchmarks) ---
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
        # --- 2. Evaluation Parameters (Consistent across benchmarks) ---
        'metrics': ["Recall", "NDCG"],
        'topk': [20],
        'valid_metric': 'NDCG@20',
         # --- 3. Model (BERT4Rec) Hyperparameters ---
        'model': 'BERT4Rec',
        # Core Transformer architecture parameters, similar to SASRec.
        'embedding_size': 32,
        'n_layers': 1,
        'n_heads': 1,
        'hidden_size': 32,
        'inner_size': 128,
        'hidden_dropout_prob': 0.5,
        'attn_dropout_prob': 0.5,
        'hidden_act': 'gelu',
        'layer_norm_eps': 1e-12,
        'initializer_range': 0.02,
        #Crucial hyperparameters specific to BERT4Rec's training methodology.
        'mask_ratio': 0.2,      # The proportion of items to mask in each input sequence for the MLM training task.
        'loss_type': 'CE',      # Cross-Entropy loss is the standard for predicting the masked items from the full vocabulary
        'train_neg_sample_args': None, # Explicitly disable negative sampling, as it's not used with CE loss.

        # --- 4. Training Parameters ---
        'epochs': 200,
        'train_batch_size': 128,
        'eval_batch_size': 256,
        'learning_rate': 0.01,
        'stopping_step': 10,

        # --- 5. General Settings (Consistent across benchmarks) ---
        'checkpoint_dir': 'saved_models/',
        'reproducibility': True,
        'seed': 2020,
        'use_gpu': False,

        # --- 6. Experiment Tracking (Consistent across benchmarks) ---
        'use_tensorboard': True,
        'tensorboard_dir': 'tb_logs/',
        'log_wandb': False,
    }

# --- Experiment Execution ---
# 1. Initialize the configuration object from the dictionary.
config = Config(config_dict=config_dict)
# 2. Initialize the environment: set the random seed for reproducibility and configure the logger.
init_seed(config['seed'], config['reproducibility'])
init_logger(config)
logger = logging.getLogger()
logger.info(config)


# 3. Load the dataset using RecBole's utilities.
dataset = create_dataset(config)
logger.info(dataset)

# 4. Prepare the data splits into training, validation, and test data loaders.
train_data, valid_data, test_data = data_preparation(config, dataset)

# 5. Instantiate the BERT4Rec model with the specified hyperparameters.
model = BERT4Rec(config, train_data.dataset).to(config['device'])
logger.info(model)

trainer = Trainer(config, model)
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
test_result = trainer.evaluate(test_data)

logger.info('best valid result: {}'.format(best_valid_result))
logger.info('test result: {}'.format(test_result))