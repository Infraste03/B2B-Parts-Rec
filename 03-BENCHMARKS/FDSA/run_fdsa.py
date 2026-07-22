# 03-BENCHMARKS\FDSA\run_fdsa.py
# =============================================================================
# AUTHOR INFORMATION AND SCRIPT DESCRIPTION
# =============================================================================
# Author: Francesca Stefano
#Affiliation: University of Parma
#Email:francesca.stefano@unipr.it
# =============================================================================

"""FDSA Recommendation Experiment Script
This script defines and executes a complete training and evaluation pipeline for the
FDSA (Feature-aware Transformer-based Sequential Recommendation) model using the
RecBole library.

This model is a sophisticated sequential recommender that enhances the standard
Transformer architecture (like in SASRec) by incorporating item-specific side
information (features). In this configuration, it is set up to leverage both
`project_id` and `machine_id` as item features to improve recommendation accuracy.

Key Characteristics of FDSA:
---------------------------
- **Feature-Aware Model**: Unlike models that only use user-item interactions,
  FDSA integrates item features directly into its architecture. It learns
  embeddings for these features and uses them to enrich the representation of items.
- **Sequential Nature**: Like SASRec and BERT4Rec, it processes user interaction
  history as an ordered sequence, making it well-suited for next-item prediction.
- **Attention Mechanism**: It uses a self-attention mechanism, but the attention scores
  are influenced not only by the items themselves but also by their associated features.

Usage:
------
Run the script directly from the command line:
    python .\03-BENCHMARKS\FDSA\run_fdsa.py

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- The dataset must be pre-processed and located in the `dataset/` directory.
  Crucially, this script requires TWO data files:
  1. Interaction files (`.inter` suffix) containing `user_id`, `item_id`, `timestamp`.
  2. An item feature file (`.item` suffix) containing `item_id`, `project_id`, `machine_id`.
     The headers must be exactly as specified for RecBole to load them correctly.
"""


from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import FDSA
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging

if __name__ == '__main__':
    run_name = "FDSA_project_id_machine_id"

    config_dict = {
        'dataset': 'b2b_data',
        'data_path': '../../dataset/',
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        #  Crucial for FDSA: specifies which columns to load from which files.
        # 'inter' loads user-item interactions.
        # 'item' loads item features from the .item file.
        'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp'],
            'item':  ['item_id', 'project_id', 'machine_id']
        },
        'field_separator': '\t',
        'train_file': 'b2b_data.train.inter',
        'valid_file': 'b2b_data.valid.inter',
        'test_file': 'b2b_data.test.inter',
        'selected_features': ['project_id', 'machine_id'],
        'metrics': ['Recall', 'NDCG'],
        'topk': [20],
        'valid_metric': 'Recall@20',
        'eval_setting': 'LOO_full', # Leave-One-Out evaluation with full ranking
        'model': 'FDSA',
        'embedding_size': 64,
        'n_layers': 2,
        'n_heads': 2,
        'hidden_size': 64,
        'inner_size': 256,
        'hidden_dropout_prob': 0.5,
        'attn_dropout_prob': 0.5,
        'loss_type': 'BPR',
        'epochs': 1,
        'train_batch_size': 2048,
        'eval_batch_size': 4096,
        'learning_rate': 0.001,
        'stopping_step': 10,
        'checkpoint_dir': f'saved_models/b2b_data/{run_name}/',
        'reproducibility': True,
        'seed': 2024,
        'use_gpu': True,
        'save_dataset': False,
        'load_dataset': False,
    }


    config = Config(config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = logging.getLogger()
    logger.info(f"Avvio esperimento: {run_name}")
    logger.info(config)
    dataset = create_dataset(config)

    # Sanity checks
    logger.info(f"Loaded fields: {list(dataset.field2type.keys())}")
    assert 'project_id' in dataset.field2type, "ERROR: 'project_id' not found. Check the header of the .item file."
    assert 'machine_id' in dataset.field2type, "ERROR: 'machine_id' not found. Check the header of the .item file."
    logger.info("Sanity check passed: both item features were loaded correctly.")

    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = FDSA(config, train_data.dataset).to(config['device'])
    logger.info(model)
    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=True
    )
    test_result = trainer.evaluate(test_data, show_progress=True)
    logger.info(f'Miglior risultato su validation set: {best_valid_result}')
    logger.info(f'Risultato finale su test set: {test_result}')