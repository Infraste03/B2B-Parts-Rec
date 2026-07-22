#03-BENCHMARKS\GRU4Rec\run_gru4rec.py
# =============================================================================
# AUTHOR INFORMATION AND SCRIPT DESCRIPTION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Contact: francesca.stefano@unipr.it
# =============================================================================

"""GRU4Rec B2B Recommendation Experiment Script

This script defines and executes a complete training and evaluation pipeline for the
GRU4Rec (Gated Recurrent Unit for Recommendation) model using the RecBole library.
It is configured to run on a pre-processed B2B dataset, which has been split
into training, validation, and test sets.

The entire experiment, including dataset paths, model hyperparameters, training
settings, and evaluation metrics, is self-contained within this script.

Workflow:
---------
1.  **Configuration Definition**: All experiment parameters are specified in the
    `config_dict`. This includes dataset details, specific hyperparameters for GRU4Rec,
    training settings, and evaluation metrics.
2.  **Environment Initialization**: The script initializes the RecBole environment,
    setting the random seed for reproducibility and configuring a logger for detailed output.
3.  **Data Loading**: RecBole's data utilities load the specified training,
    validation, and test files. The paths are configured to align with RecBole's
    expected directory structure (`data_path/dataset_name/`).
4.  **Model Instantiation**: The GRU4Rec model is created with the hyperparameters
    defined in the configuration.
5.  **Training and Validation**: A `Trainer` object manages the training loop. The
    model is trained on the training set and its performance is evaluated on the
    validation set to find the best model checkpoint based on NDCG@20.
6.  **Final Evaluation**: After training, the best model is loaded and its final
    performance is measured on the held-out test set.
7.  **Result Logging**: The best validation score and the final test results are
    logged for analysis and comparison.

Usage:
------
Run the script directly from the command line:
    python .\03-BENCHMARKS\FDSA\run_gru4rec.py

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- The dataset must be pre-processed and located in the `dataset/b2b_data/`
  directory. The files must be named `train.csv`, `validation.csv`, and `test.csv`.

Dependencies:
-------------
- Python 3.7+
- RecBole
- PyTorch
- logging
"""
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec  # Importa specificamente GRU4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging

if __name__ == '__main__':
    # A single dictionary to hold all experiment configurations for GRU4Rec.
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
        # --- 2. Evaluation Parameters ---
        'metrics': ["Recall", "NDCG"],
        'topk': [20],
        'valid_metric': 'NDCG@20',  # Metric to monitor for early stopping and model saving.
        # --- 3. Model (GRU4Rec) Hyperparameters ---
        # Specific settings for the GRU4Rec architecture.
        'model': 'GRU4Rec',
        'embedding_size': 64,      # Dimensionality of the item embeddings.
        'hidden_size': 128,        # The size of the GRU's hidden state.
        'num_layers': 1,           # Number of stacked GRU layers.
        'dropout_prob': 0.3,       # Dropout probability for regularization.
        'loss_type': 'BPR',        # BPR (Bayesian Personalized Ranking) loss is standard for GRU4Rec.

        # --- 4. Training Parameters ---
        'epochs': 2,
        'train_batch_size': 128,
        'eval_batch_size': 256,
        'learning_rate': 0.001,
        'stopping_step': 10,  # Stop training if 'valid_metric' does not improve for 10 epochs.

        # --- 5. General Settings ---
        'checkpoint_dir': 'saved_models/GRU4Rec/',
        'reproducibility': True,
        'seed': 2024,
        'use_gpu': False,
        # --- 6. Experiment Tracking (Optional) ---
        'use_tensorboard': True,
        'tensorboard_dir': 'tb_logs/GRU4Rec/',  # Log to a model-specific subdirectory.
        'log_wandb': False,
    }

    # --- Experiment Execution ---

    # 1. Initialize the configuration object.
    config = Config(config_dict=config_dict) # Pass model name explicitly

    # 2. Initialize environment: seed for reproducibility and logger for output.
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = logging.getLogger()
    logger.info("--- GRU4Rec Experiment Configuration ---")
    logger.info(config)
    logger.info("--------------------------------------\n")

    # 3. Load the dataset according to the specified paths and settings.
    dataset = create_dataset(config)
    logger.info("--- Dataset Information ---")
    logger.info(dataset)
    logger.info("-------------------------\n")

    # 4. Prepare data loaders for training, validation, and testing.
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 5. Instantiate the GRU4Rec model.
    model = GRU4Rec(config, train_data.dataset).to(config['device'])
    logger.info("--- Model Architecture ---")
    logger.info(model)
    logger.info("------------------------\n")

    # 6. Initialize the Trainer.
    trainer = Trainer(config, model)

    # 7. Start the training and validation loop.
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # 8. Evaluate the best model on the test set.
    test_result = trainer.evaluate(test_data)

    # 9. Log the final results.
    logger.info('\n-- GRU4Rec Final Results --')
    logger.info('Best validation result: {}'.format(best_valid_result))
    logger.info('Test result: {}'.format(test_result))
    logger.info('---------------------------')