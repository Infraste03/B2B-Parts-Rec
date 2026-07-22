# 00 - Project Datasets

This directory contains all data files used and generated throughout this research project. It serves as the single source of truth for the dataset in its various stages of processing.

## Directory Structure

-   **/1_anonymous_dataset/**
    This subdirectory holds the raw, corrected, and anonymized dataset files provided by the industrial partner. The data is split into yearly `.csv` files (e.g., `2024_full_anonymus_v2_final.csv`). These files are the starting point for all subsequent analysis and preprocessing tasks.

-   **/2_preprocessed_for_graphs/**
    This subdirectory contains the graph-structured representation of the dataset. The data is organized into year-specific subfolders (e.g., `/graph_2024/`), each containing distinct node and edge lists in `.csv` format, ready to be used by Graph Neural Network (GNN) frameworks like PyTorch Geometric or DGL.