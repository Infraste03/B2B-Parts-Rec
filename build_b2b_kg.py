# build_b2b_kg.py
#*OKI

# =============================================================================
# AUTHOR INFORMATION AND SCRIPT DESCRIPTION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
#
# Description:
#   Builds the Knowledge Graph (KG) and associated link file required by certain
#   knowledge-aware recommendation models (like KGAT) from the cleaned B2B dataset.
#   It ensures perfect consistency with RecBole’s pre-generated .inter file and
#   runs an integrated verification check upon completion to validate the KG's
#   integrity and coverage.
# =============================================================================
"""Knowledge Graph Construction Script for KGAT

This script is a dedicated data preprocessing utility designed to build the two
key files required by knowledge graph-based recommendation models like KGAT:
1.  A Knowledge Graph (KG) file (`.kg`): A tab-separated file containing
    triplets in the format `head<tab>relation<tab>tail`.
2.  A Link File (`link.csv`): A file that maps original item IDs to their
    corresponding entity IDs within the knowledge graph.

Purpose:
--------
This script transforms a flat, tabular dataset with contextual features into a
structured knowledge graph. It extracts relationships between entities (e.g., an
item `fits_in_machine` a specific machine, a user is `located_in` a location)
and represents them as triplets. This structured KG can then be used by models
like KGAT to learn richer entity representations by incorporating relational data.

Workflow:
---------
1.  **Load Valid Entities**: Reads the master RecBole `.inter` file to get the
    exact sets of user and item IDs that are part of the benchmark experiments. This
    ensures that the KG only contains entities relevant to the recommendation task.
2.  **Load Full Dataset**: Reads the unified `full_dataset.csv` which contains all
    interactions and associated contextual features (side information).
3.  **Create Link File**: Generates `link.csv`, which in this simple case, maps
    each item ID to itself, as items are the primary entities in the KG.
4.  **Generate KG Triplets**: Iterates through the full dataset and creates triplets
    based on a predefined `RELATION_MAP`. For example, for a given row, it creates
    triplets like `(ITEM_ID, belongs_to_project, PROJECT_ID)`.
5.  **Save KG File**: Saves the unique, sorted triplets to the `.kg` file.
6.  **Integrated Verification**: After construction, it runs a series of automated
    checks to verify the integrity and coverage of the generated KG against the
    original interaction data, logging key statistics for each relation.

Usage:
------
The script is run directly from the command line. The input and output paths are
hardcoded as constants at the top of the file.

    python build_b2b_kg.py

Prerequisites:
--------------
- The master interaction file (`b2b_data.inter`) must exist in the specified path.
- The unified dataset with features (`full_dataset.csv`) must exist in the specified path.
"""

import pandas as pd
import os
import logging

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)

DATA_FOLDER = r'\B2B_ANALYSIS\dataset'

INTER_FILE = os.path.join(DATA_FOLDER, 'b2b_data', 'b2b_data.inter')
FULL_CSV = os.path.join(DATA_FOLDER, 'full_dataset.csv')

OUTPUT_LINK = os.path.join(DATA_FOLDER, 'link.csv')
OUTPUT_KG = os.path.join(DATA_FOLDER, 'b2b_data.kg')

# Define which side-info columns to use and their corresponding relation names
RELATION_MAP = {
    'MACHINE_ID': 'fits_in_machine',
    'PROJECT_ID': 'belongs_to_project',
    'PRICE_RANGE': 'has_price_level',
    'LOCATION': 'located_in'
}

# =============================================================================
# VERIFICATION FUNCTION (INTEGRATED)
# =============================================================================

def run_verification_checks(kg_df, inter_df, relation_map):
    """
    Run integrated verification checks on the generated KG against the .inter file.
    Logs statistics for each relation defined in relation_map.


    """
    logging.info("\n" + "="*80)
    logging.info("STARTING INTEGRATED KNOWLEDGE GRAPH VERIFICATION")
    logging.info("="*80)

    for relation_name in relation_map.values():
        if relation_name not in kg_df['relation'].values:
            logging.warning(f"Relation '{relation_name}' not found in KG. Skipping checks.")
            continue

        subset = kg_df[kg_df['relation'] == relation_name]
        total_triples = len(subset)

        logging.info(f"\n===== VERIFYING RELATION: {relation_name.upper()} =====")
        logging.info(f"Total triples: {total_triples:,}")

        # Determine if the relation is item-based or user-based
        entity_type = 'user' if relation_name == 'located_in' else 'item'
        head_field = 'user_id:token' if entity_type == 'user' else 'item_id:token'

        # Count involved entities and calculate coverage
        heads_with_relation = subset['head'].nunique()
        total_heads = inter_df[head_field].nunique()
        coverage = (heads_with_relation / total_heads * 100) if total_heads > 0 else 0

        logging.info(f"{entity_type.capitalize()} totals in .inter: {total_heads:,}")
        logging.info(f"{entity_type.capitalize()} with at least one relation '{relation_name}': {heads_with_relation:,}")
        logging.info(f"Relation coverage: {coverage:.2f}%")

        # Analyze tail value distribution
        logging.info("Tail value distribution (top 10 categories by value):")
        tail_counts = subset['tail'].value_counts().nlargest(10)
        for label, count in tail_counts.items():
            logging.info(f"  - '{label}': {count:,} triplets")

        # Find missing entities
        heads_with_relation_set = set(subset['head'])
        all_heads_set = set(inter_df[head_field])
        missing_heads_count = len(all_heads_set - heads_with_relation_set)
        logging.info(f"{entity_type.capitalize()} without relation '{relation_name}': {missing_heads_count:,}")

    logging.info("\n" + "="*80)
    logging.info("INTEGRATED VERIFICATION COMPLETE")
    logging.info("="*80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    logging.info("Starting Knowledge Graph construction (Phase B)...")

    # 1️ --- Load .inter file to collect valid users/items ---
    logging.info("Reading .inter file to collect valid user_id and item_id...")
    inter = pd.read_csv(INTER_FILE, sep='\t', engine='python')
    valid_users = set(inter['user_id:token'])
    valid_items = set(inter['item_id:token'])
    logging.info(f"Valid users: {len(valid_users)}, valid items: {len(valid_items)}")

    # 2️ --- Load full CSV with side info ---
    logging.info("Loading full_dataset.csv with side information...")
    df = pd.read_csv(FULL_CSV, sep=';', low_memory=False)

    # Basic cleaning: strip spaces and unify casing for text fields
    for col in RELATION_MAP.keys():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            if col == 'LOCATION':
                df[col] = df[col].str.title()

    # 3️ --- Create link.csv (item_id ↔ entity_id) ---
    logging.info("Creating link.csv file...")
    link_df = pd.DataFrame({'item_id': list(valid_items), 'entity_id': list(valid_items)})
    link_df.to_csv(OUTPUT_LINK, index=False)
    logging.info(f"Saved {len(link_df)} rows to link.csv")

    # 4️ --- Generate triplets for b2b_data.kg ---
    logging.info("Generating knowledge triplets...")
    triplets = set()

    for _, row in df.iterrows():
        item = str(row['ITEM_ID']).strip()
        user = str(row['CUSTOMER_ID']).strip()

        if item in valid_items:
            if pd.notna(row.get('MACHINE_ID')):
                triplets.add((item, RELATION_MAP['MACHINE_ID'], str(row['MACHINE_ID']).strip()))
            if pd.notna(row.get('PROJECT_ID')):
                triplets.add((item, RELATION_MAP['PROJECT_ID'], str(row['PROJECT_ID']).strip()))
            if pd.notna(row.get('PRICE_RANGE')):
                triplets.add((item, RELATION_MAP['PRICE_RANGE'], str(row['PRICE_RANGE']).strip()))

        if user in valid_users and pd.notna(row.get('LOCATION')):
            triplets.add((user, RELATION_MAP['LOCATION'], str(row['LOCATION']).strip()))

    logging.info(f"Total triplets generated (duplicates removed): {len(triplets):,}")

    # 5️ --- Save the .kg file ---
    logging.info("Saving triplets to b2b_data.kg ...")
    with open(OUTPUT_KG, 'w', encoding='utf-8') as f:
        for h, r, t in sorted(list(triplets)): # Sorted for deterministic output
            f.write(f"{h}\t{r}\t{t}\n")

    logging.info(f"Knowledge Graph successfully saved to {OUTPUT_KG}")

    # 6️ --- Summary statistics ---
    logging.info("Summary by relation:")
    rel_counts = {}
    for _, r, _ in triplets:
        rel_counts[r] = rel_counts.get(r, 0) + 1
    for r, c in sorted(rel_counts.items()):
        logging.info(f"  {r}: {c:,} triplets")

    logging.info("Phase B construction completed successfully!")

    # 7 --- Run Integrated Verification ---
    # Convert the generated triplets into a DataFrame for analysis
    kg_df = pd.DataFrame(list(triplets), columns=['head', 'relation', 'tail'])
    run_verification_checks(kg_df, inter, RELATION_MAP)



if __name__ == '__main__':
    main()


exit(0)
#TO TRASFORM!!!

file_path = r'B2B_ANALYSIS\dataset\b2b_data\b2b_data.link'
df = pd.read_csv(file_path, sep=',')
df.to_csv(file_path, sep='\t', index=False)