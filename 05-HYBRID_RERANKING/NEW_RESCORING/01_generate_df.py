"""
===================================================================================================
TWO-STAGE RECOMMENDER PIPELINE: CANDIDATE GENERATION & FEATURE ENGINEERING
===================================================================================================
This module executes Stage 1 (Candidate Generation) using a pre-trained SASRec model.
It extracts the Top-K candidates for each event and enriches them with a highly specialized
Feature Space tailored for B2B industrial domains, preventing data leakage by dynamically
updating user states exclusively up to time t-1.

--- FEATURE SPACE DICTIONARY (Categorized) ---

1. SEQUENTIAL SIGNALS (Latent Representations via SASRec)
   - sasrec_score : Raw logit score emitted by the Stage 1 model. Serves as the primary
                    sequential baseline score for integration.
   - sasrec_rank  : The topological position (1 to K) predicted by SASRec. Crucial for
                    capturing the non-linear decay of relevance.

2. ECONOMIC & BUDGET CONSTRAINTS
   - user_avg_price : Historical rolling average price of user's past purchases (leakage-safe).
   - item_proxy_price : Median price of the candidate item extrapolated from the TRAIN set.
   - price_diff   : Absolute difference |item_proxy_price - user_avg_price|.
   - price_ratio  : Ratio mapping compatibility between item cost and user's budget tolerance.

3. DOMAIN COMPATIBILITY & TECHNICAL RELATIONS (Strict B2B Constraints)
   - is_compatible_with_current_machine : Binary flag (1/NaN). Defines if the item has ever been
                                          mounted on the current machine in the global train set.
   - item_machine_support : Integer count of co-occurrences (Item <-> Machine) in history.
   - is_compatible_with_current_prodmodel : Binary flag for Product Model compatibility.
   - item_prodmodel_support : Integer count of co-occurrences (Item <-> Product Model).
   - is_same_product_model_as_last : Validates if the candidate belongs to the exact technical
                                     family of the user's strictly preceding purchase.

4. RECENCY & FREQUENCY (Temporal Dynamics)
   - days_since_last_purchase : Time delta since the user's last global interaction.
   - days_since_last_purchase_same_item : Time delta since the user last interacted with the 
                                          SPECIFIC candidate item (captures recurring supply needs).
   - user_item_freq_hist : Cumulative frequency of the user purchasing the specific candidate item.

5. CONTEXTUAL NOVELTY & ROBUSTNESS
   - item_popularity : Global baseline popularity of the item extracted from the Train set.
   - item_n_machines_train : Technical versatility of the item (how many distinct machines use it).
   - is_new_machine : Flag tracking if the current event involves a machine never witnessed in Train.
===================================================================================================
"""

import os
import ast
import time
import logging
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tqdm
import torch

from recbole.config import Config
from recbole.utils import init_seed
from recbole.data import create_dataset
from recbole.data.dataset import Dataset
from recbole.data.interaction import Interaction
from recbole.model.sequential_recommender import SASRec

warnings.filterwarnings("ignore")


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_folder='logs'):
    os.makedirs(log_folder, exist_ok=True)
    log_filename = os.path.join(
        log_folder,
        f'create_training_df_{time.strftime("%Y%m%d-%H%M%S")}.log'
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()
        ]
    )

# -----------------------------
# UTILS
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return float(a) / float(b + eps)


def prev_month_period(ts: pd.Timestamp) -> pd.Period:
    return ts.to_period("M") - 1


def sanity_check_id_alignment(target_item_internal: int, scores_np, item_token_to_id: dict, item_id2token_list: list, target_item_token: str | None = None,verbose: bool = False):
    """
    Validates the alignment and consistency of item IDs, tokens, and scores.
    Performs multiple assertions to ensure:
    - scores_np and item_id2token_list have matching lengths
    - target_item_internal index is within valid bounds
    - token-to-ID mapping is bidirectionally consistent (round-trip validation)
    - optional target_item_token matches the resolved token
    - [PAD] token ID follows RecBole standard (ID = 0)
    Args:
        target_item_internal (int): The internal ID index to validate.
        scores_np: Array/sequence of scores with length matching item_id2token_list.
        item_token_to_id (dict): Mapping from token strings to integer IDs.
        item_id2token_list (list): Ordered list mapping internal IDs to token strings.
        target_item_token (str | None, optional): Expected token string to verify against resolved token. Defaults to None.
        verbose (bool, optional): If True, logs validation success details. Defaults to False.
    Raises:
        AssertionError: If any alignment check fails, with descriptive [MISALIGN] error messages.
    Returns:
        None
    """

    assert len(scores_np) == len(item_id2token_list), (
        f"[MISALIGN] len(scores_np)={len(scores_np)} "
        f"!= len(item_id2token_list)={len(item_id2token_list)}"
    )

    assert 0 <= target_item_internal < len(scores_np), (
        f"[MISALIGN] target_item_internal={target_item_internal} "
        f"out of bounds for scores_np len={len(scores_np)}"
    )
    token_from_id = item_id2token_list[target_item_internal]
    assert token_from_id in item_token_to_id, (
        f"[MISALIGN] token '{token_from_id}' not found in item_token_to_id"
    )
    id_roundtrip = int(item_token_to_id[token_from_id])

    assert id_roundtrip == int(target_item_internal), (
        f"[MISALIGN] round-trip failed: "
        f"id {target_item_internal} -> token '{token_from_id}' -> id {id_roundtrip}"
    )
    if target_item_token is not None:
        assert str(target_item_token) == str(token_from_id), (
            f"[MISALIGN] row token '{target_item_token}' "
            f"!= token_from_id '{token_from_id}'"
        )
    if "[PAD]" in item_token_to_id:
        pad_id = int(item_token_to_id["[PAD]"])
        assert pad_id == 0, (
            f"[MISALIGN] PAD id is {pad_id}, expected 0 (RecBole standard)"
        )
    if verbose:
        logging.info(
            f"[SANITY-ID] OK | internal_id={target_item_internal} "
            f"| token='{token_from_id}' "
            f"| score={float(scores_np[target_item_internal])}"
        )


def sanity_check_row_token_matches_internal_id(
    target_item_internal: int,
    target_item_token: str | None,
    item_id2token_list: list,
    verbose: bool = False
):
    """
    Validates that a token from a data row matches its corresponding token in the vocabulary.
    This function performs sanity checks to ensure data integrity by verifying that the token
    associated with an item ID in the input data matches the token stored in the vocabulary list.
    Args:
        target_item_internal (int): The internal ID/index of the target item in the vocabulary.
        target_item_token (str | None): The token string from the data row. If None, the check is skipped.
        item_id2token_list (list): A list mapping internal IDs to their corresponding token strings.
        verbose (bool, optional): If True, prints a success message when tokens match. Defaults to False.
    Raises:
        AssertionError: If target_item_internal is out of bounds in item_id2token_list.
        AssertionError: If the row token does not match the vocabulary token for the given ID.
    Returns:
        None"""

    if target_item_token is None:
        return

    assert 0 <= target_item_internal < len(item_id2token_list), (
        f"[MISALIGN] target_item_internal={target_item_internal} out of bounds "
        f"for item_id2token_list len={len(item_id2token_list)}"
    )

    token_from_vocab = item_id2token_list[target_item_internal]

    assert str(target_item_token) == str(token_from_vocab), (
        f"[MISALIGN] Row token != vocab token: "
        f"row='{target_item_token}' vs vocab='{token_from_vocab}' (id={target_item_internal})"
    )

    if verbose:
        print(f"[OK] Row token matches vocab: id={target_item_internal} token='{token_from_vocab}'")

# -----------------------------
# SASREC SCORING
# -----------------------------

@torch.no_grad()
def sasrec_full_scores_from_sequence(model: SASRec, dataset, item_seq: torch.Tensor, item_seq_len: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    Compute full-sort scores over ALL items given a (1, seq_len) item sequence.

    Returns:
        scores: (n_items,) float tensor (logits)
    """
    model.eval()
    inter = {
        model.ITEM_SEQ: item_seq.to(device),
        model.ITEM_SEQ_LEN: item_seq_len.to(device),
    }

    inter = Interaction(inter).to(device)
    scores = model.full_sort_predict(inter)
    scores = scores.squeeze(0).detach().cpu()
    return scores



def build_sasrec_topk_candidates(
    model: SASRec,
    dataset,
    hist_items_internal: List[int],
    max_seq_len: int,
    device: torch.device,
    target_item_internal: int,
    k: int = 200,
    item_token_to_id: dict = None,
    item_id2token_list: list = None,
    target_item_token: str | None = None,
    verbose = np.random.rand() < 0.001,
) -> Tuple[np.ndarray, np.ndarray, float]: 
    """
    Compute SASRec full-sort over catalog, then take top-k items.
    """
    # build padded sequence
    seq = hist_items_internal[-max_seq_len:]
    seq_len = len(seq)

    if seq_len == 0:
        # no history => cannot produce meaningful SASRec scores
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32), 0.0

    pad_len = max_seq_len - seq_len
    padded = seq + [0] * pad_len
    #assert len(padded) == max_seq_len, f"Sequence length {len(padded)} != max_seq_len {max_seq_len}"
    item_seq = torch.tensor([padded], dtype=torch.long)
    item_seq_len = torch.tensor([seq_len], dtype=torch.long)
    all_scores = sasrec_full_scores_from_sequence(model, dataset, item_seq, item_seq_len, device)

    # get top-k (ignore padding item 0)
    scores_np = all_scores.numpy()
    scores_np[0] = -np.inf

    if item_token_to_id is not None and item_id2token_list is not None:
        sanity_check_id_alignment(
            target_item_internal=target_item_internal,
            scores_np=scores_np,
            item_token_to_id=item_token_to_id,
            item_id2token_list=item_id2token_list,
            target_item_token=target_item_token,
            verbose=verbose
        )

    target_score = float(scores_np[target_item_internal])
    if np.random.rand() < 0.001:
        logging.info(
            f"[SANITY-TARGET-SCORE] target_id={target_item_internal} "
            f"| raw_score={target_score:.4f}"
        )

    #top_idx = np.argpartition(-scores_np, k)[:k]PRIMA ERA COSì
    top_idx = np.argpartition(-scores_np, k-1)[:k]
    top_idx = top_idx[np.argsort(-scores_np[top_idx])]

    return top_idx.astype(np.int64), scores_np[top_idx].astype(np.float32), target_score


def replace_inject_target_into_topk(
    cand_items: np.ndarray,
    cand_scores: np.ndarray,
    target_item_internal: int,
    target_score: float,
    k: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    REPLACE injection (keep always exactly k candidates):
    - If target already in top-k: nothing to do
    - Else: drop the last (worst) item and insert target, then re-sort by score desc
    """
    if cand_items.size == 0:
        # caller handles empty history separately
        return cand_items, cand_scores
    if target_item_internal in set(cand_items.tolist()):
        return cand_items, cand_scores

    cand_items = cand_items[:k].copy()
    cand_scores = cand_scores[:k].copy()
    cand_items[-1] = int(target_item_internal)
    cand_scores[-1] = float(target_score)
    order = np.argsort(-cand_scores)
    cand_items = cand_items[order]
    cand_scores = cand_scores[order]
    return cand_items, cand_scores


# -----------------------------
# PROCESS SPLIT
# -----------------------------
def process_split(
    split_name: str,
    split_df: pd.DataFrame,
    out_dir: str,
    model: SASRec,
    dataset,
    device: torch.device,
    max_seq_len: int,
    k: int,
    ITEM_POPULARITY_DICT: dict,     # item_internal -> count in train
    ITEM_MEDIAN_PRICE_DICT: dict,   # item_internal -> median price in train
    global_price_median: float,
    USER_LOCATION_DICT: dict,       # user_internal -> location (customer-level)
    #USER_AVG_PRICE_DICT: dict,
    ITEM_MACHINE_SET: set,          # (item_internal, machine_id) pairs seen in TRAIN
    ITEM_PRODMODEL_SET: set,        # (item_internal, product_model_id) pairs seen in TRAIN
    ITEM_MACHINE_SUPPORT: dict,     # (item_internal, machine_id) -> count in TRAIN
    ITEM_PRODMODEL_SUPPORT: dict,   # (item_internal, product_model_id) -> count in TRAIN
    ITEM_N_MACHINES: dict,          # item_internal -> n_machines in train
    TRAIN_MACHINES_SET: set,        # machine_id values seen in TRAIN
    ITEM_TO_PRODMODEL_DICT: dict,
    ITEM_DESCRIPTION_DICT: dict,
    user_state: dict,
    force_positive: bool = True,
    flush_rows: int = 50000,
    item_id2token_list: Optional[List[str]] = None,
    item_token_to_id: Optional[dict] = None
) -> str:
    """
    Build rescoring dataset for one split.
    Event definition: (user_id_internal, global_user_event_idx)
    Single-target per event.
    """
    ensure_dir(out_dir)
    SPECIAL_COLD_USER = {11,27,64}
    user_state.setdefault("global_user_event_idx", {})
    if ITEM_DESCRIPTION_DICT and len(ITEM_DESCRIPTION_DICT) > 0:
        _any_emb = next(iter(ITEM_DESCRIPTION_DICT.values()))
        EMBEDDING_DIM = len(_any_emb)
    else:
        raise ValueError("ITEM_DESCRIPTION_DICT is empty: cannot infer embedding_dim.")

    DEFAULT_EMB = [0.0] * EMBEDDING_DIM
    DESC_COLS = [f"item_desc_emb_{i}" for i in range(EMBEDDING_DIM)]

    #out_path = os.path.join(out_dir, f"{split_name}_part_{int(time.time())}.parquet")

    rows_buffer = []
    rows_buffer_count = 0
    total_events = 0
    miss_groups = 0
    forced_positive_count = 0

    for uid_internal, ugroup in tqdm.tqdm(split_df.groupby("user_id_internal", sort=False), desc=f"{split_name} Users"):
        logging.info(f"[{split_name}] Processing user_id_internal={uid_internal}, eventi={len(ugroup)}")

        if "row_id" in ugroup.columns:
            ugroup = ugroup.sort_values(["timestamp", "row_id"]).reset_index(drop=True)
        else:
            ugroup = ugroup.sort_values(["timestamp"]).reset_index(drop=True)

        # user-level constant location (customer-level)
        user_location = USER_LOCATION_DICT.get(uid_internal, None)
        running_sum = float(user_state["running_sum_price"].get(uid_internal, 0.0))
        running_cnt = int(user_state["running_count"].get(uid_internal, 0))

        # history trackers (shared across splits via user_state)
        hist_items = user_state["hist_items"].get(uid_internal, [])
        hist_ts = user_state["hist_ts"].get(uid_internal, [])
        hist_item_last_ts = user_state["hist_item_last_ts"].get(uid_internal, {})   # item -> last ts
        hist_item_freq = user_state["hist_item_freq"].get(uid_internal, {})         # item -> freq
        last_prodmodel = user_state["last_product_model"].get(uid_internal, None)

        for local_idx in range(len(ugroup)):
            # current event
            current_ts = pd.Timestamp(ugroup.iloc[local_idx]["timestamp"])
            target_item_internal = int(ugroup.iloc[local_idx]["item_id_internal"])
            target_item_token = ugroup.iloc[local_idx].get("item_id", None)


            if item_id2token_list is not None and np.random.rand() < 0.001:
                sanity_check_row_token_matches_internal_id(
                    target_item_internal=target_item_internal,
                    target_item_token=target_item_token,
                    item_id2token_list=item_id2token_list,
                    verbose=False
                )
            #current_price = float(ugroup.iloc[local_idx]["price"])
            current_price = float(ugroup.iloc[local_idx].get("price", np.nan))
            if np.isnan(current_price):
                current_price = global_price_median
            user_event_idx = int(ugroup.iloc[local_idx].get("user_event_idx", local_idx))
            global_user_event_idx = int(user_state["global_user_event_idx"].get(uid_internal, 0))

            # event context from the *actual* row (single-item event)
            event_machine_id = ugroup.iloc[local_idx].get("machine_id", None)
            event_product_model_id = ugroup.iloc[local_idx].get("product_model_id", None)

            # NEW: event-level flag -> machine never seen in TRAIN (leakage-safe)
            if pd.isna(event_machine_id) or (event_machine_id is None):
                is_new_machine = 1
            else:
                is_new_machine = 0 if event_machine_id in TRAIN_MACHINES_SET else 1

            # compute recency features
            if len(hist_ts) == 0:
                days_since_last_purchase = -1
            else:
                days_since_last_purchase = int((current_ts - hist_ts[-1]).days)

            used_fallback = False
            if len(hist_items) == 0:
                if split_name =="train" and uid_internal in  SPECIAL_COLD_USER:
                    logging.info(f"[COLD-START] user_id_internal={uid_internal} has no history, using popularity fallback for train split.")

                    top_pop_items = sorted(
                        ITEM_POPULARITY_DICT.items(),
                        key=lambda x: -x[1]
                    )[:k]
                    cand_items_internal = np.array([it for it, _ in top_pop_items], dtype=np.int64)

                    cand_scores = np.linspace(k, 1, num=len(cand_items_internal)).astype(np.float32)

                    target_score = float(k + 1)

                    if target_item_internal not in cand_items_internal:
                        cand_items_internal[-1] = target_item_internal
                        cand_scores[-1] = target_score
                        # riordina
                    order = np.argsort(-cand_scores)
                    cand_items_internal = cand_items_internal[order]
                    cand_scores = cand_scores[order]
                    used_fallback = True

                    assert target_item_internal in set(cand_items_internal.tolist()), \
                        "[FALLBACK INJECTION ERROR] target not present after popularity fallback"

                else:
                    hist_items.append(target_item_internal)
                    hist_ts.append(current_ts)
                    hist_item_last_ts[target_item_internal] = current_ts
                    hist_item_freq[target_item_internal] = hist_item_freq.get(target_item_internal, 0) + 1
                    last_prodmodel = event_product_model_id if not pd.isna(event_product_model_id) else last_prodmodel

                    user_state["global_user_event_idx"][uid_internal] = global_user_event_idx + 1
                    # update running price stats also for the first event
                    running_sum += current_price
                    running_cnt += 1
                    continue

            if not used_fallback:
                cand_items_internal, cand_scores, target_score = build_sasrec_topk_candidates(
                    model=model,
                    dataset=dataset,
                    hist_items_internal=hist_items,
                    max_seq_len=max_seq_len,
                    device=device,
                    target_item_internal=target_item_internal,
                    k=k,
                    item_token_to_id=item_token_to_id,
                    item_id2token_list=item_id2token_list,
                    target_item_token=target_item_token,
                )
            else:
                pass

            if np.random.rand() < 0.001:
                logging.info(
                    f"[SASREC-STATS] "
                    f"min={cand_scores.min():.4f} "
                    f"max={cand_scores.max():.4f} "
                    f"mean={cand_scores.mean():.4f}"
                )
            if target_item_internal not in set(cand_items_internal.tolist()):
                miss_groups += 1
                if force_positive:

                    # target_score = float(cand_scores[-1]) + 1e-6 #
                    cand_items_internal, cand_scores = replace_inject_target_into_topk(
                        cand_items_internal, cand_scores, target_item_internal, target_score, k=k 
                    )
                    forced_positive_count += 1
                if force_positive:
                    assert target_item_internal in cand_items_internal, \
                        "[INJECTION ERROR] target not present after injection"

                    if np.random.rand() < 0.001:
                        logging.info(
                            f"[INJECTION-OK] "
                            f"user={uid_internal} "
                            f"event={global_user_event_idx} "
                            f"target={target_item_internal}"
                        )
            # FINAL candidate list for this event (after possible injection)
            cand_items = cand_items_internal.astype(np.int64, copy=False)

            # leakage-proof: media storica fino a t-1
            if running_cnt > 0:
                user_avg_price = running_sum / running_cnt
            else:
                user_avg_price = global_price_median

            cand_df = pd.DataFrame({
                "split": split_name,
                "user_id_internal": uid_internal,
                "user_event_idx": user_event_idx,
                "global_user_event_idx": global_user_event_idx,
                "target_item_id_token": ugroup.iloc[local_idx].get("item_id", None),
                "target_item_id_internal": target_item_internal,
                "candidate_item_id_internal": cand_items,
                "sasrec_score": cand_scores,
                "sasrec_rank": np.arange(1, len(cand_items) + 1, dtype=np.int32),

                # user/event-level constants (repeat on 200 rows)
                "user_location": user_location,
                "event_machine_id": event_machine_id,
                "is_new_machine": int(is_new_machine),

                "user_avg_price": user_avg_price,
                "days_since_last_purchase": days_since_last_purchase,
            })

            # target label
            cand_df["target_boost"] = (cand_df["candidate_item_id_internal"] == target_item_internal).astype(np.float32)

            if item_id2token_list is not None:
                cand_df['candidate_item_id_token'] = cand_df['candidate_item_id_internal'].apply(
                    lambda x: item_id2token_list[int(x)] if int(x) < len(item_id2token_list) else "ID_FUORI_MAPPA"
                )
            # price proxy + derived
            cand_df["item_price_proxy"] = cand_df["candidate_item_id_internal"].map(ITEM_MEDIAN_PRICE_DICT).fillna(global_price_median)
            # ============================================================
            # SANITY CHECK — item_price_proxy propagation (throttled)
            # ============================================================
            if split_name == "train" and np.random.rand() < 0.001:
                # 1) item_price_proxy must be finite
                bad_proxy = cand_df[~np.isfinite(cand_df["item_price_proxy"].astype(float))]
                assert len(bad_proxy) == 0, "[PRICE] item_price_proxy has non-finite values"

                # 2) how many candidates used fallback (= global_price_median)
                fallback_mask = np.isclose(cand_df["item_price_proxy"].astype(float), float(global_price_median), rtol=0, atol=0)
                fallback_rate = float(fallback_mask.mean())
                logging.info(f"[PRICE][{split_name}] fallback_rate={fallback_rate:.1%} | uid={uid_internal} global_evt={global_user_event_idx}")

                # 3) if candidate item exists in dict, it should NOT fallback (unless its median equals global median)
                cand_items_list = cand_df["candidate_item_id_internal"].astype(int).tolist()
                exists_in_dict = np.array([it in ITEM_MEDIAN_PRICE_DICT for it in cand_items_list], dtype=bool)

                suspicious = cand_df[exists_in_dict & fallback_mask]
                # This can be legit if item median equals global median, so don't assert hard; log examples
                if len(suspicious) > 0:
                    sample_ids = suspicious["candidate_item_id_internal"].astype(int).head(10).tolist()

                    debug_rows = []
                    for it in sample_ids:
                        v = ITEM_MEDIAN_PRICE_DICT.get(it, None)

                        if it not in ITEM_MEDIAN_PRICE_DICT:
                            reason = "MISSING_KEY"
                        elif v is None:
                            reason = "DICT_NONE"
                        elif isinstance(v, float) and np.isnan(v):
                            reason = "DICT_NAN_MEDIAN"
                        elif np.isclose(float(v), float(global_price_median), rtol=0, atol=0):
                            reason = "DICT_EQUALS_GLOBAL_MEDIAN"
                        else:
                            reason = "UNEXPECTED_FALLBACK"

                        debug_rows.append({
                            "item": it,
                            "dict_median": v,
                            "global_median": global_price_median,
                            "reason": reason
                        })

                    dbg = pd.DataFrame(debug_rows)
                    logging.warning(
                        f"[PRICE] fallback used but key exists. Breakdown:\n{dbg.to_string(index=False)}"
                    )

            cand_df["price_diff"] = (cand_df["item_price_proxy"] - cand_df["user_avg_price"]).abs()
            cand_df["price_ratio"] = cand_df["item_price_proxy"] / (cand_df["user_avg_price"] + 1e-9)

            # popularity
            cand_df["item_popularity"] = cand_df["candidate_item_id_internal"].map(ITEM_POPULARITY_DICT).fillna(0).astype(np.int64)

            # user-item history features (freq)
            cand_df["user_item_freq_hist"] = cand_df["candidate_item_id_internal"].map(hist_item_freq).fillna(0).astype(np.int64)


            cand_df["is_same_product_model_as_last"] = 0
            if last_prodmodel is not None and not pd.isna(last_prodmodel):
                lp = last_prodmodel
                cand_df["is_same_product_model_as_last"] = np.fromiter(
                    (1 if lp in ITEM_TO_PRODMODEL_DICT.get(int(it), set()) else 0 for it in cand_items),
                    dtype=np.int8,
                    count=len(cand_items)
                ).astype(np.int32)

            if pd.isna(event_machine_id) or (event_machine_id is None):
                # no machine context => unknown, not incompatible
                cand_df["is_compatible_with_current_machine"] = np.nan
                cand_df["item_machine_support"] = 0
            else:
                em = event_machine_id
                cand_df["is_compatible_with_current_machine"] = cand_df["candidate_item_id_internal"].map(
                    lambda it: 1.0 if (int(it), em) in ITEM_MACHINE_SET else np.nan
                ).astype(np.float32)
                cand_df["item_machine_support"] = cand_df["candidate_item_id_internal"].map(
                    lambda it: int(ITEM_MACHINE_SUPPORT.get((int(it), em), 0))
                ).astype(np.int32)

            # Product model compatibility vs current event product model
            # Prodmodel compatibility: 1 = observed support, NaN = unknown
            if pd.isna(event_product_model_id) or (event_product_model_id is None):
                cand_df["is_compatible_with_current_prodmodel"] = np.nan
                cand_df["item_prodmodel_support"] = 0
            else:
                pm = event_product_model_id
                cand_df["is_compatible_with_current_prodmodel"] = cand_df["candidate_item_id_internal"].map(
                    lambda it: 1.0 if (int(it), pm) in ITEM_PRODMODEL_SET else np.nan
                ).astype(np.float32)
                cand_df["item_prodmodel_support"] = cand_df["candidate_item_id_internal"].map(
                    lambda it: int(ITEM_PRODMODEL_SUPPORT.get((int(it), pm), 0))
                ).astype(np.int32)

            # generic candidate feature: how many machines item had in train
            cand_df["item_n_machines_train"] = cand_df["candidate_item_id_internal"].map(ITEM_N_MACHINES).fillna(0).astype(np.int64)

            # per-candidate same-item recency (relative to user history)
            def _days_since_user_bought_item(it: int) -> int:
                last_ts = hist_item_last_ts.get(int(it), None)
                if last_ts is None:
                    return -1
                return int((current_ts - last_ts).days)

            cand_df["days_since_last_purchase_same_item"] = [
                _days_since_user_bought_item(int(it)) for it in cand_df["candidate_item_id_internal"].values
            ]

            #cand_items = cand_df["candidate_item_id_internal"].to_numpy(dtype=np.int64)
            # build list-of-lists (fast Python loop, avoids pandas apply)
            desc_list = []
            for it in cand_items:
                v = ITEM_DESCRIPTION_DICT.get(int(it), None)
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) == EMBEDDING_DIM:
                    desc_list.append(v)
                else:
                    desc_list.append(DEFAULT_EMB)

            item_desc_df = pd.DataFrame(desc_list, columns=DESC_COLS).astype(np.float32)
            cand_df = pd.concat([cand_df, item_desc_df], axis=1)
            if split_name == "train":
                if np.random.rand() < 0.001:

                    assert cand_df["target_boost"].sum() == 1, \
                        "Sanity Check faild: more than 1 target"

                    assert cand_df["sasrec_rank"].min() == 1, \
                        "Sanity Check faild: rank min is not 1"
                    assert cand_df["sasrec_rank"].max() == len(cand_df), \
                        "Sanity Check faild: rank max not correct"

                    critical_cols = [
                        "user_id_internal",
                        "candidate_item_id_internal",
                        "sasrec_score",
                        "sasrec_rank",
                        "target_boost",
                        "item_price_proxy",
                        "price_diff",
                        "price_ratio",
                        "item_popularity",
                        "user_item_freq_hist",
                        "days_since_last_purchase",
                        "item_n_machines_train",
                        "is_same_product_model_as_last",
                        "item_machine_support",
                        "item_prodmodel_support",
                    ] + DESC_COLS

                    if "days_since_last_purchase_same_item" in cand_df.columns:
                        critical_cols.append("days_since_last_purchase_same_item")

                    if "days_since_last_purchase_same_item" in cand_df.columns:
                        critical_cols.append("days_since_last_purchase_same_item")

                        bad_prev = cand_df[
                            (cand_df["user_item_freq_hist"] > 0) &
                            (cand_df["days_since_last_purchase_same_item"] < 0)
                        ]

                        bad_new = cand_df[
                            (cand_df["user_item_freq_hist"] == 0) &
                            (cand_df["days_since_last_purchase_same_item"] >= 0)
                        ]

                        assert len(bad_prev) == 0, f"Sanity: freq>0 but recency<0 found ({len(bad_prev)})"
                        assert len(bad_new) == 0, f"Sanity: freq==0 but recency>=0 found ({len(bad_new)})"

                        assert not cand_df["days_since_last_purchase_same_item"].isnull().any(), \
                         "Sanity: NaN in days_since_last_purchase_same_item"

                    missing_cols = [c for c in critical_cols if c not in cand_df.columns]
                    assert len(missing_cols) == 0, f"Sanity Check: missing columns {missing_cols}"

                    assert not cand_df[critical_cols].isnull().values.any(), \
                        "Sanity Check faild: NaN found in critical columns"


            # buffer rows
            rows_buffer.append(cand_df)
            rows_buffer_count += len(cand_df)


            total_events += 1

            if rows_buffer_count >= flush_rows:

                part_path = os.path.join(out_dir, f"{split_name}_part_{int(time.time_ns())}.parquet")
                big = pd.concat(rows_buffer, ignore_index=True)
                rows_buffer = []
                rows_buffer_count = 0
                big.to_parquet(part_path, index=False, engine="pyarrow", compression="snappy")
                logging.info(f"[{split_name}] Flush: {part_path} | rows={len(big)}")

            # update user state with current interaction
            hist_items.append(target_item_internal)


            hist_ts.append(current_ts)
            hist_item_last_ts[target_item_internal] = current_ts
            hist_item_freq[target_item_internal] = hist_item_freq.get(target_item_internal, 0) + 1
            last_prodmodel = event_product_model_id if not pd.isna(event_product_model_id) else last_prodmodel
            user_state["global_user_event_idx"][uid_internal] = global_user_event_idx + 1
            # update running price stats AFTER using them (leakage-safe)
            running_sum += current_price
            running_cnt += 1

        # save back per-user states
        user_state["hist_items"][uid_internal] = hist_items
        user_state["hist_ts"][uid_internal] = hist_ts
        user_state["hist_item_last_ts"][uid_internal] = hist_item_last_ts
        user_state["hist_item_freq"][uid_internal] = hist_item_freq
        user_state["last_product_model"][uid_internal] = last_prodmodel
        user_state["running_sum_price"][uid_internal] = running_sum
        user_state["running_count"][uid_internal] = running_cnt

    # final flush
    if len(rows_buffer) > 0:
        part_path = os.path.join(out_dir, f"{split_name}_part_{int(time.time_ns())}.parquet")
        big = pd.concat(rows_buffer, ignore_index=True)
        big.to_parquet(part_path, index=False, engine="pyarrow", compression="snappy")
        logging.info(f"[{split_name}] Final flush: {part_path} | rows={len(big)}")

    logging.info(f"[{split_name}] Done. total_events={total_events} forced_positive_count={forced_positive_count} miss_groups={miss_groups} (miss@200 groups in valid/test is expected)")


def load_sasrec_model(checkpoint_path: str,config_dict: dict,device: str) -> Tuple[SASRec, 'Dataset']:
    """
    This Python function loads a pre-trained SASRec model and its associated RecBole dataset based on
    the provided checkpoint path, configuration dictionary, and device.

    :param checkpoint_path: The `checkpoint_path` parameter is the file path where the pre-trained model
    checkpoint is saved. This checkpoint contains the trained weights and model configuration that can
    be loaded to resume training or use the model for inference
    :type checkpoint_path: str
    :param config_dict: The `config_dict` parameter is a dictionary containing configuration settings
    for the SASRec model and dataset. It likely includes hyperparameters, model settings, and
    dataset-specific configurations needed to initialize the model and dataset properly. These settings
    could include things like the number of items, embedding dimensions, learning rates, batch
    :type config_dict: dict
    :param device: The `device` parameter in the `load_sasrec_model` function is a string that specifies
    the device on which the model will be loaded and trained. It can be set to either "cpu" or "cuda"
    depending on whether you want to use the CPU or GPU for computations
    :type device: str
    :return: The function `load_sasrec_model` returns a tuple containing an instance of the SASRec model
    and the associated dataset RecBole.
    """

    config = Config(model="SASRec", dataset="b2b_data", config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    model = SASRec(config, dataset).to(device)

    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
        logging.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        logging.warning("Checkpoint not found, using randomly initialized SASRec!")
    return model, dataset

def load_recbole_interactions(file_path: str) -> pd.DataFrame:
    """
    Load RecBole interaction data from a tab-separated file.
    This function reads  file containing user-item interaction records
    in RecBole format and returns them as a pandas DataFrame with proper
    column naming and type casting.
    Args:
        file_path (str): Path to the tab-separated values file containing
                        interaction data.
    Returns:
        pd.DataFrame: DataFrame with columns:
                     - 'user_id:token' (str): User identifier
                     - 'item_id:token' (str): Item identifier
                     - 'timestamp:float' (float): Interaction timestamp
    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        pd.errors.ParserError: If the file cannot be parsed as tab-separated values.
    """
    return pd.read_csv(
        file_path,
        sep='\t',
        header=0,
        names=['user_id:token', 'item_id:token', 'timestamp:float'],
        dtype={'user_id:token': str, 'item_id:token': str}
    )

def add_internal_ids(df: pd.DataFrame, user_map: dict, item_map: dict) -> pd.DataFrame:

    df['user_id_internal'] = df['user_id:token'].map(user_map)
    df['item_id_internal'] = df['item_id:token'].map(item_map)
    df.dropna(subset=['user_id_internal', 'item_id_internal'], inplace=True)
    df['user_id_internal'] = df['user_id_internal'].astype(int)
    df['item_id_internal'] = df['item_id_internal'].astype(int)
    return df

# -----------------------------
# MAIN
# -----------------------------
def main():
    """
    Main execution pipeline: orchestrate the two-stage recommender system.
    Workflow:
    1. Load pre-trained SASRec model and RecBole dataset
    2. Load and prepare train/valid/test interaction data with metadata
    3. Build train-only dictionaries (frozen across all splits to prevent leakage)
    4. Process each split sequentially with leakage-safe feature engineering
    5. Output rescoring datasets as Parquet files (partitioned for memory efficiency)
    """
    setup_logging()
    start_time = time.time()
    logging.info("Starting to create datasets for LightGBM re-scoring (train/valid/test).")

    # =============================================================================
    # PATHS & DIRECTORY SETUP
    # =============================================================================
    DATA_ROOT = os.path.join(os.getcwd() , "dataset")
    OUT_ROOT = os.path.join(os.getcwd(), "05-HYBRID_RERANKING", "NEW_RESCORING", "15processed_data", "15rescoring_parts")
    CHECKPOINT_PATH = os.path.join(os.getcwd(), "05-HYBRID_RERANKING", "NEW_RESCORING", "SASRec-Oct-01-2025_17-39-33.pth")

    TRAIN_DIR = os.path.join(OUT_ROOT, "train")
    VALID_DIR = os.path.join(OUT_ROOT, "valid")
    TEST_DIR = os.path.join(OUT_ROOT, "test")
    ensure_dir(TRAIN_DIR)
    ensure_dir(VALID_DIR)
    ensure_dir(TEST_DIR)

    # =============================================================================
    # SASREC HYPERPARAMETERS (from best tuning run)
    # =============================================================================
    BEST_SASREC_HYPERPARAMS = {
        "embedding_size": 512,
        "n_layers": 2,
        "n_heads": 8,
        "learning_rate": 0.001,
        "hidden_dropout_prob": 0.5,
        "dropout_rate": 0.3,
        "weight_decay": 1e-05,
        "max_seq_length": 300,
        "loss_type": "BPR",
        "sampling_size": 5
    }

    # =============================================================================
    # DEVICE SELECTION & MODEL LOADING
    # =============================================================================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {DEVICE}")

    config_dict = {
        'data_path': DATA_ROOT,
        'USER_ID_FIELD': 'user_id', 'ITEM_ID_FIELD': 'item_id', 'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        'field_separator': "\t", 'use_gpu': torch.cuda.is_available(),
    }
    config_dict.update(BEST_SASREC_HYPERPARAMS)

    try:
        # Load pre-trained SASRec model and extract vocabulary mappings
        model, recbole_dataset = load_sasrec_model(CHECKPOINT_PATH, config_dict, DEVICE)

        # Extract actual max sequence length from model's position embedding
        # (ensures consistency with checkpoint, not just config)
        MODEL_MAX_SEQ_LEN = int(model.position_embedding.num_embeddings)
        logging.info(f"[SASREC] checkpoint max_seq_length (position_embedding) = {MODEL_MAX_SEQ_LEN}")
        max_seq_len = MODEL_MAX_SEQ_LEN

        # Token-to-ID mappings (vocabulary)
        user_token_to_id = recbole_dataset.field2token_id['user_id']
        item_token_to_id = recbole_dataset.field2token_id['item_id']
        item_id2token_list = recbole_dataset.field2id_token['item_id']

        # =============================================================================
        # LOAD METADATA
        # =============================================================================
        metadata_path = os.path.join(DATA_ROOT, "full_dataset.csv")
        meta_df = pd.read_csv(metadata_path, sep=';', dtype={'CUSTOMER_ID': str, 'ITEM_ID': str})

        # Standardize column names
        meta_df.rename(columns={
            'CUSTOMER_ID': 'user_id',
            'ITEM_ID': 'item_id',
            'REQUEST_DATE': 'timestamp',
            'LOCATION': 'location',
            'LINE_ID': 'line_id',
            'MACHINE_ID': 'machine_id',
            'PRODUCT_MODEL_ID': 'product_model_id',
            'PROJECT_ID': 'project_id',
            'ORDER_ID': 'order_id',
            'ITEM_DESCRIPTION': 'item_description'
        }, inplace=True)
        # Parse timestamps and prices
        meta_df['timestamp'] = pd.to_datetime(meta_df['timestamp'])
        meta_df['price'] = pd.to_numeric(meta_df['PRICE_EXACT'], errors='coerce')

        # Deduplicate by request date and calculate occurrence count
        meta_df["request_date"] = meta_df["timestamp"].dt.floor("D")
        meta_df = meta_df.sort_values(["user_id", "item_id", "request_date", "order_id", "line_id"], na_position="last").reset_index(drop=True)
        meta_df["occ"] = meta_df.groupby(["user_id", "item_id", "request_date"]).cumcount()

        # =============================================================================
        # HELPER: Load & Prepare Interaction Splits
        # =============================================================================
        def load_and_prepare_split(inter_path, user_map, item_map, full_meta_df):
            """
            Load and prepare interaction data by merging with metadata and creating internal IDs.
            This function loads RecBole interaction data, adds internal user and item IDs,
            processes timestamps, and merges the data with full metadata information.
            The result is sorted by user, timestamp, and row ID for downstream processing.
            Args:
                inter_path (str): File path to the RecBole interactions data file.
                user_map (dict): Mapping dictionary to convert user tokens to internal user IDs.
                item_map (dict): Mapping dictionary to convert item tokens to internal item IDs.
                full_meta_df (pd.DataFrame): Full metadata DataFrame containing user, item, request_date,
                                             and occurrence information for matching interactions.
            Returns:
                pd.DataFrame: Merged DataFrame with columns:
                    - row_id: Original row index
                    - user_id_internal: Internal user ID
                    - item_id_internal: Internal item ID
                    - timestamp: Datetime of interaction
                    - event_date: Date of interaction (floored to day)
                    - user_id: Original user token
                    - item_id: Original item token
                    - occ: Occurrence count within user-item-date group
                    - [metadata columns from full_meta_df]
                    Sorted by user_id_internal, timestamp, and row_id in ascending order.
            Raises:
                AssertionError: If merge validation fails (1:1 relationship expected).
            Note:
                - Timestamps are converted from Unix timestamps (seconds)
                - Occurrences are counted per user-item-date combination
                - Merged using 1:1 validation to ensure data integrity
            """
            # Load interaction data from RecBole format
            df = load_recbole_interactions(inter_path)

            # Add row identifier for deduplication tracking
            df = df.reset_index(drop=True).reset_index().rename(columns={"index": "row_id"})

            # Convert user/item tokens to internal IDs and drop unmapped rows
            df = add_internal_ids(df, user_map, item_map)

            # Parse Unix timestamp (seconds) to datetime and extract date component
            df['timestamp'] = pd.to_datetime(df['timestamp:float'], unit='s')
            df["event_date"] = df["timestamp"].dt.floor("D")

            # Standardize token column names
            df.rename(columns={'user_id:token': 'user_id', 'item_id:token': 'item_id'}, inplace=True)

            # Sort for stable occurrence counting
            df = df.sort_values(["user_id", "item_id", "event_date", "row_id"]).reset_index(drop=True)
            df["occ"] = df.groupby(["user_id", "item_id", "event_date"]).cumcount()

            # Merge with metadata using 1:1 validation (each interaction row maps to exactly 1 metadata row)
            merged = pd.merge(
                df[['row_id', 'user_id_internal', 'item_id_internal', 'timestamp', 'event_date', 'user_id', 'item_id', 'occ']],
                full_meta_df,
                left_on=['user_id', 'item_id', 'event_date', 'occ'],
                right_on=['user_id', 'item_id', 'request_date', 'occ'],
                how='left',
                suffixes=('', '_meta'),
                validate='1:1'
            )
            logging.info(f"[{os.path.basename(inter_path)}] merged columns: {list(merged.columns)}")

            # Return sorted by user and timestamp (preserves chronology for leakage-safe feature engineering)
            return merged.sort_values(by=['user_id_internal', 'timestamp', 'row_id']).reset_index(drop=True)

        # =============================================================================
        # LOAD TRAIN/VALID/TEST SPLITS WITH METADATA
        # =============================================================================
        train_df = load_and_prepare_split(os.path.join(DATA_ROOT, 'b2b_data', 'b2b_data.train.inter'), user_token_to_id, item_token_to_id, meta_df)
        valid_df = load_and_prepare_split(os.path.join(DATA_ROOT, 'b2b_data', 'b2b_data.valid.inter'), user_token_to_id, item_token_to_id, meta_df)
        test_df  = load_and_prepare_split(os.path.join(DATA_ROOT, 'b2b_data', 'b2b_data.test.inter'), user_token_to_id, item_token_to_id, meta_df)

        # =============================================================================
        # SANITY CHECK — merge coverage for critical metadata
        # =============================================================================
        def report_merge_coverage(df, name):
            """Log NaN rates for critical features to detect merge issues."""
            logging.info(
                f"[MERGE][{name}] price NaN={df['price'].isna().mean():.2%} | "
                f"machine_id NaN={df['machine_id'].isna().mean():.2%} | "
                f"product_model_id NaN={df['product_model_id'].isna().mean():.2%} | "
                f"location NaN={df['location'].isna().mean():.2%}"
            )
        logging.info(f"[COVERAGE train] price NaN rate: {train_df['price'].isna().mean():.4f}")
        logging.info(f"[COVERAGE valid] price NaN rate: {valid_df['price'].isna().mean():.4f}")
        logging.info(f"[COVERAGE test ] price NaN rate: {test_df['price'].isna().mean():.4f}")
        report_merge_coverage(train_df, "train")
        report_merge_coverage(valid_df, "valid")
        report_merge_coverage(test_df,  "test")

        logging.info(f"Data loaded. Train shape: {train_df.shape}, Valid shape: {valid_df.shape}, Test shape: {test_df.shape}")

        # =============================================================================
        # DEBUG MODE: Run on single user for quick testing (optional)
        # =============================================================================
        DEBUG_ONE_USER = False
        DEBUG_USER_ID_INTERNAL = 1   # <-- insert the id of a user present in the train set for quick debugging

        if DEBUG_ONE_USER:
            train_df = train_df[train_df["user_id_internal"] == DEBUG_USER_ID_INTERNAL].copy()
            valid_df = valid_df[valid_df["user_id_internal"] == DEBUG_USER_ID_INTERNAL].copy()
            test_df  = test_df[test_df["user_id_internal"] == DEBUG_USER_ID_INTERNAL].copy()

            logging.info(
                f"DEBUG_ONE_USER enabled | user_id_internal={DEBUG_USER_ID_INTERNAL} | "
                f"train={train_df.shape} valid={valid_df.shape} test={test_df.shape}"
            )

        # =============================================================================
        # PARSE ITEM DESCRIPTIONS: Convert string embeddings to Python lists
        # =============================================================================
        def parse_embedding(x):
            """Safely parse string representation of embedding to Python list."""
            if isinstance(x, str):
                try:
                    v = ast.literal_eval(x)
                    return v if isinstance(v, list) else None
                except Exception:
                    return None
            return x

        train_df["ITEM_DESCRIPTION"] = train_df["item_description"].apply(parse_embedding)
        valid_df["ITEM_DESCRIPTION"] = valid_df["item_description"].apply(parse_embedding)
        test_df["ITEM_DESCRIPTION"]  = test_df["item_description"].apply(parse_embedding)

        # =============================================================================
        # BUILD TRAIN-ONLY DICTIONARIES (Frozen across all splits to prevent leakage)
        # =============================================================================
        logging.info("Precomputing train-only dictionaries (frozen for all splits).")

        # Item popularity: raw count in train set
        ITEM_POPULARITY_DICT = train_df["item_id_internal"].value_counts().to_dict()

        # Item price proxy: median price per item in train (used for test items never seen in train)
        ITEM_MEDIAN_PRICE_DICT = train_df.groupby("item_id_internal")["price"].median().to_dict()

        # Global fallback: median price across entire train set
        global_price_median = float(train_df["price"].median())

        # =============================================================================
        # SANITY CHECKS — PRICE DICTS (one-shot, after train dict build)
        # =============================================================================
        n_train_rows = len(train_df)
        n_price_nan = int(train_df["price"].isna().sum())
        logging.info(f"[PRICE] train rows={n_train_rows} | price NaN={n_price_nan} ({n_price_nan/n_train_rows:.3%})")

        # Validate global median is finite
        assert np.isfinite(global_price_median), f"[PRICE] global_price_median not finite: {global_price_median}"

        n_items_train = int(train_df["item_id_internal"].nunique())
        n_items_with_median = len(ITEM_MEDIAN_PRICE_DICT)
        logging.info(f"[PRICE] unique items in train={n_items_train} | items with median in dict={n_items_with_median}")

        # Check for NaN values inside the dictionary (should not exist after median computation)
        dict_nan = sum((v is None) or (isinstance(v, float) and np.isnan(v)) for v in ITEM_MEDIAN_PRICE_DICT.values())
        logging.info(f"[PRICE] NaN medians inside ITEM_MEDIAN_PRICE_DICT = {dict_nan}")

        # Spot-check: verify sample items' medians match recomputed values
        sample_items = train_df["item_id_internal"].dropna().astype(int).sample(min(10, n_items_train), random_state=42).tolist()
        for it in sample_items:
            med_recompute = float(train_df.loc[train_df["item_id_internal"] == it, "price"].median())
            med_dict = float(ITEM_MEDIAN_PRICE_DICT.get(it, np.nan))
            if not (np.isfinite(med_recompute) and np.isfinite(med_dict)):
                logging.warning(f"[PRICE] item {it}: non-finite median (recompute={med_recompute}, dict={med_dict})")
            else:
                if not np.isclose(med_recompute, med_dict, rtol=1e-6, atol=1e-6):
                    logging.warning(f"[PRICE] item {it}: median mismatch (recompute={med_recompute}, dict={med_dict})")

        # =============================================================================
        # BUILD B2B DOMAIN-SPECIFIC DICTIONARIES (Machine & Product Model compatibility)
        # =============================================================================

        # User location: mode location per user (customer-level constant)
        USER_LOCATION_DICT = train_df.groupby("user_id_internal")["location"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_dict()

        # Extract item-machine and item-prodmodel compatibility data from train
        tm = train_df[["item_id_internal", "machine_id", "product_model_id"]].copy()

        # Machine compatibility: pairs (item, machine) observed in train
        tm_machine = tm.dropna(subset=["machine_id"])
        ITEM_MACHINE_SET = set(zip(tm_machine["item_id_internal"].astype(int), tm_machine["machine_id"]))
        ITEM_MACHINE_SUPPORT = tm_machine.groupby(["item_id_internal", "machine_id"]).size().to_dict()

        # Product model compatibility: pairs (item, prodmodel) observed in train
        tm_pm = tm.dropna(subset=["product_model_id"])
        ITEM_PRODMODEL_SET = set(zip(tm_pm["item_id_internal"].astype(int), tm_pm["product_model_id"]))
        ITEM_PRODMODEL_SUPPORT = tm_pm.groupby(["item_id_internal", "product_model_id"]).size().to_dict()

        # Item versatility: count of distinct machines per item in train
        ITEM_N_MACHINES = tm.groupby("item_id_internal")["machine_id"].nunique().to_dict()

        # Set of all machines observed in train (for cold machine detection)
        TRAIN_MACHINES_SET = set(tm["machine_id"].dropna().unique())

        # Item-to-product models mapping (for compatibility checks)
        ITEM_TO_PRODMODEL_DICT = train_df.dropna(subset=['product_model_id']) \
                                    .groupby('item_id_internal')['product_model_id'] \
                                    .agg(set).to_dict()

        # Item description embeddings: deduplicate and build lookup dict
        ITEM_DESCRIPTION_DICT = (
                                train_df.dropna(subset=["ITEM_DESCRIPTION"])
                                        .drop_duplicates(subset=["item_id_internal"])
                                        .set_index("item_id_internal")["ITEM_DESCRIPTION"]
                                        .to_dict()
                            )

        logging.info("Train-only dicts ready.")

        # =============================================================================
        # INITIALIZE USER STATE TRACKER (Shared across splits for chronological safety)
        # =============================================================================
        user_state = {
            "hist_items": {},                # user -> list of item IDs (chronological history)
            "hist_ts": {},                   # user -> list of timestamps
            "hist_item_last_ts": {},         # user -> {item: last_timestamp}
            "hist_item_freq": {},            # user -> {item: frequency_count}
            "last_product_model": {},        # user -> product_model at last event
            "running_sum_price": {},         # user -> cumulative price sum (for avg)
            "running_count": {},             # user -> event count (for avg)
            "global_user_event_idx": {}      # user -> event counter (for event ID)
        }

        k = 200  # Top-K candidates from SASRec

        # =============================================================================
        # PROCESS TRAIN SPLIT
        # =============================================================================
        logging.info("[train] Starting processing.")
        process_split(
            "train",
            train_df,
            TRAIN_DIR,
            model,
            recbole_dataset,
            DEVICE,
            max_seq_len,
            k,
            ITEM_POPULARITY_DICT,
            ITEM_MEDIAN_PRICE_DICT,
            global_price_median,
            USER_LOCATION_DICT,
            ITEM_MACHINE_SET,
            ITEM_PRODMODEL_SET,
            ITEM_MACHINE_SUPPORT,
            ITEM_PRODMODEL_SUPPORT,
            ITEM_N_MACHINES,
            TRAIN_MACHINES_SET,
            ITEM_TO_PRODMODEL_DICT,
            ITEM_DESCRIPTION_DICT,
            user_state,
            force_positive=True,  # Ensure all positive items in top-k (train)
            item_id2token_list=item_id2token_list,
            item_token_to_id=item_token_to_id
        )

        # =============================================================================
        # PROCESS VALIDATION SPLIT (with shared user state from train)
        # =============================================================================
        logging.info("[valid] Starting processing. force_positive=False")
        process_split(
            "valid",
            valid_df,
            VALID_DIR,
            model,
            recbole_dataset,
            DEVICE,
            max_seq_len,
            k,
            ITEM_POPULARITY_DICT,
            ITEM_MEDIAN_PRICE_DICT,
            global_price_median,
            USER_LOCATION_DICT,
            ITEM_MACHINE_SET,
            ITEM_PRODMODEL_SET,
            ITEM_MACHINE_SUPPORT,
            ITEM_PRODMODEL_SUPPORT,
            ITEM_N_MACHINES,
            TRAIN_MACHINES_SET,
            ITEM_TO_PRODMODEL_DICT,
            ITEM_DESCRIPTION_DICT,
            user_state,
            force_positive=False,  # Allow miss@200 (no target guarantee)
            item_id2token_list=item_id2token_list,
            item_token_to_id=item_token_to_id
        )

        # =============================================================================
        # PROCESS TEST SPLIT (with shared user state from train + valid)
        # =============================================================================
        logging.info("[test] Starting processing. force_positive=False")
        process_split(
            "test",
            test_df,
            TEST_DIR,
            model,
            recbole_dataset,
            DEVICE,
            max_seq_len,
            k,
            ITEM_POPULARITY_DICT,
            ITEM_MEDIAN_PRICE_DICT,
            global_price_median,
            USER_LOCATION_DICT,
            ITEM_MACHINE_SET,
            ITEM_PRODMODEL_SET,
            ITEM_MACHINE_SUPPORT,
            ITEM_PRODMODEL_SUPPORT,
            ITEM_N_MACHINES,
            TRAIN_MACHINES_SET,
            ITEM_TO_PRODMODEL_DICT,
            ITEM_DESCRIPTION_DICT,
            user_state,
            force_positive=False,  # Allow miss@200
            item_id2token_list=item_id2token_list,
            item_token_to_id=item_token_to_id
        )

        # =============================================================================
        # SUMMARY
        # =============================================================================
        end_time = time.time()
        logging.info(f"ALL SPLITS COMPLETED in {end_time - start_time:.2f} seconds.")
        logging.info(f"Output root folder: {OUT_ROOT}")

    except Exception:
        logging.error("Fatal error during execution.", exc_info=True)



if __name__ == "__main__":
    main()
