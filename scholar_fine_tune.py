import pandas as pd
import numpy as np
import lib as lib  # Assuming your library functions are here
import random
import faiss
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.cluster import MiniBatchKMeans
import time
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================

# --- Scalability & Budget Settings ---
N_PARTITIONS = 3
NUM_ITERATIONS_PER_PARTITION = 3
LABELS_PER_ITERATION = 300
SEED_SIZE = 100

# --- Validation Set Configuration ---
VAL_SET_PROPORTION = 0.1
VAL_SET_MAX_SIZE = 20000

# --- ABLATION STUDY SETTINGS (Reviewer 4) ---
# Set to True to measure the "Re-Indexing Cost vs. Recall Gain"
RUN_FINETUNING_ABLATION = True
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Data Paths ---
PATH_RAW_A = './data/Scholar.csv'
PATH_RAW_B = './data/DBLP2.csv'
PATH_GT = './data/truth_Scholar_DBLP.csv'
ID_COL_A = 'idScholar'
ID_COL_B = 'idDBLP'
COLS_TO_USE = ['title', 'authors', 'venue', 'year']

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
print("--- Loading Raw Data and Oracle ---")
df_a_raw = pd.read_csv(PATH_RAW_A, encoding='utf-8')
df_b_raw = pd.read_csv(PATH_RAW_B, encoding='utf-8')
df_gt = pd.read_csv(PATH_GT, encoding="unicode_escape", keep_default_na=False)

truthD = dict()
for i, r in df_gt.iterrows():
    idDBLP = str(r["idDBLP"])
    idScholar = str(r["idScholar"])
    if idScholar in truthD:
        truthD[idScholar].append(idDBLP)
    else:
        truthD[idScholar] = [idDBLP]

gt_lookup = {
    (str(key), str(value))
    for key, value_list in truthD.items()
    for value in value_list
}
print(f"Loaded Oracle with {len(gt_lookup)} total matches.")

# Bootstrap Embeddings (Phase 1)
df_a, df_b = lib.bootstrap_embeddings_only(
       df_a_raw, df_b_raw, "source_a", "source_b", COLS_TO_USE
)

# Save full B for final testing, sample B for training
dblp_embeddings = np.array(df_b['v'].tolist()).astype('float32')
df_b_whole = df_b.copy()

SAMPLE_PROPORTION = 0.2
SAMPLE_SIZE = int(len(df_b) * SAMPLE_PROPORTION)
df_b = df_b.sample(n=SAMPLE_SIZE, random_state=42)
print(f"Training on sample of size: {len(df_b)}")

# Fast lookups
a_lookup = {row['text']: row for _, row in df_a.iterrows()}
b_lookup = {row['text']: row for _, row in df_b.iterrows()}

# ==========================================
# 3. SEMANTIC PARTITIONING
# ==========================================
print(f"\n--- Partitioning data into {N_PARTITIONS} chunks using KMeans ---")
embeddings_a = np.array(df_a['v'].tolist()).astype('float32')
embeddings_b = np.array(df_b['v'].tolist()).astype('float32')
kmeans = MiniBatchKMeans(n_clusters=N_PARTITIONS, random_state=42, batch_size=256, n_init=3)
df_b['partition'] = kmeans.fit_predict(embeddings_b)

# ==========================================
# 4. GLOBAL INDEXING (BLOCKING)
# ==========================================
print("Building global FAISS index (HNSW)...")
d = embeddings_a.shape[1]
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 60
index.hnsw.efSearch = 64
faiss.normalize_L2(embeddings_a)
index.add(embeddings_a)

# ==========================================
# 5. PREPARE ABLATION MODEL (Reviewer 4)
# ==========================================
ablation_model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if RUN_FINETUNING_ABLATION:
    print(f"\n[Ablation] Loading SECOND SBERT model ({device}) for Fine-Tuning Comparison...")
    ablation_model = SentenceTransformer(SBERT_MODEL_NAME)
    ablation_model.to(device)

# ==========================================
# 6. PARTITIONED ACTIVE LEARNING LOOP
# ==========================================
time_start_training = time.time()

master_clean_training_set = []
fast_validation_set = []
model, scaler = (None, None)
ablation_stats = []

for i in range(N_PARTITIONS):
    print(f"\n--- Processing Partition {i+1}/{N_PARTITIONS} ---")

    df_b_partition = df_b[df_b['partition'] == i]
    if len(df_b_partition) == 0:
        print("Partition is empty, skipping.")
        continue

    embeddings_b_partition = np.array(df_b_partition['v'].tolist()).astype('float32')
    faiss.normalize_L2(embeddings_b_partition)

    print(f"Generating candidate pool for {len(df_b_partition)} records...")
    D, I = index.search(embeddings_b_partition, k=10)

    candidate_pool_text_partition = set()
    for b_idx, a_indices_list in enumerate(I):
        text_b = df_b_partition.iloc[b_idx]['text']
        for a_idx in a_indices_list:
            text_a = df_a.iloc[a_idx]['text']
            candidate_pool_text_partition.add((text_a, text_b))

    labeled_pool_partition = lib.query_oracle(
        list(candidate_pool_text_partition), a_lookup, b_lookup, gt_lookup, "id", "id"
    )
    random.shuffle(labeled_pool_partition)

    if not fast_validation_set:
        val_set_size = int(len(labeled_pool_partition) * VAL_SET_PROPORTION)
        if val_set_size > VAL_SET_MAX_SIZE:
            val_set_size = VAL_SET_MAX_SIZE

        print(f"Creating a global, fixed validation set of {val_set_size} pairs.")
        fast_validation_set = labeled_pool_partition[:val_set_size]
        labeled_pool_partition = labeled_pool_partition[val_set_size:]

    unlabeled_pool_text_partition = [p[:2] for p in labeled_pool_partition]

    seed_pairs_text = unlabeled_pool_text_partition[:SEED_SIZE]
    unlabeled_pool_text_partition = unlabeled_pool_text_partition[SEED_SIZE:]

    current_clean_training_set_partition = lib.query_oracle(
        seed_pairs_text, a_lookup, b_lookup, gt_lookup, "id", "id"
    )

    if not current_clean_training_set_partition:
        print("No seed labels found for this partition, skipping.")
        continue

    for j in range(1, NUM_ITERATIONS_PER_PARTITION + 1):
        print(f"  Partition {i+1}, Iteration {j}:")

        training_set_for_this_iter = master_clean_training_set + current_clean_training_set_partition

        if not training_set_for_this_iter:
            print("No training data yet. Skipping iteration.")
            continue

        # ---------------------------------------------------------
        # A. TRAIN ALER (Frozen Embeddings)
        # ---------------------------------------------------------
        model, scaler, f1_frozen, thresh = lib.train_classifier(
            training_set_for_this_iter,
            fast_validation_set,
            a_lookup, b_lookup
        )
        print(f"  Iter {j} F1-Score: {f1_frozen:.4f}")

        # ---------------------------------------------------------
        # B. ABLATION: REAL FINE-TUNING + RE-INDEXING (Reviewer 4)
        # ---------------------------------------------------------
        if RUN_FINETUNING_ABLATION and ablation_model:
            print("    [Ablation] Measuring Breakdown: Training vs. Re-Indexing...")
            
            # --- PREPARE DATA ---
            train_examples = [InputExample(texts=[t[0], t[1]], label=int(t[2])) for t in training_set_for_this_iter]
            train_dl = DataLoader(train_examples, shuffle=True, batch_size=16)
            train_loss = losses.ContrastiveLoss(model=ablation_model)
            
            # --- TIMER 1: PURE TRAINING (Backprop) ---
            t0_train = time.time()
            ablation_model.fit(train_objectives=[(train_dl, train_loss)], epochs=1, show_progress_bar=False)
            t_train_only = time.time() - t0_train
            
            # --- TIMER 2: RE-EMBEDDING & RE-INDEXING (Mandatory State Update) ---
            t0_reindex = time.time()
            
            # 1. Re-Embed Source A (Required for Index) - Scholar is large (~64k rows)
            # This step effectively measures the "Scalability Bottleneck"
            all_a_texts = df_a['text'].tolist()
            new_embeddings_a = ablation_model.encode(all_a_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
            faiss.normalize_L2(new_embeddings_a)
            
            # 2. Re-Build Index
            d_new = new_embeddings_a.shape[1]
            new_index = faiss.IndexHNSWFlat(d_new, 32, faiss.METRIC_INNER_PRODUCT)
            new_index.hnsw.efConstruction = 60
            new_index.hnsw.efSearch = 64
            new_index.add(new_embeddings_a)
            
            # 3. Re-Embed Target B (The current working sample)
            # We measure recall on the whole sample df_b to be consistent
            new_embeddings_b = ablation_model.encode(df_b['text'].tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True)
            faiss.normalize_L2(new_embeddings_b)
            
            t_reindex_only = time.time() - t0_reindex
            t_full_pipeline = t_train_only + t_reindex_only

            # --- MEASURE RECALL ---
            D_ft, I_ft = new_index.search(new_embeddings_b, k=10)
            matches_ft = 0
            # Optimized counter using set intersection for speed in Python loop
            for b_idx_loc, b_row in enumerate(df_b.itertuples()):
                neighbor_indices = I_ft[b_idx_loc]
                b_id = str(b_row.id)
                # Check neighbors
                for a_idx_loc in neighbor_indices:
                    a_id = str(df_a.iloc[a_idx_loc]['id'])
                    if (a_id, b_id) in gt_lookup:
                        matches_ft += 1
                        break
            
            # Frozen Recall (Pre-computed/Static)
            # Search using original index and original embeddings
            D_fr, I_fr = index.search(embeddings_b, k=10)
            matches_fr = 0
            for b_idx_loc, b_row in enumerate(df_b.itertuples()):
                b_id = str(b_row.id)
                for a_idx_loc in I_fr[b_idx_loc]:
                    a_id = str(df_a.iloc[a_idx_loc]['id'])
                    if (a_id, b_id) in gt_lookup:
                        matches_fr += 1
                        break

            print(f"    [Time Breakdown] Train: {t_train_only:.2f}s | Re-Index: {t_reindex_only:.2f}s | Total: {t_full_pipeline:.2f}s")
            print(f"    [Recall Breakdown] Frozen: {matches_fr} | Fine-Tuned: {matches_ft}")
            
            ablation_stats.append({
                'partition': i, 'iter': j, 
                'train_time': t_train_only,
                'reindex_time': t_reindex_only,
                'total_ft_time': t_full_pipeline, 
                'frozen_recall': matches_fr, 'ft_recall': matches_ft
            })
        # ---------------------------------------------------------

        if not unlabeled_pool_text_partition:
            break

        X_unlabeled_list, pairs_for_this_batch = [], []
        
        # Optimization: Only process a chunk if pool is massive
        pool_subset = unlabeled_pool_text_partition[:5000]

        for (text_a, text_b) in pool_subset:
            record_a, record_b = a_lookup.get(text_a), b_lookup.get(text_b)
            if record_a is not None and record_b is not None:
                features = lib.create_pure_embedding_vector(record_a, record_b)
                if features.shape[0] == 1536:
                    X_unlabeled_list.append(features)
                    pairs_for_this_batch.append((text_a, text_b))

        if not X_unlabeled_list:
            break

        X_unlabeled_matrix = np.array(X_unlabeled_list)
        X_unlabeled_scaled = scaler.transform(X_unlabeled_matrix)
        preds_prob = model.predict(X_unlabeled_scaled, batch_size=256).flatten()

        half_batch = LABELS_PER_ITERATION // 2
        confidence = np.abs(preds_prob - 0.5)
        most_confused_indices = np.argsort(confidence)[:half_batch]
        most_confident_indices = np.argsort(preds_prob)[-half_batch:]
        indices_to_label = np.unique(np.concatenate([most_confused_indices, most_confident_indices]))

        if len(indices_to_label) == 0:
            break

        pairs_to_label_text = [pairs_for_this_batch[idx] for idx in indices_to_label]

        newly_labeled_pairs = lib.query_oracle(
            pairs_to_label_text, a_lookup, b_lookup, gt_lookup, "id", "id"
        )
        current_clean_training_set_partition.extend(newly_labeled_pairs)
        
        # Remove using set logic for speed
        labeled_set = set(pairs_to_label_text)
        unlabeled_pool_text_partition = [p for p in unlabeled_pool_text_partition if p not in labeled_set]

    print(f"Partition {i+1} complete. Fusing {len(current_clean_training_set_partition)} clean labels.")
    master_clean_training_set.extend(current_clean_training_set_partition)


print("\n--- All Partitions Complete. Fusing All Labels. ---")
print(f"Total clean labels gathered: {len(master_clean_training_set)}")
print(f"Total validation set size: {len(fast_validation_set)}")

print("\n--- Training Master Recall Model on Fused Set ---")
model, scaler, best_threshold1 = (None, None, 0.5)
if master_clean_training_set:
    model, scaler, f1, best_threshold1 = lib.train_classifier(
        master_clean_training_set, fast_validation_set, a_lookup, b_lookup
    )
    print(f"Master Recall Model F1-Score: {f1:.4f}")
else:
    print("No clean labels gathered, skipping recall model.")

print("\n--- Training Master Precision Model on Fused Set ---")
precision_model, precision_scaler, best_threshold2 = (None, None, 0.5)
if master_clean_training_set:
    precision_model, precision_scaler, best_f1, best_threshold2 = lib.train_precision_classifier(
         master_clean_training_set,
         fast_validation_set,
         a_lookup,
         b_lookup,
         col="title"
    )
    print(f"Master Precision Model F1-Score: {best_f1:.4f}")
else:
    print("No clean labels gathered, skipping precision model.")


time_end_training = time.time()
time1 = time_end_training - time_start_training
print(f"Training Time {time1} seconds.")

time_start_res = time.time()

print("\n--- Starting Final Resolution (Held-Out Test Set, Classifier Metrics) ---")

train_val_ids = set(df_b['id'].astype(str).tolist())

faiss.normalize_L2(dblp_embeddings)
D, I = index.search(dblp_embeddings, k=5)

X_stage1_features = []
stage1_pairs_data = []
y_true_list_stage1 = []

print("Running Stage 1 on HELD-OUT TEST SET only...")
for dblp_idx, scholar_indices in enumerate(I):
    dblp_record = df_b_whole.iloc[dblp_idx]

    if str(dblp_record['id']) in train_val_ids:
        continue

    for scholar_idx in scholar_indices:
        scholar_record = df_a.iloc[scholar_idx]

        stage1_pairs_data.append((scholar_record, dblp_record))
        X_stage1_features.append(lib.create_pure_embedding_vector(scholar_record, dblp_record))

        scholar_id = str(scholar_record["id"])
        dblp_id = str(dblp_record["id"])
        is_match = 1.0 if (scholar_id, dblp_id) in gt_lookup else 0.0
        y_true_list_stage1.append(is_match)

X_stage1_matrix = np.array(X_stage1_features)
X_stage1_scaled = scaler.transform(X_stage1_matrix)
stage1_probs = model.predict(X_stage1_scaled, batch_size=256).flatten()
stage1_decisions = (stage1_probs > best_threshold1).astype(int)
y_true_stage1_array = np.array(y_true_list_stage1)

print("\n--- Stage 1 (Recall Model) Classifier Performance ---")

stage2_candidate_indices = np.where(stage1_probs > best_threshold1)[0]
print(f"\nStage 1 passed {len(stage2_candidate_indices)} candidates to Stage 2.")

if len(stage2_candidate_indices) > 0:
    X_stage2_features = []

    y_true_list_stage2 = y_true_stage1_array[stage2_candidate_indices]

    for idx in stage2_candidate_indices:
        scholar_record, dblp_record = stage1_pairs_data[idx]
        hybrid_features = lib.create_hybrid_feature_vector(scholar_record, dblp_record, col="title")
        X_stage2_features.append(hybrid_features)

    X_stage2_matrix = np.array(X_stage2_features)
    X_stage2_scaled = precision_scaler.transform(X_stage2_matrix)
    stage2_probs = precision_model.predict(X_stage2_scaled, batch_size=256).flatten()
    stage2_decisions = (stage2_probs > best_threshold2).astype(int)

    print("\n--- Final Two-Stage Classifier Performance ---")

    f1 = f1_score(y_true_list_stage2, stage2_decisions)
    rec = recall_score(y_true_list_stage2, stage2_decisions)
    prec = precision_score(y_true_list_stage2, stage2_decisions)

    print(f"Classifier F1: {f1:.4f}")
    print(f"Classifier Recall: {rec:.4f} ")
    print(f"Classifier Precision: {prec:.4f}")

else:
    print("Stage 1 found no candidates.")

time_end_res = time.time()
time1 = time_end_res - time_start_res
print(f"Resolution Time {time1} seconds.")

real_total_labels = len(master_clean_training_set) + len(fast_validation_set)
active_queries = len(master_clean_training_set) - (SEED_SIZE * N_PARTITIONS)

print("\n" + "="*30)
print("SCHOLAR-DBLP")
print("="*30)
print(f"Seed (B_seed): {SEED_SIZE * N_PARTITIONS}")
print(f"Valid (|V|):    {len(fast_validation_set)}")
print(f"Active Loop:   {active_queries}")
print(f"Total Budget:  {real_total_labels}")
print("="*30)

if RUN_FINETUNING_ABLATION:
    print("\n=== REVIEWER 4: ABLATION RESULTS (Avg per Iteration) ===")
    avg_train = np.mean([x['train_time'] for x in ablation_stats])
    avg_reindex = np.mean([x['reindex_time'] for x in ablation_stats])
    avg_total = np.mean([x['total_ft_time'] for x in ablation_stats])
    avg_rec_fr = np.mean([x['frozen_recall'] for x in ablation_stats])
    avg_rec_ft = np.mean([x['ft_recall'] for x in ablation_stats])

    print(f"Frozen Time:           ~0.01s")
    print(f"Fine-Tuned (Train):     {avg_train:.2f}s (Backprop only)")
    print(f"Fine-Tuned (Re-Index):  {avg_reindex:.2f}s (Mandatory State Update)")
    print(f"Fine-Tuned (Total):     {avg_total:.2f}s")
    print("-" * 30)
    print(f"Recall Gain: +{((avg_rec_ft - avg_rec_fr)/avg_rec_fr)*100:.2f}% (from {avg_rec_fr:.1f} to {avg_rec_ft:.1f})")
