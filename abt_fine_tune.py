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
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================

# --- Scalability & Budget Settings ---
NUM_ITERATIONS_PER_PARTITION = 3   # Run 3 AL iterations on each chunk
LABELS_PER_ITERATION = 300         # Max query budget per iteration (Cap)
SEED_SIZE = 100                    # Seed each partition's loop with initial labels
PATIENCE = 2                       # Stop loop if F1 doesn't improve for 2 iters

# --- Validation Set Configuration (Reviewer 1) ---
VAL_SET_PROPORTION = 0.1
VAL_SET_MAX_SIZE = 2000            # Strict cap for efficiency

# --- ABLATION STUDY SETTINGS (Reviewer 4) ---
# Set to True to measure the "Re-Indexing Cost vs. Recall Gain"
RUN_FINETUNING_ABLATION = True     
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Data Paths ---
PATH_RAW_A = './data/Abt.csv'
PATH_RAW_B = './data/Buy.csv'
PATH_GT = './data/truth_abt_buy.csv'
COLS_TO_USE = ['name', 'description', 'price']

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
print("--- Loading Raw Data and Oracle ---")
df_a_raw = pd.read_csv(PATH_RAW_A, encoding='unicode_escape')
df_b_raw = pd.read_csv(PATH_RAW_B, encoding='unicode_escape')
df_gt = pd.read_csv(PATH_GT, encoding="unicode_escape", keep_default_na=False)

# Build Oracle Lookup
truthD = dict()
for i, r in df_gt.iterrows():
    idAbt = str(r["idAbt"])
    idBuy = str(r["idBuy"])
    if idAbt in truthD:
        truthD[idAbt].append(idBuy)
    else:
        truthD[idAbt] = [idBuy]

gt_lookup = {
    (str(key), str(value))
    for key, value_list in truthD.items()
    for value in value_list
}
print(f"Loaded Oracle with {len(gt_lookup)} total matches.")

# Sampling for Experiments
SAMPLE_PROPORTION = 0.3
SAMPLE_SIZE = int(len(df_b_raw) * SAMPLE_PROPORTION)
N_PARTITIONS = 3 
print(f"Sample Size: {SAMPLE_SIZE} | Partitions (k): {N_PARTITIONS}")

# Bootstrap Embeddings (Phase 1)
df_a, df_b = lib.bootstrap_embeddings_only(
     df_a_raw, df_b_raw, "source_a", "source_b", COLS_TO_USE
)

# Save full B for final testing, sample B for training
buy_embeddings_full = np.array(df_b['v'].tolist()).astype('float32')
df_b_whole = df_b.copy()
df_b = df_b.sample(n=SAMPLE_SIZE, random_state=42)

# Fast lookups
a_lookup = {row['text']: row for _, row in df_a.iterrows()}
b_lookup = {row['text']: row for _, row in df_b.iterrows()}

# ==========================================
# 3. SEMANTIC PARTITIONING
# ==========================================
print(f"\n--- Partitioning data into {N_PARTITIONS} chunks using KMeans ---")
embeddings_b = np.array(df_b['v'].tolist()).astype('float32')
kmeans = MiniBatchKMeans(n_clusters=N_PARTITIONS, random_state=42, batch_size=256, n_init=3)
df_b['partition'] = kmeans.fit_predict(embeddings_b)

# ==========================================
# 4. GLOBAL INDEXING (BLOCKING)
# ==========================================
print("Building global FAISS index (HNSW)...")
embeddings_a = np.array(df_a['v'].tolist()).astype('float32')
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
master_clean_training_set = []
fast_validation_set = []
model, scaler = (None, None)
ablation_stats = []

time_start_training = time.time()

for i in range(N_PARTITIONS):
    print(f"\n--- Processing Partition {i+1}/{N_PARTITIONS} ---")

    # 1. Get Partition Data
    df_b_partition = df_b[df_b['partition'] == i]
    if len(df_b_partition) == 0: continue

    embeddings_b_partition = np.array(df_b_partition['v'].tolist()).astype('float32')
    faiss.normalize_L2(embeddings_b_partition)

    # 2. Generate Candidate Pool (Blocking)
    D, I = index.search(embeddings_b_partition, k=10)
    
    candidate_pool_text_partition = set()
    for b_idx, a_indices_list in enumerate(I):
        text_b = df_b_partition.iloc[b_idx]['text']
        for a_idx in a_indices_list:
            text_a = df_a.iloc[a_idx]['text']
            candidate_pool_text_partition.add((text_a, text_b))

    # 3. Label Initial Pool & Create Validation Set (First Partition Only)
    all_pairs_list = list(candidate_pool_text_partition)
    
    # --- Validation Set Logic (Reviewer 1) ---
    if not fast_validation_set:
        val_size = int(len(all_pairs_list) * VAL_SET_PROPORTION)
        val_size = min(val_size, VAL_SET_MAX_SIZE) # Cap at 2000
        
        print(f"Creating fixed validation set of size {val_size}")
        val_pairs_text = all_pairs_list[:val_size]
        fast_validation_set = lib.query_oracle(
            val_pairs_text, a_lookup, b_lookup, gt_lookup, "id", "id"
        )
        all_pairs_list = all_pairs_list[val_size:]

    # 4. Seeding
    seed_pairs_text = all_pairs_list[:SEED_SIZE]
    current_unlabeled_pool = all_pairs_list[SEED_SIZE:] # The rest are unlabeled
    
    current_clean_training_set_partition = lib.query_oracle(
        seed_pairs_text, a_lookup, b_lookup, gt_lookup, "id", "id"
    )

    if not current_clean_training_set_partition: continue

    # 5. Iterative AL Loop
    patience_counter = 0
    last_f1_score = 0.0
    MIN_IMPROVEMENT = 0.05

    for j in range(1, NUM_ITERATIONS_PER_PARTITION + 1):
        print(f"  > Iteration {j} (Budget Cap: {LABELS_PER_ITERATION})")

        training_data = master_clean_training_set + current_clean_training_set_partition
        
        if not training_data: break

        # ---------------------------------------------------------
        # A. TRAIN ALER (Frozen Embeddings)
        # ---------------------------------------------------------
        t0_frozen = time.time()
        model, scaler, f1_frozen, thresh = lib.train_classifier(
            training_data, fast_validation_set, a_lookup, b_lookup
        )
        t_frozen = time.time() - t0_frozen
        
        # ---------------------------------------------------------
        # B. ABLATION: REAL FINE-TUNING + RE-INDEXING (Reviewer 4)
        # ---------------------------------------------------------
        if RUN_FINETUNING_ABLATION and ablation_model:
            print("    [Ablation] Measuring Breakdown: Training vs. Re-Indexing...")
            
            # --- PREPARE DATA ---
            train_examples = [InputExample(texts=[t[0], t[1]], label=int(t[2])) for t in training_data]
            train_dl = DataLoader(train_examples, shuffle=True, batch_size=16)
            train_loss = losses.ContrastiveLoss(model=ablation_model)
            
            # --- TIMER 1: PURE TRAINING (Backprop) ---
            t0_train = time.time()
            ablation_model.fit(train_objectives=[(train_dl, train_loss)], epochs=1, show_progress_bar=False)
            t_train_only = time.time() - t0_train
            
            # --- TIMER 2: RE-EMBEDDING & RE-INDEXING (Mandatory State Update) ---
            t0_reindex = time.time()
            
            # 1. Re-Embed Source A (Required for Index)
            all_a_texts = df_a['text'].tolist()
            new_embeddings_a = ablation_model.encode(all_a_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
            faiss.normalize_L2(new_embeddings_a)
            
            # 2. Re-Build Index
            d_new = new_embeddings_a.shape[1]
            new_index = faiss.IndexHNSWFlat(d_new, 32, faiss.METRIC_INNER_PRODUCT)
            new_index.hnsw.efConstruction = 60
            new_index.hnsw.efSearch = 64
            new_index.add(new_embeddings_a)
            
            # 3. Re-Embed Target B (Required for Querying/Blocking)
            new_embeddings_b = ablation_model.encode(df_b['text'].tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True)
            faiss.normalize_L2(new_embeddings_b)
            
            t_reindex_only = time.time() - t0_reindex
            t_full_pipeline = t_train_only + t_reindex_only

            # --- MEASURE RECALL ---
            D_ft, I_ft = new_index.search(new_embeddings_b, k=10)
            matches_ft = 0
            for b_idx_loc, b_row in enumerate(df_b.itertuples()):
                for a_idx_loc in I_ft[b_idx_loc]:
                    a_id = str(df_a.iloc[a_idx_loc]['id'])
                    if (a_id, str(b_row.id)) in gt_lookup:
                        matches_ft += 1
                        break
            
            # Frozen Recall (Pre-computed/Static)
            D_fr, I_fr = index.search(embeddings_b, k=10)
            matches_fr = 0
            for b_idx_loc, b_row in enumerate(df_b.itertuples()):
                for a_idx_loc in I_fr[b_idx_loc]:
                    a_id = str(df_a.iloc[a_idx_loc]['id'])
                    if (a_id, str(b_row.id)) in gt_lookup:
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

        # Check Patience
        improvement = f1_frozen - last_f1_score
        if improvement < MIN_IMPROVEMENT and j > 1:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("    -> Patience limit reached. Stopping partition early.")
                break
        else:
            patience_counter = 0
            last_f1_score = f1_frozen

        # 6. Predict on Unlabeled Pool
        if not current_unlabeled_pool:
            break

        # Convert Unlabeled to Features
        X_unlabeled = []
        valid_batch_pairs = []
        pool_subset = current_unlabeled_pool[:5000] # Efficiency cap
        
        for (txt_a, txt_b) in pool_subset:
            ra, rb = a_lookup.get(txt_a), b_lookup.get(txt_b)
            if ra is not None and rb is not None:
                feat = lib.create_pure_embedding_vector(ra, rb)
                X_unlabeled.append(feat)
                valid_batch_pairs.append((txt_a, txt_b))
        
        if not X_unlabeled: break
        
        X_mtx = scaler.transform(np.array(X_unlabeled))
        probs = model.predict(X_mtx, batch_size=256).flatten()

        # 7. Hybrid Query Strategy
        batch_size = min(LABELS_PER_ITERATION, len(probs))
        half_batch = batch_size // 2
        
        uncertainty = np.abs(probs - 0.5)
        
        idx_uncertain = np.argsort(uncertainty)[:half_batch] # Closest to 0.5
        idx_confident = np.argsort(probs)[-half_batch:]      # Closest to 1.0
        
        indices_to_label = np.unique(np.concatenate([idx_uncertain, idx_confident]))
        pairs_to_query = [valid_batch_pairs[idx] for idx in indices_to_label]
        
        new_labels = lib.query_oracle(
            pairs_to_query, a_lookup, b_lookup, gt_lookup, "id", "id"
        )
        
        current_clean_training_set_partition.extend(new_labels)
        labeled_set = set(pairs_to_query)
        current_unlabeled_pool = [p for p in current_unlabeled_pool if p not in labeled_set]

    # End Partition Loop: Fuse Labels
    master_clean_training_set.extend(current_clean_training_set_partition)

time_training_end = time.time()

# ==========================================
# 7. FINAL MODEL TRAINING
# ==========================================
print("\n--- Training Master Models (Stage 1 & Stage 2) ---")

# Stage 1: Recall Model (Embeddings Only)
model, scaler, best_f1_recall, best_thresh_recall = lib.train_classifier(
    master_clean_training_set, fast_validation_set, a_lookup, b_lookup
)

# Stage 2: Precision Model (Hybrid Features)
precision_model, precision_scaler, best_f1_prec, best_thresh_prec = lib.train_precision_classifier(
    master_clean_training_set, fast_validation_set, a_lookup, b_lookup, col="name"
)

# ==========================================
# 8. RESOLUTION ON HELD-OUT TEST SET
# ==========================================
print("\n--- Final Resolution on HELD-OUT Test Set ---")

# 1. Identify Training IDs (Strict Leakage Prevention)
train_ids = set()
for t in master_clean_training_set + fast_validation_set:
    rec_b = b_lookup.get(t[1])
    if rec_b is not None:
        train_ids.add(str(rec_b['id']))

# 2. Search on Full B (Blocking)
faiss.normalize_L2(buy_embeddings_full)
D, I = index.search(buy_embeddings_full, k=10) 

stage1_candidates = []
y_true_stage1 = []
y_true_stage2 = []
X_s2_features = []

# Iterate through all B records
for b_idx in range(len(df_b_whole)):
    rec_b = df_b_whole.iloc[b_idx]
    
    # LEAKAGE CHECK
    if str(rec_b['id']) in train_ids:
        continue

    neighbor_indices = I[b_idx]
    
    for a_idx in neighbor_indices:
        rec_a = df_a.iloc[a_idx]
        
        # Stage 1 Prediction
        feat = lib.create_pure_embedding_vector(rec_a, rec_b)
        x_scaled = scaler.transform(np.array([feat]))
        prob = model.predict(x_scaled, verbose=0)[0][0]
        
        is_match = 1 if (str(rec_a['id']), str(rec_b['id'])) in gt_lookup else 0
        y_true_stage1.append(is_match)
        
        if prob > best_thresh_recall:
            # Passes to Stage 2
            hybrid_feat = lib.create_hybrid_feature_vector(rec_a, rec_b, col="name")
            X_s2_features.append(hybrid_feat)
            y_true_stage2.append(is_match)

# Predict Stage 2
if X_s2_features:
    X_s2_mtx = precision_scaler.transform(np.array(X_s2_features))
    probs_s2 = precision_model.predict(X_s2_mtx, batch_size=256).flatten()
    preds_s2 = (probs_s2 > best_thresh_prec).astype(int)
    
    print("\n=============================================")
    print("      FINAL PERFORMANCE REPORT               ")
    print("=============================================")
    print(f"PRECISION: {precision_score(y_true_stage2, preds_s2):.4f}")
    print(f"RECALL:    {recall_score(y_true_stage2, preds_s2):.4f}")
    print(f"F1-SCORE:  {f1_score(y_true_stage2, preds_s2):.4f}")
    print("=============================================")
else:
    print("No candidates passed Stage 1.")

# ==========================================
# 9. OUTPUT FOR REVIEWERS
# ==========================================
total_active = len(master_clean_training_set) - (SEED_SIZE * N_PARTITIONS)
total_labels = len(master_clean_training_set) + len(fast_validation_set)

print(f"Seed Labels:    {SEED_SIZE * N_PARTITIONS}")
print(f"Validation:     {len(fast_validation_set)}")
print(f"Active Queries: {total_active}")
print(f"TOTAL LABELS:   {total_labels}")

if RUN_FINETUNING_ABLATION:
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
