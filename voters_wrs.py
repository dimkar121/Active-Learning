import pandas as pd
import numpy as np
import lib as lib
import random
import faiss
import math
import heapq
import time
import sys
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# --- 1. Configuration & Setup ---

# --- Scalability Settings (WRS-ALER) ---
NUM_ITERATIONS = 15         # Replaces 5 partitions * 3 iterations
LABELS_PER_ITERATION = 300
SEED_SIZE = 300

# --- Validation Set Configuration ---
VAL_SET_PROPORTION = 0.1
VAL_SET_MAX_SIZE = 2000

# --- Chunk Sizes (Large Scale) ---
INDEXING_BATCH_SIZE = 50_000   # Add to FAISS in batches of 50k
RESOLUTION_BATCH_SIZE = 20_000 # Search/Predict in batches of 20k

# --- WRS-ALER Specific Parameters ---
ALPHA = 0.5                 # Split ratio for exploration/exploitation reservoirs
ETA_MULTIPLIER = 0.15       # Determines the base capacity eta
K_NEIGHBORS = 10            # Top-k neighbors to retrieve
EPSILON = 1e-6              # Smoothing term
GAMMA = 0.01                # Dynamic truncation threshold

PATH_RAW_A = './data/test_voters_A_1M.txt'
PATH_RAW_B = './data/test_voters_B_1M.txt'
PATH_GT = './data/truth_VOTERS_1M.csv'
ID_COL_A = 'id1'
ID_COL_B = 'id2'
COLS_TO_USE = ["surname", "name", "address", "town", "ps"]

start_total_time = time.time()

# --- 2. Load Data and Oracle ---
print("--- Loading Raw Data and Oracle ---")
cols = ["id", "surname", "name", "address", "town", "ps"]
df_a_raw = pd.read_csv(PATH_RAW_A, sep=",", encoding="unicode_escape", names=cols, on_bad_lines='skip')
df_b_raw = pd.read_csv(PATH_RAW_B, sep=",", encoding="unicode_escape", names=cols, on_bad_lines='skip')
df_gt = pd.read_csv(PATH_GT, encoding="utf-8", keep_default_na=False, nrows=200_000)

truthD = dict()
for i, r in df_gt.iterrows():
      id1 = r["id1"]
      id2 = [r["id2"]]
      truthD[id1] = id2

matches = len(truthD.keys())
print("total matches=", matches)
gt_lookup = {
    (str(key), str(value))
    for key, value_list in truthD.items()
    for value in value_list
}
print(f"Loaded Oracle with {len(gt_lookup)} total matches.")


# --- 3. Bootstrap Embeddings (Phase 1) ---
print("\n--- Generating SBERT Embeddings ---")
start_embed_time = time.time()

df_a, df_b = lib.bootstrap_embeddings_only(
       df_a_raw, df_b_raw, "source_a", "source_b", COLS_TO_USE
)

end_embed_time = time.time()
embedding_time = end_embed_time - start_embed_time
print(f"Embedding Computation Time: {embedding_time:.2f} seconds")

# Prepare embeddings
b_embeddings = np.array(df_b['v'].tolist()).astype('float32')
embeddings_a = np.array(df_a['v'].tolist()).astype('float32')
faiss.normalize_L2(embeddings_a) 
faiss.normalize_L2(b_embeddings)

embed_memory_size = b_embeddings.nbytes + embeddings_a.nbytes
print(f"Embeddings Memory Footprint: {embed_memory_size / (1024*1024):.2f} MB")

df_b_whole = df_b

# WRS-ALER evaluates the space dynamically, no static downsampling needed
SAMPLE_PROPORTION = 1.0

a_lookup = {row['text']: row for _, row in df_a.iterrows()}
b_lookup = {row['text']: row for _, row in df_b.iterrows()}


# --- 4. Build Global FAISS Index for A (Chunked Indexing) ---
print("\n--- Building Global FAISS Index (Chunked) ---")
start_index_time = time.time()

d = embeddings_a.shape[1]
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 60
index.hnsw.efSearch = 64

num_records_a = embeddings_a.shape[0]
print(f"Indexing {num_records_a} records in batches of {INDEXING_BATCH_SIZE}...")

for start_idx in tqdm(range(0, num_records_a, INDEXING_BATCH_SIZE)):
    end_idx = min(start_idx + INDEXING_BATCH_SIZE, num_records_a)
    batch_emb = embeddings_a[start_idx:end_idx]
    index.add(batch_emb)

end_index_time = time.time()
indexing_time = end_index_time - start_index_time
print(f"HNSW Index Build Time: {indexing_time:.2f} seconds")
print(f"Total Index Size: {index.ntotal} records")

index_memory_footprint = (d * 4 * index.ntotal) + (32 * 32 * 4 * index.ntotal)
print(f"FAISS Index Memory Footprint (Est.): {index_memory_footprint / (1024*1024):.2f} MB")


# --- 5. WRS-ALER: Dynamic Probabilistic Filtering ---
print("\n--- Executing WRS-ALER Dynamic Filtering ---")

ETA = int(ETA_MULTIPLIER * len(b_embeddings) * K_NEIGHBORS)
ETA_B = int(ALPHA * ETA)
ETA_C = int((1 - ALPHA) * ETA)
print(f"Reservoir Capacities - Total: {ETA}, R_b: {ETA_B}, R_c: {ETA_C}")

R_b = [] 
R_c = [] 

ins = 0
no = 0

time_start_filtering = time.time()

for i in tqdm(range(len(b_embeddings)), desc="Querying Candidates"):
    query_vec = b_embeddings[i:i+1]
    D, I = index.search(query_vec, k=K_NEIGHBORS)
    
    inserted_this_step = False
    
    for j in range(K_NEIGHBORS):
        a_idx = I[0][j]
        sim = D[0][j]
        
        xi = (sim + 1.0) / 2.0
        xi = max(EPSILON, min(1.0 - EPSILON, xi)) 
        
        w_b = -xi * math.log(xi) - (1.0 - xi) * math.log(1.0 - xi) + EPSILON
        w_c = abs(xi - 0.5) + EPSILON
        
        u = random.uniform(0, 1)
        tau_b = math.pow(u, 1.0 / w_b)
        tau_c = math.pow(u, 1.0 / w_c)
        
        if len(R_b) < ETA_B or tau_b > R_b[0][0]:
            if len(R_b) == ETA_B:
                heapq.heappop(R_b)
            heapq.heappush(R_b, (tau_b, a_idx, i))
            inserted_this_step = True
            
        if len(R_c) < ETA_C or tau_c > R_c[0][0]:
            if len(R_c) == ETA_C:
                heapq.heappop(R_c)
            heapq.heappush(R_c, (tau_c, a_idx, i))
            inserted_this_step = True
            
    no += K_NEIGHBORS
    if inserted_this_step:
        ins += 1
        
    if len(R_b) == ETA_B and len(R_c) == ETA_C:
        insertion_rate = ins / no
        if insertion_rate < GAMMA and no > ETA:
            print(f"\nDynamic truncation triggered! Stopping at record {i}/{len(b_embeddings)}")
            print(f"Final Insertion Rate: {insertion_rate:.4f} < {GAMMA}")
            break

candidate_pool_text = set()
for _, a_idx, b_idx in R_b + R_c:
    text_a = df_a.iloc[a_idx]['text']
    text_b = df_b.iloc[b_idx]['text']
    candidate_pool_text.add((text_a, text_b))

print(f"Filtering Time: {time.time() - time_start_filtering:.2f} seconds.")
print(f"Formulated Bounded Candidate Pool (C) size: {len(candidate_pool_text)}")


# --- 6. Active Learning Loop ---
st_training = time.time()
master_clean_training_set = []

labeled_pool = lib.query_oracle(
    list(candidate_pool_text), a_lookup, b_lookup, gt_lookup, "id", "id"
)
random.shuffle(labeled_pool)

val_set_size = int(len(labeled_pool) * VAL_SET_PROPORTION)
val_set_size = min(val_set_size, VAL_SET_MAX_SIZE)
fast_validation_set = labeled_pool[:val_set_size]
print(f"Created a global, fixed validation set of {val_set_size} pairs.")

unlabeled_pool_text = [p[:2] for p in labeled_pool[val_set_size:]]

seed_pairs_text = unlabeled_pool_text[:SEED_SIZE]
unlabeled_pool_text = unlabeled_pool_text[SEED_SIZE:]
master_clean_training_set = lib.query_oracle(
    seed_pairs_text, a_lookup, b_lookup, gt_lookup, "id", "id"
)

MIN_IMPROVEMENT_THRESHOLD = 0.05
PATIENCE = 3 
patience_counter = 0
last_f1_score = 0.0

model, scaler = (None, None)

print("\n--- Running Global WRS-ALER Active Learning Loop ---")
for j in range(1, NUM_ITERATIONS + 1):
    print(f"\n  Iteration {j}:")
    
    if not master_clean_training_set:
        print("No training data yet. Skipping iteration.")
        break
        
    model, scaler, f1, thresh = lib.train_classifier(
        master_clean_training_set, 
        fast_validation_set,  
        a_lookup, b_lookup
    )
    print(f"  Iter {j} F1-Score: {f1:.4f}")
    
    improvement = f1 - last_f1_score
    if improvement < MIN_IMPROVEMENT_THRESHOLD and j > 1:
       patience_counter += 1
       print(f"  F1-Score did not improve significantly. Patience counter: {patience_counter}/{PATIENCE}")
       if patience_counter >= PATIENCE:
          print(f"  F1-Score plateaued for {PATIENCE} iterations. Stopping loop early.")
          break
    else:
        patience_counter = 0
        last_f1_score = f1

    if not unlabeled_pool_text:
        print("Unlabeled pool exhausted.")
        break 
        
    X_unlabeled_list, pairs_for_this_batch = [], []
    for (text_a, text_b) in unlabeled_pool_text:
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
    
    pairs_to_label_text = [pairs_for_this_batch[idx] for idx in indices_to_label]
    
    newly_labeled_pairs = lib.query_oracle(
        pairs_to_label_text, a_lookup, b_lookup, gt_lookup, "id", "id"
    )
    master_clean_training_set.extend(newly_labeled_pairs)
    
    unlabeled_pool_text = list(set(unlabeled_pool_text) - set(pairs_to_label_text))


print("\n--- Training Phase Complete ---")
total_labels = len(master_clean_training_set)
num_positives = sum(1 for p in master_clean_training_set if p[2] == 1.0)
num_negatives = total_labels - num_positives

print(f"Total Labels Collected: {total_labels}")
print(f"  - Matches (TP): {num_positives}")
print(f"  - Non-Matches (TN): {num_negatives}")

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
         col="name"
    )
    print(f"Master Precision Model F1-Score: {best_f1:.4f}")
else:
    print("No clean labels gathered, skipping precision model.")

end_training = time.time()
print(f"Active Learning Training Time: {end_training - st_training:.2f} seconds")


# --- 7. Final Two-Stage Resolution ---
print("\n--- Starting Final Resolution (Held-Out Test Set, Chunked Processing) ---")

# Apply leakage fix: extract IDs only from the explicitly labeled records
labeled_pairs = master_clean_training_set + fast_validation_set
train_val_ids = {str(b_lookup[text_b]['id']) for (_, text_b, _) in labeled_pairs}

num_queries = b_embeddings.shape[0]
print(f"Processing {num_queries} queries in batches of {RESOLUTION_BATCH_SIZE}...")

total_tp_stage1 = 0
total_fp_stage1 = 0
total_fn_stage1 = 0 
total_tp_stage2 = 0
total_fp_stage2 = 0
total_fn_stage2 = 0 

for start_idx in tqdm(range(0, num_queries, RESOLUTION_BATCH_SIZE)):
    end_idx = min(start_idx + RESOLUTION_BATCH_SIZE, num_queries)
    batch_embeddings = b_embeddings[start_idx:end_idx]
    
    # 1. Search Batch
    D, I = index.search(batch_embeddings, k=5)
    
    X_stage1_features_batch = []
    stage1_pairs_data_batch = []
    y_true_list_stage1_batch = []
    
    for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
        b_record = df_b_whole.iloc[global_idx]
        
        # Leakage Filter
        if str(b_record['id']) in train_val_ids:
            continue

        a_indices = I[local_idx]
        
        for a_idx in a_indices:
            a_record = df_a.iloc[a_idx]
            
            stage1_pairs_data_batch.append((a_record, b_record))
            X_stage1_features_batch.append(lib.create_pure_embedding_vector(a_record, b_record))
            
            a_id = str(a_record["id"])
            b_id = str(b_record["id"])
            is_match = 1.0 if (a_id, b_id) in gt_lookup else 0.0
            y_true_list_stage1_batch.append(is_match)

    if not X_stage1_features_batch:
        continue

    # 2. Stage 1 Prediction
    X_stage1_matrix = np.array(X_stage1_features_batch)
    X_stage1_scaled = scaler.transform(X_stage1_matrix)
    stage1_probs = model.predict(X_stage1_scaled, batch_size=1024, verbose=0).flatten() 
    stage1_decisions = (stage1_probs > best_threshold1).astype(int)
    y_true_stage1_array = np.array(y_true_list_stage1_batch)

    # Accumulate Stage 1 Metrics
    tp = np.sum((stage1_decisions == 1) & (y_true_stage1_array == 1))
    fp = np.sum((stage1_decisions == 1) & (y_true_stage1_array == 0))
    fn = np.sum((stage1_decisions == 0) & (y_true_stage1_array == 1))
    
    total_tp_stage1 += tp
    total_fp_stage1 += fp
    total_fn_stage1 += fn

    # 3. Stage 2 Filter 
    stage2_candidate_indices = np.where(stage1_probs > best_threshold1)[0]
    
    if len(stage2_candidate_indices) > 0:
        X_stage2_features_batch = []
        y_true_list_stage2_batch = y_true_stage1_array[stage2_candidate_indices]
        
        for idx in stage2_candidate_indices:
            a_record, b_record = stage1_pairs_data_batch[idx]
            hybrid_features = lib.create_hybrid_feature_vector(a_record, b_record, col="name")
            X_stage2_features_batch.append(hybrid_features)
            
        X_stage2_matrix = np.array(X_stage2_features_batch)
        X_stage2_scaled = precision_scaler.transform(X_stage2_matrix)
        stage2_probs = precision_model.predict(X_stage2_scaled, batch_size=1024, verbose=0).flatten()
        stage2_decisions = (stage2_probs > best_threshold2).astype(int)
        
        # Accumulate Stage 2 Metrics
        tp2 = np.sum((stage2_decisions == 1) & (y_true_list_stage2_batch == 1))
        fp2 = np.sum((stage2_decisions == 1) & (y_true_list_stage2_batch == 0))
        fn2 = np.sum((stage2_decisions == 0) & (y_true_list_stage2_batch == 1))
        
        total_tp_stage2 += tp2
        total_fp_stage2 += fp2
        total_fn_stage2 += fn2


# --- Final Metric Calculation (Aggregated) ---

print("\n--- Final Aggregated Results (Held-Out Test Set) ---")

# Stage 1 Metrics
recall1 = total_tp_stage1 / (total_tp_stage1 + total_fn_stage1) if (total_tp_stage1 + total_fn_stage1) > 0 else 0
precision1 = total_tp_stage1 / (total_tp_stage1 + total_fp_stage1) if (total_tp_stage1 + total_fp_stage1) > 0 else 0
f1_1 = 2 * (precision1 * recall1) / (precision1 + recall1) if (precision1 + recall1) > 0 else 0

print(f"Stage 1 Classifier F1:        {f1_1:.4f}")
print(f"Stage 1 Classifier Recall:    {recall1:.4f}")
print(f"Stage 1 Classifier Precision: {precision1:.4f}")

# Stage 2 Metrics
recall2 = total_tp_stage2 / (total_tp_stage2 + total_fn_stage2) if (total_tp_stage2 + total_fn_stage2) > 0 else 0
precision2 = total_tp_stage2 / (total_tp_stage2 + total_fp_stage2) if (total_tp_stage2 + total_fp_stage2) > 0 else 0
f1_2 = 2 * (precision2 * recall2) / (precision2 + recall2) if (precision2 + recall2) > 0 else 0

print("-" * 30)
print(f"Stage 2 Classifier F1:        {f1_2:.4f}")
print(f"Stage 2 Classifier Recall:    {recall2:.4f}")
print(f"Stage 2 Classifier Precision: {precision2:.4f}")


real_total_labels = len(master_clean_training_set) + len(fast_validation_set)
active_queries = len(master_clean_training_set) - SEED_SIZE

print("\n" + "="*30)
print("PAPER TABLE 2 DATA (VOTERS)")
print("="*30)
print(f"Seed (B_seed): {SEED_SIZE}") 
print(f"Valid (|V|):   {len(fast_validation_set)}")
print(f"Active Loop:   {active_queries}") 
print(f"Total Budget:  {real_total_labels}")
print("="*30)

end_total_time = time.time()
print(f"\nTotal End-to-End Wall Clock Time: {end_total_time - start_total_time:.2f} seconds")
