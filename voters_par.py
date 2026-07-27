import pandas as pd
import numpy as np
import lib as lib
import random
import faiss
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.cluster import MiniBatchKMeans
import time
import sys

# --- 1. Configuration & Setup ---

N_PARTITIONS = 5
NUM_ITERATIONS_PER_PARTITION = 3
LABELS_PER_ITERATION = 300
SEED_SIZE = 300

VAL_SET_PROPORTION = 0.1
VAL_SET_MAX_SIZE = 2000

# Chunk Sizes
INDEXING_BATCH_SIZE = 50_000   # Add to FAISS in batches of 50k
RESOLUTION_BATCH_SIZE = 20_000 # Search/Predict in batches of 20k

PATH_RAW_A = './data/test_voters_A_1M.txt'
PATH_RAW_B = './data/test_voters_B_1M.txt'
PATH_GT = './data/truth_VOTERS_1M.csv'
ID_COL_A = 'id1'
ID_COL_B = 'id2'
COLS_TO_USE = [ "surname", "name", "address" ,"town" ,"ps" ]

start_total_time = time.time()

# --- 2. Load Data and Oracle ---
print("--- Loading Raw Data and Oracle ---")
cols=["id", "surname", "name", "address" ,"town" ,"ps" ]
# nrows=500_000 for testing. Remove 'nrows' for full run.
df_a_raw = pd.read_csv(PATH_RAW_A, sep=",",encoding="unicode_escape",names=cols, on_bad_lines='skip')
df_b_raw = pd.read_csv(PATH_RAW_B, sep=",",encoding="unicode_escape",names=cols, on_bad_lines='skip')
df_gt = pd.read_csv(PATH_GT, encoding="utf-8", keep_default_na=False, nrows=200_000)

truthD = dict()
for i, r in df_gt.iterrows():
      id1 = r["id1"]
      id2 = [r["id2"]]
      truthD[id1] = id2

matches = len(truthD.keys())
print("total matches=",matches)
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
faiss.normalize_L2(embeddings_a) # Normalize ONCE here before chunking
faiss.normalize_L2(b_embeddings)

# Memory Footprint Calculation
embed_memory_size = b_embeddings.nbytes + embeddings_a.nbytes
print(f"Embeddings Memory Footprint: {embed_memory_size / (1024*1024):.2f} MB")

df_b_whole  = df_b
SAMPLE_PROPORTION = 0.2
SAMPLE_SIZE= int(len(df_b) * SAMPLE_PROPORTION)
df_b = df_b.sample(n=SAMPLE_SIZE, random_state=42)

a_lookup = {row['text']: row for _, row in df_a.iterrows()}
b_lookup = {row['text']: row for _, row in df_b.iterrows()}

# --- 4. Partitioning ---
print(f"\n--- Partitioning data into {N_PARTITIONS} chunks using KMeans ---")
# Only use the sample for KMeans training to be fast
embeddings_b_sample = np.array(df_b['v'].tolist()).astype('float32')
faiss.normalize_L2(embeddings_b_sample)
kmeans = MiniBatchKMeans(n_clusters=N_PARTITIONS, random_state=42, batch_size=256, n_init=3)
df_b['partition'] = kmeans.fit_predict(embeddings_b_sample)


# --- 5. Build Global FAISS Index for A (Chunked Indexing) ---
print("\n--- Building Global FAISS Index (Chunked) ---")
start_index_time = time.time()

d = embeddings_a.shape[1]
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 60
index.hnsw.efSearch = 64

# --- CHUNKED INDEXING LOOP ---
num_records_a = embeddings_a.shape[0]
print(f"Indexing {num_records_a} records in batches of {INDEXING_BATCH_SIZE}...")

for start_idx in tqdm(range(0, num_records_a, INDEXING_BATCH_SIZE)):
    end_idx = min(start_idx + INDEXING_BATCH_SIZE, num_records_a)
    batch_emb = embeddings_a[start_idx:end_idx]
    
    # Note: We already normalized embeddings_a above, so batch_emb is normalized
    index.add(batch_emb)

end_index_time = time.time()
indexing_time = end_index_time - start_index_time
print(f"HNSW Index Build Time: {indexing_time:.2f} seconds")
print(f"Total Index Size: {index.ntotal} records")

# FAISS Memory Footprint (Estimate)
index_memory_footprint = (d * 4 * index.ntotal) + (32 * 32 * 4 * index.ntotal)
print(f"FAISS Index Memory Footprint (Est.): {index_memory_footprint / (1024*1024):.2f} MB")


st_training = time.time()
master_clean_training_set = []
fast_validation_set = [] 
model, scaler = (None, None) 

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

        model, scaler, f1, thresh = lib.train_classifier(
            training_set_for_this_iter,
            fast_validation_set,  
            a_lookup, b_lookup
        )
        print(f"  Iter {j} F1-Score: {f1:.4f}")

        if not unlabeled_pool_text_partition:
            break 

        X_unlabeled_list, pairs_for_this_batch = [], []
        for (text_a, text_b) in unlabeled_pool_text_partition:
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
        unlabeled_pool_text_partition = list(set(unlabeled_pool_text_partition) - set(pairs_to_label_text))

    print(f"Partition {i+1} complete. Fusing {len(current_clean_training_set_partition)} clean labels.")
    master_clean_training_set.extend(current_clean_training_set_partition)


print("\n--- All Partitions Complete. Fusing All Labels. ---")

total_labels = len(master_clean_training_set)
num_positives = 0
num_negatives = 0

for pair_tuple in master_clean_training_set:
    label = pair_tuple[2]  
    if label == 1.0:
        num_positives += 1
    else:
        num_negatives += 1

print(f"--- Final Master Training Set Stats ---")
print(f"Total Labels Collected: {total_labels}")
print(f"  - Positives (Matches):    {num_positives}")
print(f"  - Negatives (No Matches): {num_negatives}")

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


print("\n--- Starting Final Resolution (Held-Out Test Set, Chunked Processing) ---")
# Only calculating retrieval/resolution time here as requested by end-to-end flow
start_resolution = time.time()

train_val_ids = set(df_b['id'].astype(str).tolist()) 

# b_embeddings are already normalized above

# --- CHUNKED SEARCH AND PREDICTION ---
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
    
    # Iterate through batch results
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
active_queries = len(master_clean_training_set) - (SEED_SIZE * N_PARTITIONS) 

print("\n" + "="*30)
print("PAPER TABLE 2 DATA (VOTERS)")
print("="*30)
print(f"Seed (B_seed): {SEED_SIZE * N_PARTITIONS}") 
print(f"Valid (|V|):   {len(fast_validation_set)}")
print(f"Active Loop:   {active_queries}") 
print(f"Total Budget:  {real_total_labels}")
print("="*30)

end_total_time = time.time()
print(f"\nTotal End-to-End Wall Clock Time: {end_total_time - start_total_time:.2f} seconds")
