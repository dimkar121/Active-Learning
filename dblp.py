import pandas as pd
import numpy as np
import lib as lib
import random
import faiss
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.cluster import MiniBatchKMeans
import time

# --- 1. Configuration & Setup ---

# --- Scalability Settings ---
N_PARTITIONS = 7 # Split the dataset into 20 chunks
NUM_ITERATIONS_PER_PARTITION = 4 # Run 3 AL iterations on each chunk
LABELS_PER_ITERATION = 300 # Query 200 labels per iteration
SEED_SIZE = 1000             # Seed each partition's loop with 20 labels

# --- NEW: Validation Set Configuration ---
# We create one fixed, fast validation set.
# We'll make it proportional to the *first* partition's candidate pool,
# but cap it to ensure it's always fast.
VAL_SET_PROPORTION = 0.1
VAL_SET_MAX_SIZE = 20000  # Cap at 20,000 pairs

# --- Data paths and columns ---
PATH_RAW_A = './data/test_dblp_A.txt'
PATH_RAW_B = './data/test_dblp_B.txt'
PATH_GT = './data/truth_DBLP.csv'
ID_COL_A = 'id1'
ID_COL_B = 'id2'
COLS_TO_USE = ["author1","author2","title","year"]

# --- 2. Load Data and Oracle ---
print("--- Loading Raw Data and Oracle ---")
cols=["id","author1","author2","title","year"]
df_a_raw = pd.read_csv(PATH_RAW_A, sep=",",encoding="utf-8",names=cols, on_bad_lines='skip' )
df_b_raw = pd.read_csv(PATH_RAW_B, sep=",",encoding="utf-8",names=cols, on_bad_lines='skip' )
df_gt = pd.read_csv(PATH_GT, encoding="utf-8",  keep_default_na=False)

# Build the Oracle (gt_lookup)
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
df_a, df_b = lib.bootstrap_embeddings_only(
      df_a_raw, df_b_raw, "source_a", "source_b", COLS_TO_USE
)


b_embeddings = np.array(df_b['v'].tolist()).astype('float32')
df_b_whole  = df_b
SAMPLE_PROPORTION = 0.3
SAMPLE_SIZE= int(len(df_b) * SAMPLE_PROPORTION)
df_b = df_b.sample(n=SAMPLE_SIZE, random_state=42)

# Create fast lookup dicts (text -> full record)
a_lookup = {row['text']: row for _, row in df_a.iterrows()}
b_lookup = {row['text']: row for _, row in df_b.iterrows()}

# --- 4. Partitioning (The "Chunks" Strategy) ---
print(f"\n--- Partitioning data into {N_PARTITIONS} chunks using KMeans ---")
embeddings_a = np.array(df_a['v'].tolist()).astype('float32')
embeddings_b = np.array(df_b['v'].tolist()).astype('float32')
kmeans = MiniBatchKMeans(n_clusters=N_PARTITIONS, random_state=42, batch_size=256, n_init=3)
df_b['partition'] = kmeans.fit_predict(embeddings_b)

# --- 5. Build Global FAISS Index for B (DBLP) ---
print("Building global FAISS index...")
d = embeddings_a.shape[1]
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 60
index.hnsw.efSearch = 64
BATCH_SIZE = 1_000
for i in tqdm(range(0, len(embeddings_a), BATCH_SIZE)):
    # Slice the batch
    batch = embeddings_a[i : i + BATCH_SIZE]    
    # Normalize BATCH (if using Inner Product/Cosine)
    faiss.normalize_L2(batch)    
    # Add to index
    index.add(batch)

print(f"Index built. Total vectors: {index.ntotal}")
#faiss.normalize_L2(embeddings_a) # Normalize for inner product (cosine sim)
#index.add(embeddings_a)


st = time.time()
# --- 6. Partitioned Active Learning Loop (The New Core) ---
master_clean_training_set = []
fast_validation_set = [] # <--- This will be our fixed, fast validation set
model, scaler = (None, None) # We'll carry over the model from one loop to the next

for i in range(N_PARTITIONS):
    print(f"\n--- Processing Partition {i+1}/{N_PARTITIONS} ---")
    
    # 1. Get this partition's data
    df_b_partition = df_b[df_b['partition'] == i]
    if len(df_b_partition) == 0:
        print("Partition is empty, skipping.")
        continue
        
    embeddings_b_partition = np.array(df_b_partition['v'].tolist()).astype('float32')
    faiss.normalize_L2(embeddings_b_partition) # Normalize for search

    # 2. Generate Candidate Pool for THIS partition
    print(f"Generating candidate pool for {len(df_b_partition)} records...")
    D, I = index.search(embeddings_b_partition, k=10)
    
    candidate_pool_text_partition = set()
    for b_idx, a_indices_list in enumerate(I):
        text_b = df_b_partition.iloc[b_idx]['text']
        for a_idx in a_indices_list:
            text_a = df_a.iloc[a_idx]['text']
            candidate_pool_text_partition.add((text_a, text_b))
    
    # 3. Label and Split this partition's pool
    labeled_pool_partition = lib.query_oracle(
        list(candidate_pool_text_partition), a_lookup, b_lookup, gt_lookup, "id", "id"
    )
    random.shuffle(labeled_pool_partition)
    
    # --- NEW VALIDATION SET LOGIC ---
    if not fast_validation_set:
        # Create the fixed validation set ONCE from the first partition
        val_set_size = int(len(labeled_pool_partition) * VAL_SET_PROPORTION)
        if val_set_size > VAL_SET_MAX_SIZE:
            val_set_size = VAL_SET_MAX_SIZE
        
        print(f"Creating a global, fixed validation set of {val_set_size} pairs.")
        fast_validation_set = labeled_pool_partition[:val_set_size]
        labeled_pool_partition = labeled_pool_partition[val_set_size:] # The rest is for training
    # --- END NEW LOGIC ---
    
    # All pairs from this chunk (minus the val set, if it was chunk 0)
    # are now available for the unlabeled pool
    unlabeled_pool_text_partition = [p[:2] for p in labeled_pool_partition]
    
    # 4. Seed this partition's loop
    seed_pairs_text = unlabeled_pool_text_partition[:SEED_SIZE]
    unlabeled_pool_text_partition = unlabeled_pool_text_partition[SEED_SIZE:]
    
    current_clean_training_set_partition = lib.query_oracle(
        seed_pairs_text, a_lookup, b_lookup, gt_lookup, "id", "id"
    )
    
    if not current_clean_training_set_partition:
        print("No seed labels found for this partition, skipping.")
        continue
        
    # 5. Run the "mini" Active Learning Loop for this partition
    for j in range(1, NUM_ITERATIONS_PER_PARTITION + 1):
        print(f"  Partition {i+1}, Iteration {j}:")
        
        # Train on *all* labels found so far
        training_set_for_this_iter = master_clean_training_set + current_clean_training_set_partition
        
        if not training_set_for_this_iter:
            print("No training data yet. Skipping iteration.")
            continue
            
        model, scaler, f1, thresh = lib.train_classifier(
            training_set_for_this_iter, 
            fast_validation_set,  # <--- ALWAYS use the fixed, fast validation set
            a_lookup, b_lookup
        )
        print(f"  Iter {j} F1-Score: {f1:.4f}")
        
        # --- Predict on this partition's unlabeled pool ---
        if not unlabeled_pool_text_partition:
            break # This partition is out of labels
            
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
        
        # --- Query (Hybrid Strategy) ---
        half_batch = LABELS_PER_ITERATION // 2
        confidence = np.abs(preds_prob - 0.5)
        most_confused_indices = np.argsort(confidence)[:half_batch]
        most_confident_indices = np.argsort(preds_prob)[-half_batch:]
        indices_to_label = np.unique(np.concatenate([most_confused_indices, most_confident_indices]))
        
        if len(indices_to_label) == 0:
            break
            
        pairs_to_label_text = [pairs_for_this_batch[idx] for idx in indices_to_label]
        
        # --- Label & Add ---
        newly_labeled_pairs = lib.query_oracle(
            pairs_to_label_text, a_lookup, b_lookup, gt_lookup, "id", "id"
        )
        current_clean_training_set_partition.extend(newly_labeled_pairs)
        unlabeled_pool_text_partition = list(set(unlabeled_pool_text_partition) - set(pairs_to_label_text))

    # --- Fuse Labels ---
    print(f"Partition {i+1} complete. Fusing {len(current_clean_training_set_partition)} clean labels.")
    master_clean_training_set.extend(current_clean_training_set_partition)


print("\n--- All Partitions Complete. Fusing All Labels. ---")
print(f"Total clean labels gathered: {len(master_clean_training_set)}")
print(f"Total validation set size: {len(fast_validation_set)}")

# --- 7. Phase 3: Train Final Master Models ---
# We use the *same* fast_validation_set for our final test
# to be consistent with the loop.
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


end = time.time()
time1 = end - st
print(f"Training time {time1} seconds")


# --- 8. Phase 4: Final Two-Stage Resolution ---
# (This section is the same as your scholar.py script)
# ...


print("\n--- Starting Final Two-Stage Resolution ---")

# 1. Build the *global* FAISS index for df_a (Scholar)
#print(f"Building final FAISS index for Scholar (df_a)...")
#a_embeddings = np.array(df_a['v'].tolist()).astype(np.float32)

print(f"Resolving {len(b_embeddings)} records in batches of {BATCH_SIZE}...")

final_matches = []        # Store the final (record_a, record_b) matches
y_true_final = []         # Store true labels for metrics (optional)

# Iterate through B (Query Side)
for start_idx in tqdm(range(0, len(b_embeddings), BATCH_SIZE), desc="Resolving"):
    end_idx = min(start_idx + BATCH_SIZE, len(b_embeddings))
    
    # A. Get Batch of Query Vectors
    query_batch = b_embeddings[start_idx:end_idx]
    
    # B. Search Index
    # D_batch: distances, I_batch: indices of neighbors in df_a
    D_batch, I_batch = index.search(query_batch, 5)
    
    # C. Build Prediction Matrix for THIS Batch Only
    X_batch_features = []
    batch_pairs_metadata = [] # To track which records correspond to which row in X
    batch_labels = []         # To track ground truth for this batch
    
    for local_i in range(len(query_batch)):
        # Reconstruct global index for B
        global_b_idx = start_idx + local_i
        
        # Get Record B
        b_record = df_b_whole.iloc[global_b_idx]
        b_id = str(b_record["id"])
        
        # Iterate through its neighbors (from A)
        for neighbor_rank, global_a_idx in enumerate(I_batch[local_i]):
            if global_a_idx == -1: continue # Padding case
            
            # Get Record A
            a_record = df_a.iloc[global_a_idx]
            a_id = str(a_record["id"])
            
            # 1. Create Fast Features (Recall Model)
            # This vector is small (1536 floats)
            feats = lib.create_pure_embedding_vector(a_record, b_record)
            
            X_batch_features.append(feats)
            batch_pairs_metadata.append((a_record, b_record, a_id, b_id))
            
            # Check Ground Truth (if available)
            is_match = 1.0 if (a_id, b_id) in gt_lookup else 0.0
            batch_labels.append(is_match)

    # If no candidates found in this batch, skip
    if not X_batch_features:
        continue

    # D. Predict Stage 1 (Recall)
    # This matrix is small! (e.g. 1000 * 5 = 5000 rows)
    X_batch_np = np.array(X_batch_features)
    X_batch_scaled = scaler.transform(X_batch_np)
    
    # Fast prediction
    probs_s1 = model.predict(X_batch_scaled, batch_size=256, verbose=0).flatten()
    
    # Filter Candidates
    pass_s1_indices = np.where(probs_s1 > best_threshold1)[0]
    
    # E. Predict Stage 2 (Precision) - Only on survivors
    if len(pass_s1_indices) > 0:
        X_s2_features = []
        survivor_metadata = []
        survivor_labels = []
        
        for idx in pass_s1_indices:
            rec_a, rec_b, aid, bid = batch_pairs_metadata[idx]
            
            # Create Hybrid Features (Expensive Jaro-Winkler)
            # We only do this for the small % of pairs that passed Stage 1
            hybrid_feats = lib.create_hybrid_feature_vector(rec_a, rec_b, col="title")
            
            X_s2_features.append(hybrid_feats)
            survivor_metadata.append((rec_a, rec_b))
            survivor_labels.append(batch_labels[idx])
            
        # Predict Stage 2
        X_s2_np = np.array(X_s2_features)
        X_s2_scaled = precision_scaler.transform(X_s2_np)
        probs_s2 = precision_model.predict(X_s2_scaled, batch_size=256, verbose=0).flatten()
        
        # Final Filter
        final_indices = np.where(probs_s2 > best_threshold2)[0]
        
        # Store Results
        for final_idx in final_indices:
            # We store the pair and its true label (to calc final metrics)
            # If you just need the IDs, store (aid, bid)
            final_matches.append((survivor_metadata[final_idx], survivor_labels[final_idx]))
            
            # Save true label for global metrics (only for predicted positives)
            y_true_final.append(survivor_labels[final_idx])

# --- 4. Calculate Final Metrics ---
# Metrics based on "Predicted Positives"
# Precision = TP / (TP + FP)
true_positives = sum(y_true_final)
predicted_positives = len(y_true_final)

if predicted_positives > 0:
    precision = true_positives / predicted_positives
else:
    precision = 0.0

# Recall = TP / Total_Actual_Positives
# Note: 'matches' must be calculated from your full GT file at the start
total_actual_matches = len(gt_lookup) # Or the 'matches' variable you calculated earlier

if total_actual_matches > 0:
    recall = true_positives / total_actual_matches
else:
    recall = 0.0

f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

print(f"\n--- Final Scalable Resolution Results ---")
print(f"Pairs Found: {predicted_positives}")
print(f"True Matches Found: {int(true_positives)}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
