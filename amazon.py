import pandas as pd
import numpy as np
import lib as lib
import random
import faiss
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.cluster import MiniBatchKMeans

# --- 1. Configuration & Setup ---

# --- Scalability Settings ---
N_PARTITIONS = 4 # Split the dataset into 20 chunks
NUM_ITERATIONS_PER_PARTITION = 8 # Run 3 AL iterations on each chunk
LABELS_PER_ITERATION = 200 # Query 200 labels per iteration
SEED_SIZE = 100             # Seed each partition's loop with 20 labels

# --- NEW: Validation Set Configuration ---
# We create one fixed, fast validation set.
# We'll make it proportional to the *first* partition's candidate pool,
# but cap it to ensure it's always fast.
VAL_SET_PROPORTION = 0.1
VAL_SET_MAX_SIZE = 20000  # Cap at 20,000 pairs

# --- Data paths and columns ---

PATH_RAW_A = './data/Amazon.csv'
PATH_RAW_B = './data/GoogleProducts.csv'
PATH_GT = './data/truth_amazon_google.csv'
ID_COL_A = 'idAmazon'
ID_COL_B = 'idGoogle'
COLS_TO_USE = ["name","description","manufacturer"] # The columns to build the 'text' from


# --- 1. Load Data and Oracle ---
print("--- Loading Raw Data and Oracle ---")
df_a_raw = pd.read_csv(PATH_RAW_A, encoding='unicode_escape')
df_b_raw = pd.read_csv(PATH_RAW_B, encoding='unicode_escape')
df_gt = pd.read_csv(PATH_GT, encoding="unicode_escape", keep_default_na=False)


truthD = dict()
a = 0
for i, r in df_gt.iterrows():
        idAmazon = r["idAmazon"]
        idGoogle = r["idGoogle"]
        if idAmazon in truthD.keys():
            ids = truthD[idAmazon]
            ids.append(idGoogle)
            a += 1
        else:
            truthD[idAmazon] = [idGoogle]
            
matches = len(truthD.keys()) + a
print("No of matches=", matches)


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


google_embeddings = np.array(df_b['v'].tolist()).astype('float32')
df_b_whole  = df_b
SAMPLE_PROPORTION = 0.4
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
faiss.normalize_L2(embeddings_a) # Normalize for inner product (cosine sim)
index.add(embeddings_a)


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
        
    # 5. Run the "mini" Active Learning Loop for this partitiona

    
 
    MIN_IMPROVEMENT_THRESHOLD = 0.005
    PATIENCE = 2  # Wait for 2 *consecutive* iterations of no improvement
    patience_counter = 0
    last_f1_score = 0.0
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
        improvement = f1 - last_f1_score
        if improvement < MIN_IMPROVEMENT_THRESHOLD and j > 1:
           patience_counter += 1
           print(f"F1-Score did not improve. Patience counter: {patience_counter}/{PATIENCE}")
           if patience_counter >= PATIENCE:
              print(f"\nF1-Score has not improved for {PATIENCE} iterations. Stopping loop early.")
              break
        else:
          # If there *is* improvement, reset the patience
            patience_counter = 0
            last_f1_score = f1
        
        # --- Predict on this partition's unlabeled pool ---
        if not unlabeled_pool_text_partition:
            print("no unlabeled pool partition")
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
            print("no x_unlabeled list")
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
            print("no indices left to label")
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

total_labels = len(master_clean_training_set)
num_positives = 0
num_negatives = 0

for pair_tuple in master_clean_training_set:
    label = pair_tuple[2]  # Get the label
    if label == 1.0:
        num_positives += 1
    else:
        num_negatives += 1

print(f"--- Final Master Training Set Stats ---")
print(f"Total Labels Collected: {total_labels}")
print(f"  - Positives (Matches):    {num_positives}")
print(f"  - Negatives (No Matches): {num_negatives}")


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
         col="name"
    )
    print(f"Master Precision Model F1-Score: {best_f1:.4f}")
else:
    print("No clean labels gathered, skipping precision model.")


# --- 8. Phase 4: Final Two-Stage Resolution ---
# (This section is the same as your scholar.py script)
# ...


print("\n--- Starting Final Two-Stage Resolution ---")


# 2. Search with all of df_b (DBLP)
faiss.normalize_L2(google_embeddings)
D, I = index.search(google_embeddings, k=5)

# --- 3. Stage 1 (Recall Filter) ---
X_stage1_features = []
stage1_pairs_data = [] # Store (scholar_record, dblp_record)
y_true_list_stage1 = [] # Store all true labels for a *full* recall calculation

print("Running Stage 1 (Fast Recall)...")
for google_idx, amazon_indices in enumerate(I):
    google_record = df_b_whole.iloc[google_idx]

    for amazon_idx in amazon_indices:
        amazon_record = df_a.iloc[amazon_idx]

        # Add to lists
        stage1_pairs_data.append((amazon_record, google_record))
        X_stage1_features.append(lib.create_pure_embedding_vector(amazon_record, google_record))

        # Get true label for this pair
        amazon_id = str(amazon_record["id"])
        google_id = str(google_record["id"])
        y_true_list_stage1.append(1.0 if (amazon_id, google_id) in gt_lookup else 0.0)

X_stage1_matrix = np.array(X_stage1_features)
X_stage1_scaled = scaler.transform(X_stage1_matrix)
stage1_probs = model.predict(X_stage1_scaled, batch_size=256).flatten()
stage1_decisions = (stage1_probs > best_threshold1).astype(int)
y_true_stage1_array = np.array(y_true_list_stage1)

print("\n--- Stage 1 (Recall Model) Performance on ALL Candidates ---")
#print(classification_report(y_true_stage1_array, stage1_decisions, target_names=["No Match", "Match"]))
f1 = f1_score(y_true_stage1_array, stage1_decisions)
recall = recall_score(y_true_stage1_array,  stage1_decisions)
precision = precision_score(y_true_stage1_array,  stage1_decisions)
print(f"F1-score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")


# --- 4. Stage 2 (Precision Filter) ---
stage2_candidate_indices = np.where(stage1_probs > best_threshold1)[0]
print(f"Stage 1 found {len(stage2_candidate_indices)} high-recall candidates.")

if len(stage2_candidate_indices) > 0:
    X_stage2_features = []
    y_true_list_stage2 = [] # For final scoring

    print("Running Stage 2 (Smart Precision)...")
    for idx in stage2_candidate_indices:
        amazon_record, google_record = stage1_pairs_data[idx]

        # Create the SLOW, HYBRID feature vector
        hybrid_features = lib.create_hybrid_feature_vector(amazon_record, google_record, col="name")
        X_stage2_features.append(hybrid_features)

        # Get the true label
        y_true_list_stage2.append(y_true_list_stage1[idx]) # Get label from our pre-built list

    # --- STAGE 2 PREDICTION ---
    X_stage2_matrix = np.array(X_stage2_features)
    X_stage2_scaled = precision_scaler.transform(X_stage2_matrix)
    stage2_probs = precision_model.predict(X_stage2_scaled, batch_size=256).flatten()

    stage2_decisions = (stage2_probs > best_threshold2).astype(int)
    y_true_stage2_array = np.array(y_true_list_stage2)

    # --- FINAL RESULTS ---
    print("\n--- Final Two-Stage Model Performance (on Stage 1 Candidates) ---")
    #print(classification_report(y_true_stage2_array, stage2_decisions, target_names=["No Match", "Match"]))
    f1 = f1_score(y_true_stage2_array, stage2_decisions)
    recall = recall_score(y_true_stage2_array,  stage2_decisions)
    precision = precision_score(y_true_stage2_array,  stage2_decisions)
    print(f"F1-score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")


else:
    print("Stage 1 found no candidates.")

