import pandas as pd
import numpy as np
import lib as lib
import random
import faiss
import math
import heapq
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# --- 1. Configuration & Setup ---

# --- Scalability Settings (WRS-ALER) ---
NUM_ITERATIONS = 9          # Replaces 3 partitions * 3 iterations
LABELS_PER_ITERATION = 300 
SEED_SIZE = 100              

# --- Validation Set Configuration ---
VAL_SET_PROPORTION = 0.1
VAL_SET_MAX_SIZE = 1000  

# --- WRS-ALER Specific Parameters ---
ALPHA = 0.5                 # Split ratio for exploration/exploitation reservoirs
ETA_MULTIPLIER = 0.25       # Determines the base capacity eta
K_NEIGHBORS = 10            # Top-k neighbors to retrieve
EPSILON = 1e-6              # Smoothing term
GAMMA = 0.01                # Dynamic truncation threshold

# --- Data paths and columns ---
PATH_RAW_A = './data/imdb.csv'
PATH_RAW_B = './data/dbpedia.csv'
PATH_GT = './data/truth_imdb_dbpedia.csv'
ID_COL_A = 'D1'
ID_COL_B = 'D2'
COLS_TO_USE = ["title", "starring"] 

print("--- Loading Raw Data and Oracle ---")

df_a_raw = pd.read_csv(PATH_RAW_A, sep="|", encoding='utf-8')
df_b_raw = pd.read_csv(PATH_RAW_B, sep="|", encoding='utf-8')
df_a = df_a_raw.dropna(subset=['title']).copy()
df_b = df_b_raw.dropna(subset=['title']).copy()
df_a['id'] = pd.to_numeric(df_a['id'], errors='coerce')
df_b['id'] = pd.to_numeric(df_b['id'], errors='coerce')

df_gt = pd.read_csv(PATH_GT, sep="|", encoding="utf-8", keep_default_na=False)
valid_d1_ids = set(df_a['id'].values)
valid_d2_ids = set(df_b['id'].values)
mask_to_keep = df_gt['D1'].isin(valid_d1_ids) & df_gt['D2'].isin(valid_d2_ids)
df_gt = df_gt[mask_to_keep].copy()

truthD = dict()
a = 0
for i, r in df_gt.iterrows():
        idimdb = r["D1"]
        iddbpedia = r["D2"]
        if not idimdb in df_a['id'].values or not iddbpedia in df_b['id'].values:
            continue
        if idimdb in truthD:
            truthD[idimdb].append(iddbpedia)
            a += 1
        else:
            truthD[idimdb] = [iddbpedia]

matches = len(truthD.keys()) + a
print("No of matches=", matches)

gt_lookup = {
     (str(key), str(value))
     for key, value_list in truthD.items()
     for value in value_list
}
print(f"Loaded Oracle with {len(gt_lookup)} total matches.")

# --- 2. Bootstrap Embeddings (Phase 1) ---
df_a, df_b = lib.bootstrap_embeddings_only(
       df_a, df_b, "source_a", "source_b", COLS_TO_USE
)

a_embeddings = np.array(df_a['v'].tolist()).astype('float32')
b_embeddings = np.array(df_b['v'].tolist()).astype('float32')
df_a_whole = df_a.copy()
df_b_whole = df_b.copy()

# WRS-ALER evaluates the space dynamically, no static downsampling needed
SAMPLE_PROPORTION = 1.0 

a_lookup = {row['text']: row for _, row in df_a.iterrows()}
b_lookup = {row['text']: row for _, row in df_b.iterrows()}

print("Building global FAISS index (IMDB)...")
d = a_embeddings.shape[1]
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 60
index.hnsw.efSearch = 64
faiss.normalize_L2(a_embeddings) 
index.add(a_embeddings)
faiss.normalize_L2(b_embeddings)

# --- 3. WRS-ALER: Dynamic Probabilistic Filtering ---
print("\n--- Executing WRS-ALER Dynamic Filtering ---")

# Define reservoir capacities
ETA = int(ETA_MULTIPLIER * len(b_embeddings) * K_NEIGHBORS)
ETA_B = int(ALPHA * ETA)
ETA_C = int((1 - ALPHA) * ETA)
print(f"Reservoir Capacities - Total: {ETA}, R_b: {ETA_B}, R_c: {ETA_C}")

R_b = [] # Min-priority queue for uncertain pairs (boundary)
R_c = [] # Min-priority queue for high-confidence pairs (core)

ins = 0
no = 0

time_start_filtering = time.time()

# Process row by row for dynamic truncation
for i in tqdm(range(len(b_embeddings)), desc="Querying DBPedia Candidates"):
    query_vec = b_embeddings[i:i+1]
    D, I = index.search(query_vec, k=K_NEIGHBORS)
    
    inserted_this_step = False
    
    for j in range(K_NEIGHBORS):
        a_idx = I[0][j]
        sim = D[0][j]
        
        # Normalize cosine similarity roughly to [0, 1]
        xi = (sim + 1.0) / 2.0
        xi = max(EPSILON, min(1.0 - EPSILON, xi)) 
        
        # Calculate Information Entropies / Weights
        w_b = -xi * math.log(xi) - (1.0 - xi) * math.log(1.0 - xi) + EPSILON
        w_c = abs(xi - 0.5) + EPSILON
        
        u = random.uniform(0, 1)
        tau_b = math.pow(u, 1.0 / w_b)
        tau_c = math.pow(u, 1.0 / w_c)
        
        # Evaluate for Boundary Reservoir R_b
        if len(R_b) < ETA_B or tau_b > R_b[0][0]:
            if len(R_b) == ETA_B:
                heapq.heappop(R_b)
            heapq.heappush(R_b, (tau_b, a_idx, i))
            inserted_this_step = True
            
        # Evaluate for Core Reservoir R_c
        if len(R_c) < ETA_C or tau_c > R_c[0][0]:
            if len(R_c) == ETA_C:
                heapq.heappop(R_c)
            heapq.heappush(R_c, (tau_c, a_idx, i))
            inserted_this_step = True
            
    no += K_NEIGHBORS
    if inserted_this_step:
        ins += 1
        
    # Dynamic Truncation Evaluation
    if len(R_b) == ETA_B and len(R_c) == ETA_C:
        insertion_rate = ins / no
        if insertion_rate < GAMMA and no > ETA:
            print(f"\nDynamic truncation triggered! Stopping at record {i}/{len(b_embeddings)}")
            print(f"Final Insertion Rate: {insertion_rate:.4f} < {GAMMA}")
            break

# Construct the highly informative bounded candidate pool C
candidate_pool_text = set()
for _, a_idx, b_idx in R_b + R_c:
    text_a = df_a.iloc[a_idx]['text']
    text_b = df_b.iloc[b_idx]['text']
    candidate_pool_text.add((text_a, text_b))

print(f"Filtering Time: {time.time() - time_start_filtering:.2f} seconds.")
print(f"Formulated Bounded Candidate Pool (C) size: {len(candidate_pool_text)}")


# --- 4. Active Learning Loop ---
time_start_training = time.time()
master_clean_training_set = []

# Label the pool to create training and validation splits
labeled_pool = lib.query_oracle(
    list(candidate_pool_text), a_lookup, b_lookup, gt_lookup, "id", "id"
)
random.shuffle(labeled_pool)

# Extract Validation Set
val_set_size = int(len(labeled_pool) * VAL_SET_PROPORTION)
val_set_size = min(val_set_size, VAL_SET_MAX_SIZE)
fast_validation_set = labeled_pool[:val_set_size]
print(f"Created a global, fixed validation set of {val_set_size} pairs.")

# Remaining is the unlabeled pool
unlabeled_pool_text = [p[:2] for p in labeled_pool[val_set_size:]]

# Extract Initial Seed
seed_pairs_text = unlabeled_pool_text[:SEED_SIZE]
unlabeled_pool_text = unlabeled_pool_text[SEED_SIZE:]
master_clean_training_set = lib.query_oracle(
    seed_pairs_text, a_lookup, b_lookup, gt_lookup, "id", "id"
)

# Loop Execution
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
    
    # Hybrid Query Strategy
    half_batch = LABELS_PER_ITERATION // 2
    confidence = np.abs(preds_prob - 0.5)
    
    most_confused_indices = np.argsort(confidence)[:half_batch]
    most_confident_indices = np.argsort(preds_prob)[-half_batch:]
    indices_to_label = np.unique(np.concatenate([most_confused_indices, most_confident_indices]))
    
    pairs_to_label_text = [pairs_for_this_batch[idx] for idx in indices_to_label]
    
    # Label & Add to Master
    newly_labeled_pairs = lib.query_oracle(
        pairs_to_label_text, a_lookup, b_lookup, gt_lookup, "id", "id"
    )
    master_clean_training_set.extend(newly_labeled_pairs)
    
    # Remove queried pairs
    unlabeled_pool_text = list(set(unlabeled_pool_text) - set(pairs_to_label_text))


print("\n--- Training Phase Complete ---")
total_labels = len(master_clean_training_set)
num_positives = sum(1 for p in master_clean_training_set if p[2] == 1.0)
num_negatives = total_labels - num_positives

print(f"Total Labels Collected: {total_labels}")
print(f"  - Matches (TP): {num_positives}")
print(f"  - Non-Matches (TN): {num_negatives}")


# --- 5. Train Final Master Models ---
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
print(f"Total Training Time: {time_end_training - time_start_training:.2f} seconds.")


# --- 6. Final Two-Stage Resolution ---
time_start_res = time.time()
print("\n--- Starting Final Two-Stage Resolution (Held-Out Test Set) ---")

# Apply leakage fix: Extract ONLY the strictly labeled IMDB records to avoid blacklisting the whole DB
labeled_pairs = master_clean_training_set + fast_validation_set
train_val_ids = {str(a_lookup[text_a]['id']) for (text_a, _, _) in labeled_pairs}

# Re-build index for evaluation on full a_embeddings (IMDB)
d = a_embeddings.shape[1]
eval_index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
eval_index.hnsw.efConstruction = 60
eval_index.hnsw.efSearch = 64
eval_index.add(a_embeddings)

# Search DBPedia against the full IMDB index
D, I = eval_index.search(b_embeddings, k=5)

X_stage1_features = []
stage1_pairs_data = [] 
y_true_list_stage1 = [] 

print("Running Stage 1 on HELD-OUT TEST SET only...")
for b_idx, a_indices in enumerate(I):
    b_record = df_b_whole.iloc[b_idx]

    for a_idx in a_indices:
        a_record = df_a_whole.iloc[a_idx]
        
        if str(a_record['id']) in train_val_ids:
            continue

        stage1_pairs_data.append((a_record, b_record))
        X_stage1_features.append(lib.create_pure_embedding_vector(a_record, b_record))

        a_id = str(a_record["id"])
        b_id = str(b_record["id"])
        is_match = 1.0 if (a_id, b_id) in gt_lookup else 0.0
        y_true_list_stage1.append(is_match)

# Predict Stage 1
if len(X_stage1_features) == 0:
    print("Error: No candidates survived the leakage filter for Stage 1.")
else:
    X_stage1_matrix = np.array(X_stage1_features)
    X_stage1_scaled = scaler.transform(X_stage1_matrix)
    stage1_probs = model.predict(X_stage1_scaled, batch_size=256).flatten()
    stage1_decisions = (stage1_probs > best_threshold1).astype(int)
    y_true_stage1_array = np.array(y_true_list_stage1)

    print("\n--- Stage 1 (Recall Model) Classifier Performance ---")
    f1 = f1_score(y_true_stage1_array, stage1_decisions)
    recall = recall_score(y_true_stage1_array,  stage1_decisions)
    precision = precision_score(y_true_stage1_array,  stage1_decisions)
    print(f"F1-score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")

    stage2_candidate_indices = np.where(stage1_probs > best_threshold1)[0]
    print(f"\nStage 1 passed {len(stage2_candidate_indices)} candidates to Stage 2.")

    if len(stage2_candidate_indices) > 0:
        X_stage2_features = []
        y_true_list_stage2 = y_true_stage1_array[stage2_candidate_indices]

        for idx in stage2_candidate_indices:
            a_record, b_record = stage1_pairs_data[idx]
            hybrid_features = lib.create_hybrid_feature_vector(a_record, b_record, col="title")
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
        print(f"Classifier Recall: {rec:.4f} (Of candidates seen)")
        print(f"Classifier Precision: {prec:.4f}")

    else:
        print("Stage 1 found no candidates.")

real_total_labels = len(master_clean_training_set) + len(fast_validation_set)
active_queries = len(master_clean_training_set) - SEED_SIZE

print("\n" + "="*30)
print("PAPER TABLE 2 DATA (IMDB-DBPEDIA)")
print("="*30)
print(f"Seed (B_seed): {SEED_SIZE}") 
print(f"Valid (|V|):   {len(fast_validation_set)}")
print(f"Active Loop:   {active_queries}") 
print(f"Total Budget:  {real_total_labels}")
print("="*30)
