import pandas as pd
import numpy as np
import lib as lib
import random
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

NUM_ITERATIONS = 8
LABELS_PER_ITERATION = 1000 # The number of new labels we "buy" from the oracle
SEED_SIZE = 100

# --- Define your data paths and column names ---
PATH_RAW_A = './data/amazon_products.csv'
PATH_RAW_B = './data/walmart_products.csv'
PATH_GT = './data/truth_amazon_walmart.tsv'
ID_COL_A = 'id1'
ID_COL_B = 'id2'
#COLS_TO_USE = ['name', 'description', 'price'] # The columns to build the 'text' from
COLS_TO_USE = [ "longdescr", "shortdescr", "title"]


# --- 1. Load Data and Oracle ---
print("--- Loading Raw Data and Oracle ---")
df_a_raw = pd.read_csv(PATH_RAW_A, encoding='unicode_escape')
df_b_raw = pd.read_csv(PATH_RAW_B, encoding='unicode_escape')
df_gt = pd.read_csv(PATH_GT, sep="\t", encoding="unicode_escape", keep_default_na=False)


df_a, df_b = lib.bootstrap_embeddings_only(
       df_a_raw, df_b_raw, "source_a", "source_b", COLS_TO_USE
)


df_a['id'] = pd.to_numeric(df_a['id'], errors='coerce')
df_a.dropna(subset=['id'], inplace=True)
df_a['id'] = df_a['id'].astype(int)
df_b['id'] = pd.to_numeric(df_b['id'], errors='coerce')
df_b.dropna(subset=['id'], inplace=True)
df_b['id'] = df_b['id'].astype(int)
df_a.reset_index(drop=True, inplace=True)
df_b.reset_index(drop=True, inplace=True)


truthD = dict()
a = 0
for i, r in df_gt.iterrows():
      id_walmart = r["id2"] #id2
      id_amazon = r["id1"]  #id1
      if not id_walmart in df_b['id'].values or not id_amazon in df_a['id'].values:
             #print(f"Disregarding {id_walmart} or {id_amazon}")
             continue

      if id_amazon in truthD:
             ids = truthD[id_amazon]
             ids.append(id_walmart)
             a += 1
      else:
             truthD[id_amazon] = [id_walmart]
matches = len(truthD.keys()) + a
print("No of matches=", matches)

#gt_lookup = {
#    (str(amazon_id), str(walmart_id)) 
#    for amazon_id, walmart_id in zip(df_gt[ID_COL_A], df_gt[ID_COL_B])
#}

gt_lookup = {
     (str(key), str(value))
     for key, value_list in truthD.items()
     for value in value_list
}




# --- 2. Iteration 0 (Bootstrapping) ---
# This is the new, crucial first step.
# It creates the .pqt files and our first noisy training set.
#df_a, df_b, noisy_training_pairs = lib.bootstrap_and_get_noisy_labels(
#    df_a_raw, df_b_raw, "source_a", "source_b", COLS_TO_USE
#)

#df_a['id'] = pd.to_numeric(df_a['id'], errors='coerce')
#df_a.dropna(subset=['id'], inplace=True)
#df_a['id'] = df_a['id'].astype(int)
#df_b['id'] = pd.to_numeric(df_b['id'], errors='coerce')
#df_b.dropna(subset=['id'], inplace=True)
#df_b['id'] = df_b['id'].astype(int)
#df_a.reset_index(drop=True, inplace=True)
#df_b.reset_index(drop=True, inplace=True)

amazon_id_mapper = df_a['id'].to_dict()
walmart_id_mapper = df_b['id'].to_dict()


# Create fast lookup dicts from the NEW dataframes (which have 'v')
a_lookup = {row['text']: row for _, row in df_a.iterrows()}
b_lookup = {row['text']: row for _, row in df_b.iterrows()}



# --- 4. Create Candidate and Test Pools (The New Logic) ---
print("\n--- Creating Test and Unlabeled Pools ---")
full_candidate_pool_text = lib.get_candidate_pool(df_a, df_b, k=10)
random.shuffle(full_candidate_pool_text)

# Split the *unlabeled text*
test_set_size = int(len(full_candidate_pool_text) * 0.2)
test_pool_text = full_candidate_pool_text[:test_set_size]
unlabeled_pool_text = full_candidate_pool_text[test_set_size:]

# Call the Oracle ONCE to create the permanent, clean test set
print("Querying Oracle *once* to create the Test Set...")
test_set = lib.query_oracle(
    test_pool_text, a_lookup, b_lookup, gt_lookup, "id" , "id" 
)
print(f"Created Test Set with {len(test_set)} pairs.")

# --- 5. Iteration 0 (Seeding) ---
print(f"\n--- Iteration 0: Training CLEAN Seed Model ---")
# Get the first SEED_SIZE pairs from the unlabeled pool
seed_pairs_text = unlabeled_pool_text[:SEED_SIZE]
unlabeled_pool_text = unlabeled_pool_text[SEED_SIZE:] # Remove them

# Call the Oracle to label *only* the seed pairs
print(f"Querying Oracle for {SEED_SIZE} seed labels...")
current_clean_training_set = lib.query_oracle(
    seed_pairs_text, a_lookup, b_lookup, gt_lookup, "id" , "id"
)

print(f"Training on initial *clean* seed set of {len(current_clean_training_set)} labels.")
model, scaler, f1, best_threshold1 = lib.train_classifier(
    current_clean_training_set, test_set, a_lookup, b_lookup
)
print(f"--- Iteration 0 (Clean) F1-Score: {f1:.4f} ---")

# --- 6. The Active Learning Loop (The New Logic) ---
last_f1_score = f1
MIN_IMPROVEMENT_THRESHOLD = 0.01

for i in range(1, NUM_ITERATIONS + 1):
    print(f"\n--- Iteration {i} ---")
    
    if not unlabeled_pool_text:
        print("Unlabeled pool is empty. Stopping iteration.")
        break

    # --- a. Predict on the UNLABELED pool ---
    print(f"Predicting on {len(unlabeled_pool_text)} unlabeled pairs...")
    
    X_unlabeled_list = []
    # We must keep track of the original text pairs
    pairs_for_this_batch = [] 
    
    for (text_a, text_b) in tqdm(unlabeled_pool_text): # Use tqdm for a progress bar
        record_a = a_lookup.get(text_a)
        record_b = b_lookup.get(text_b)
        if record_a is not None and record_b is not None:
            features = lib.create_pure_embedding_vector(record_a, record_b)
            if features.shape[0] == 1536:
                X_unlabeled_list.append(features)
                pairs_for_this_batch.append((text_a, text_b)) # Store the text tuple
                
    X_unlabeled_matrix = np.array(X_unlabeled_list)
    
    if len(X_unlabeled_matrix) == 0:
        print("No valid pairs left in unlabeled pool. Stopping.")
        break
        
    X_unlabeled_scaled = scaler.transform(X_unlabeled_matrix)
    preds_prob = model.predict(X_unlabeled_scaled, batch_size=256).flatten()
    
    # --- b. Select pairs to label (Hybrid Strategy) ---
    half_batch = LABELS_PER_ITERATION // 2
    
    # 1. "Most confused"
    confidence = np.abs(preds_prob - 0.5)
    most_confused_indices = np.argsort(confidence)[:half_batch]
    
    # 2. "Most confident"
    most_confident_indices = np.argsort(preds_prob)[-half_batch:]
    
    indices_to_label = np.unique(np.concatenate([most_confused_indices, most_confident_indices]))
    
    # Get the *text pairs* to send to the oracle
    pairs_to_label_text = [pairs_for_this_batch[idx] for idx in indices_to_label]
    
    # --- c. Query the Oracle *inside the loop* ---
    print(f"Querying Oracle for {len(pairs_to_label_text)} new pairs...")
    newly_labeled_pairs = lib.query_oracle(
        pairs_to_label_text, a_lookup, b_lookup, gt_lookup, "id" , "id"
    )
    current_clean_training_set.extend(newly_labeled_pairs)
    
    # --- d. Remove from unlabeled pool ---
    # This is now a simple set difference on the text tuples
    unlabeled_pool_text = list(set(unlabeled_pool_text) - set(pairs_to_label_text))
    
    # --- e. Re-train the model ---
    total_labels = len(current_clean_training_set)
    num_positives = sum(1 for p in current_clean_training_set if p[2] == 1.0)
    num_negatives = total_labels - num_positives
    
    print(f"Re-training model on {total_labels} total clean labels:")
    print(f"  - Positives (Matches):    {num_positives}")
    print(f"  - Negatives (No Matches): {num_negatives}")

    model, scaler, f1, best_threshold1 = lib.train_classifier(
        current_clean_training_set, test_set, a_lookup, b_lookup
    )
    
    print(f"--- Iteration {i} F1-Score: {f1:.4f} ---")
    
    improvement = f1 - last_f1_score
    if improvement < MIN_IMPROVEMENT_THRESHOLD and i > 1:
        print(f"\nF1-Score improved by only {improvement:.4f}. Stopping Active Learning loop early.")
        break
    last_f1_score = f1

print("\nActive Learning loop complete.")
print(f"Final Recall Model F1-Score: {f1:.4f} at threshold {best_threshold1:.4f}")



print("--- Starting Phase 2: Training Final Precision Model ---")

# Use your new functions to train the hybrid-feature model
# It trains on the *same* clean set, but uses *more features*
precision_model, precision_scaler, best_f1, best_threshold2 = lib.train_precision_classifier(
     current_clean_training_set,  # Your full, clean training set
     test_set,                    # Your held-out test set
     a_lookup,                    # Your lookup dict (needs 'v' and 'name')
     b_lookup,                     # Your lookup dict (needs 'v' and 'name')
     col="title"    
)
print(f"--- Precision Model Trained! ---")
print(f"Final F1-Score: {best_f1:.4f} at threshold {best_threshold2:.4f}")




amazon_embeddings = np.array(df_a['v'].tolist()).astype(np.float32)
walmart_embeddings = np.array(df_b['v'].tolist()).astype(np.float32)
#amazon_id_to_index = {id_val: index for index, id_val in amazon_id_mapper.items()}
#walmart_id_to_index = {id_val: index for index, id_val in walmart_id_mapper.items()}




print("\n--- 4. Building Faiss index for Walmart records ---")
d = walmart_embeddings.shape[1]
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 60
index.hnsw.efSearch = 64
index.add(walmart_embeddings)
print(f"Index built successfully with {index.ntotal} records.")


gt_lookup = {
    (str(amazon_id), str(walmart_id))
    for amazon_id, walmart_id_list in truthD.items()
    for walmart_id in walmart_id_list
}
print(f"Created gt_lookup set with {len(gt_lookup)} total matching pairs.")

#df_a, df_b, training_examples = fine_tune_on_ground_truth(df_amazon, df_walmart, text_columns, gt_lookup, id_col_a="id", id_col_b="id", model_path='./data/amazon_walmart_gt_wsss_ft_model')
#train_classifier(df_a, df_b,  training_examples, name="amazon_walmart")
#model_path = '/content/drive/MyDrive/data/gt_mlp_classifier_amazon_walmart.keras'
#scaler_path = '/content/drive/MyDrive/data/gt_mlp_scaler_amazon_walmart.joblib'
#classifier = tf.keras.models.load_model(model_path)
#scaler = joblib.load(scaler_path)





k = 5 # The number of top matches to retrieve for each query

D, I = index.search(amazon_embeddings, k)

true_positives = 0
false_positives = 0
X_predict = []
candidate_pairs_list = [] # Store the (record_a, record_b)
y_true_list = []          # Store the true labels for final evaluation
stage1_pairs_data = []
X_stage1_features = []


for amazon_idx, walmart_matches in enumerate(I):
    amazon_id = amazon_id_mapper[amazon_idx]
    amazon_embedding = amazon_embeddings[amazon_idx]
    amazon_record = df_a.iloc[amazon_idx] # Get the full record
    for i, walmart_idx in enumerate(walmart_matches):
       walmart_id = walmart_id_mapper[walmart_idx]
       walmart_embedding = walmart_embeddings[walmart_idx]
       walmart_record = df_b.iloc[walmart_idx] # Get the full record
       if amazon_id in truthD:
            v1 = {"v": amazon_embedding}
            v2 = {"v": walmart_embedding}
            feature_vector = lib.create_pure_embedding_vector(v2 , v1 )
            X_predict.append(feature_vector)
            candidate_pairs_list.append((v2, v1))

            stage1_pairs_data.append((amazon_record, walmart_record))
            X_stage1_features.append(feature_vector)


            if walmart_id in truthD[amazon_id]:
                y_true_list.append(1)
            else:
                y_true_list.append(0)

X_matrix = np.array(X_predict)
X_scaled = scaler.transform(X_matrix)
all_probabilities = model.predict(X_scaled, batch_size=256)
all_decisions_binary = (all_probabilities.flatten() > best_threshold1).astype(int)
y_true_array = np.array(y_true_list)

from sklearn.metrics import f1_score, precision_score, recall_score
f1 = f1_score(y_true_array, all_decisions_binary)
recall = recall_score(y_true_array, all_decisions_binary)
precision = precision_score(y_true_array, all_decisions_binary)

print(f"F1-Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")






stage1_probs = model.predict(X_scaled, batch_size=256).flatten()

# --- STAGE 1 FILTERING ---
# Find all pairs that pass our low-recall threshold
stage2_candidate_indices = np.where(stage1_probs > best_threshold1 )[0]
print(f"Stage 1 found {len(stage2_candidate_indices)} high-recall candidates.")

# --- STAGE 2: PRECISION FILTERING ---
if len(stage2_candidate_indices) > 0:
    X_stage2_features = []
    y_true_list = [] # For final scoring

    print("Running Stage 2 (Smart Precision)...")
    for idx in stage2_candidate_indices:
        # Get the record pair we saved
        amazon_record, walmart_record = stage1_pairs_data[idx]

        # Create the SLOW, HYBRID (embedding + jaro) feature vector
        hybrid_features = lib.create_hybrid_feature_vector(amazon_record, walmart_record, col="title")
        X_stage2_features.append(hybrid_features)

        # Get the true label for our final test
        amazon_id = str(amazon_record["id"])
        walmart_id = str(walmart_record["id"])
        y_true_list.append(1.0 if (amazon_id, walmart_id) in gt_lookup else 0.0)

    # --- STAGE 2 PREDICTION ---
    X_stage2_matrix = np.array(X_stage2_features)
    X_stage2_scaled = precision_scaler.transform(X_stage2_matrix)
    stage2_probs = precision_model.predict(X_stage2_scaled, batch_size=256).flatten()

    # Use the high, optimal threshold from the precision model
    stage2_decisions = (stage2_probs > best_threshold2).astype(int)
    y_true_array = np.array(y_true_list)

    # --- FINAL RESULTS ---
    print("\n--- Final Two-Stage Model Performance ---")
    print(f"(Using {len(stage2_decisions)} pairs from Stage 1)")
    f1 = f1_score(y_true_array, stage2_decisions)
    recall = recall_score(y_true_array,  stage2_decisions)
    precision = precision_score(y_true_array,  stage2_decisions)
    print(f"F1-score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")




