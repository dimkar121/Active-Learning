import pandas as pd
import numpy as np
import lib as lib
import random
import faiss


NUM_ITERATIONS = 3
LABELS_PER_ITERATION = 1000 # The number of new labels we "buy" from the oracle
SEED_SIZE = 100

# --- Define your data paths and column names ---
PATH_RAW_A = './data/Scholar.csv'
PATH_RAW_B = './data/DBLP2.csv'
PATH_GT = './data/truth_Scholar_DBLP.csv'
ID_COL_A = 'idScholar'
ID_COL_B = 'idDBLP'
COLS_TO_USE = ['title', 'authors', 'venue', 'year'] # The columns to build the 'text' from


# --- 1. Load Data and Oracle ---
print("--- Loading Raw Data and Oracle ---")
df_a_raw = pd.read_csv(PATH_RAW_A, encoding='utf-8')
df_b_raw = pd.read_csv(PATH_RAW_B, encoding='utf-8')
df_gt = pd.read_csv(PATH_GT, encoding="unicode_escape", keep_default_na=False)

gt_lookup = {
    (str(id1), str(id2)) 
    for id1, id2 in zip(df_gt[ID_COL_A], df_gt[ID_COL_B])
}

df_a, df_b = lib.bootstrap_embeddings_only(
      df_a_raw, df_b_raw, "source_a", "source_b", COLS_TO_USE
)



# --- 2. Iteration 0 (Bootstrapping) ---
# This is the new, crucial first step.
# It creates the .pqt files and our first noisy training set.
#df_a, df_b, noisy_training_pairs = lib.bootstrap_and_get_noisy_labels(
#    df_a_raw, df_b_raw, "source_a", "source_b", COLS_TO_USE
#)


scholar_id_mapper = df_a['id'].to_dict()
dblp_id_mapper = df_b['id'].to_dict()


# Create fast lookup dicts from the NEW dataframes (which have 'v')
a_lookup = {row['text']: row for _, row in df_a.iterrows()}
b_lookup = {row['text']: row for _, row in df_b.iterrows()}

# --- 3. Create Candidate and Test Pools ---
print("\n--- Creating Test and Unlabeled Pools ---")
full_candidate_pool_text = lib.get_candidate_pool(df_a, df_b, k=10)

# Label this entire pool *once* so we can create our sets
print("Querying Oracle to create Test/Unlabeled pools...")
full_labeled_pool = lib.query_oracle(
    full_candidate_pool_text, a_lookup, b_lookup, gt_lookup, "id" , "id" 
)
random.shuffle(full_labeled_pool)

# Split into a Test set (to evaluate models) and an Unlabeled Pool (to query)
test_set_size = int(len(full_labeled_pool) * 0.2)
test_set = full_labeled_pool[:test_set_size]
unlabeled_pool = full_labeled_pool[test_set_size:]

print(f"Created Test Set with {len(test_set)} pairs.")
print(f"Created Unlabeled Pool with {len(unlabeled_pool)} pairs.")


print(f"\n--- Iteration 0: Training CLEAN Seed Model ---")
# We take our first "seed" labels from the unlabeled pool
current_clean_training_set = unlabeled_pool[:SEED_SIZE]
unlabeled_pool = unlabeled_pool[SEED_SIZE:] # Remove them

print(f"Training on initial *clean* seed set of {len(current_clean_training_set)} labels.")
model, scaler, f1 = lib.train_classifier(current_clean_training_set, test_set, a_lookup, b_lookup)
print(f"--- Iteration 0 (Clean) F1-Score: {f1:.4f} ---")




# This is our set of clean, oracle-verified labels. It starts empty.
current_clean_training_set = []

# --- 5. The Active Learning Loop ---
for i in range(1, NUM_ITERATIONS + 1):
    print(f"\n--- Iteration {i} ---")
    
    if not unlabeled_pool:
        print("Unlabeled pool is empty. Stopping iteration.")
        break

    # --- a. Predict on the unlabeled pool ---
    print(f"Predicting on {len(unlabeled_pool)} unlabeled pairs...")
    
    X_unlabeled_list = []
    pairs_for_this_batch = []
    
    for pair in unlabeled_pool:
        record_a = a_lookup.get(pair[0])
        record_b = b_lookup.get(pair[1])
        if record_a is not None and record_b is not None:
            features = lib.create_pure_embedding_vector(record_a, record_b)
            if features.shape[0] == 1536:
                X_unlabeled_list.append(features)
                pairs_for_this_batch.append(pair)
                
    X_unlabeled_matrix = np.array(X_unlabeled_list)
    
    if len(X_unlabeled_matrix) == 0:
        print("No valid pairs left in unlabeled pool. Stopping.")
        break
        
    X_unlabeled_scaled = scaler.transform(X_unlabeled_matrix)
    preds_prob = model.predict(X_unlabeled_scaled).flatten()
    
    # --- b. Select pairs to label ---
    # We query the pairs the model is *most confident* are matches
    most_confident_indices = np.argsort(preds_prob)[-LABELS_PER_ITERATION:]
    pairs_to_label = [pairs_for_this_batch[idx] for idx in most_confident_indices]
    
    # --- c. "Query the Oracle" and add to training set ---
    # `pairs_to_label` already has the correct label
    new_labels_found = len(pairs_to_label)
    print(f"Adding {new_labels_found} new *clean* labels to training set.")
    current_clean_training_set.extend(pairs_to_label)
    
    # --- d. Remove these new labels from the unlabeled pool ---
    unlabeled_pool_set = set(unlabeled_pool)
    pairs_to_label_set = set(pairs_to_label)
    unlabeled_pool = list(unlabeled_pool_set - pairs_to_label_set)
    
    # --- e. Re-train the model on ALL clean labels found so far ---

    total_labels = len(current_clean_training_set)
    
    # The label is the 3rd element in the tuple (index 2)
    num_positives = sum(1 for pair in current_clean_training_set if pair[2] == 1.0)
    num_negatives = total_labels - num_positives
    
    print(f"Re-training model on {total_labels} total clean labels:")
    print(f"  - Positives (Matches):    {num_positives}")
    print(f"  - Negatives (No Matches): {num_negatives}")

    print(f"Re-training model on {len(current_clean_training_set)} total clean labels.")
    model, scaler, f1 = lib.train_classifier(current_clean_training_set, test_set, a_lookup, b_lookup)
    
    print(f"--- Iteration {i} F1-Score: {f1:.4f} ---")

print("\nActive Learning loop complete.")



scholar_embeddings = np.array(df_a['v'].tolist()).astype(np.float32)
dblp_embeddings = np.array(df_b['v'].tolist()).astype(np.float32)
#amazon_id_to_index = {id_val: index for index, id_val in amazon_id_mapper.items()}
#walmart_id_to_index = {id_val: index for index, id_val in walmart_id_mapper.items()}

print("\n--- 4. Building Faiss index for Walmart records ---")
d = scholar_embeddings.shape[1]
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 60
index.hnsw.efSearch = 64
index.add(scholar_embeddings)
print(f"Index built successfully with {index.ntotal} records.")



truthD = dict()
a = 0
for i, r in df_gt.iterrows():
      idDBLP = r["idDBLP"]
      idScholar = r["idScholar"]
      if idScholar in truthD:
          ids = truthD[idScholar]
          ids.append(idDBLP)
          a += 1
      else:
          truthD[idScholar] = [idDBLP]
          a += 1
matches = a
print("No of matches=", matches)

gt_lookup = {
    (str(id1), str(id2))
    for id1, id_list in truthD.items()
    for id2 in id_list
}
print(f"Created gt_lookup set with {len(gt_lookup)} total matching pairs.")

#df_a, df_b, training_examples = fine_tune_on_ground_truth(df_amazon, df_walmart, text_columns, gt_lookup, id_col_a="id", id_col_b="id", model_path='./data/amazon_walmart_gt_wsss_ft_model')
#train_classifier(df_a, df_b,  training_examples, name="amazon_walmart")
#model_path = '/content/drive/MyDrive/data/gt_mlp_classifier_amazon_walmart.keras'
#scaler_path = '/content/drive/MyDrive/data/gt_mlp_scaler_amazon_walmart.joblib'
#classifier = tf.keras.models.load_model(model_path)
#scaler = joblib.load(scaler_path)





k = 5 # The number of top matches to retrieve for each query

D, I = index.search(dblp_embeddings, k)

true_positives = 0
false_positives = 0
X_predict = []
candidate_pairs_list = [] # Store the (record_a, record_b)
y_true_list = []          # Store the true labels for final evaluation

for dblp_idx, scholar_matches in enumerate(I):
    dblp_id = dblp_id_mapper[dblp_idx]
    dblp_embedding = dblp_embeddings[dblp_idx]

    for i, scholar_idx in enumerate(scholar_matches):
       scholar_id = scholar_id_mapper[scholar_idx]
       scholar_embedding = scholar_embeddings[scholar_idx]
       if scholar_id in truthD:
            v1 = {"v": scholar_embedding}
            v2 = {"v": dblp_embedding}
            feature_vector = lib.create_pure_embedding_vector(v2 , v1 )
            X_predict.append(feature_vector)
            candidate_pairs_list.append((v2, v1))
            if dblp_id in truthD[scholar_id]:
                y_true_list.append(1)
            else:
                y_true_list.append(0)

X_matrix = np.array(X_predict)
X_scaled = scaler.transform(X_matrix)
all_probabilities = model.predict(X_scaled, batch_size=256)
all_decisions_binary = (all_probabilities.flatten() > 0.21).astype(int)
y_true_array = np.array(y_true_list)

from sklearn.metrics import f1_score, precision_score, recall_score
f1 = f1_score(y_true_array, all_decisions_binary)
recall = recall_score(y_true_array, all_decisions_binary)
precision = precision_score(y_true_array, all_decisions_binary)

print(f"F1-Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")









