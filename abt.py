import pandas as pd
import numpy as np
import lib as lib
import random
import faiss
from sklearn.metrics import f1_score, precision_score, recall_score


NUM_ITERATIONS = 8
LABELS_PER_ITERATION = 1000 # The number of new labels we "buy" from the oracle
SEED_SIZE = 100

# --- Define your data paths and column names ---
PATH_RAW_A = './data/Abt.csv'
PATH_RAW_B = './data/Buy.csv'
PATH_GT = './data/truth_abt_buy.csv'
ID_COL_A = 'idAbt'
ID_COL_B = 'idBuy'
COLS_TO_USE = ['name', 'description', 'price'] # The columns to build the 'text' from


# --- 1. Load Data and Oracle ---
print("--- Loading Raw Data and Oracle ---")
df_a_raw = pd.read_csv(PATH_RAW_A, encoding='unicode_escape')
df_b_raw = pd.read_csv(PATH_RAW_B, encoding='unicode_escape')
df_gt = pd.read_csv(PATH_GT, encoding="unicode_escape", keep_default_na=False)


truthD = dict()
a = 0
for i, r in df_gt.iterrows():
     idAbt = r["idAbt"]
     idBuy = r["idBuy"]
     if idAbt in truthD:
           ids = truthD[idAbt]
           ids.append(idBuy)
           a += 1
     else:
          truthD[idAbt] = [idBuy]
          a += 1
matches = len(truthD.keys()) + a
print("No of matches=", matches)

#gt_lookup = {
#    (str(id1), str(id2)) 
#    for id1, id2 in zip(df_gt[ID_COL_A], df_gt[ID_COL_B])
#}
#print(gt_lookup)
#exit(1)

gt_lookup = {
    (str(key), str(value))
    for key, value_list in truthD.items()
    for value in value_list
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


abt_id_mapper = df_a['id'].to_dict()
buy_id_mapper = df_b['id'].to_dict()


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
model, scaler, f1, bt1 = lib.train_classifier(current_clean_training_set, test_set, a_lookup, b_lookup)
print(f"--- Iteration 0 (Clean) F1-Score: {f1:.4f} ---")




# This is our set of clean, oracle-verified labels. It starts empty.
current_clean_training_set = []
last_f1_score = 0.0  # Track the F1 score from the previous iteration
MIN_IMPROVEMENT_THRESHOLD = 0.01  # Stop if F1 improves by less than 0.5%

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
    model, scaler, f1, best_threshold1 = lib.train_classifier(current_clean_training_set, test_set, a_lookup, b_lookup)
    
    print(f"--- Iteration {i} F1-Score: {f1:.4f} ---")
    improvement = f1 - last_f1_score
    if improvement < MIN_IMPROVEMENT_THRESHOLD and i > 1: # Don't stop on the first iteration
        print(f"\nF1-Score improved by only {improvement:.4f}. Stopping Active Learning loop early.")
        break # Exit the loop

    last_f1_score = f1 # Update the score for the next iteration


print("\nActive Learning loop complete.")
print(f"Final F1-Score: {f1:.4f} at threshold {best_threshold1:.4f}")





print("--- Starting Phase 2: Training Final Precision Model ---")

# Use your new functions to train the hybrid-feature model
# It trains on the *same* clean set, but uses *more features*
precision_model, precision_scaler, best_f1, best_threshold2 = lib.train_precision_classifier(
    current_clean_training_set,  # Your full, clean training set
    test_set,                    # Your held-out test set
    a_lookup,                    # Your lookup dict (needs 'v' and 'name')
    b_lookup,                     # Your lookup dict (needs 'v' and 'name')
    col="name"
)

print(f"--- Precision Model Trained! ---")
print(f"Final F1-Score: {best_f1:.4f} at threshold {best_threshold2:.4f}")






abt_embeddings = np.array(df_a['v'].tolist()).astype(np.float32)
buy_embeddings = np.array(df_b['v'].tolist()).astype(np.float32)
#amazon_id_to_index = {id_val: index for index, id_val in amazon_id_mapper.items()}
#walmart_id_to_index = {id_val: index for index, id_val in walmart_id_mapper.items()}

print("\n--- 4. Building Faiss index for Walmart records ---")
d = buy_embeddings.shape[1]
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 60
index.hnsw.efSearch = 64
index.add(abt_embeddings)
print(f"Index built successfully with {index.ntotal} records.")



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

D, I = index.search(buy_embeddings, k)

true_positives = 0
false_positives = 0
X_predict = []
candidate_pairs_list = [] # Store the (record_a, record_b)
y_true_list = []          # Store the true labels for final evaluation
stage1_pairs_data = []
X_stage1_features = []
for buy_idx, abt_matches in enumerate(I):
    buy_id = buy_id_mapper[buy_idx]
    buy_embedding = buy_embeddings[buy_idx]
    buy_record = df_b.iloc[buy_idx] # Get the full record

    for i, abt_idx in enumerate(abt_matches):
       abt_id = abt_id_mapper[abt_idx]
       abt_embedding = abt_embeddings[abt_idx]
       abt_record = df_a.iloc[abt_idx] # Get the full record
       if abt_id in truthD:
            v1 = {"v": abt_embedding}
            v2 = {"v": buy_embedding}
            stage1_pairs_data.append((abt_record, buy_record))
            feature_vector = lib.create_pure_embedding_vector(v2 , v1 )
            X_stage1_features.append(feature_vector) 
            X_predict.append(feature_vector)
            candidate_pairs_list.append((v2, v1))
            if buy_id in truthD[abt_id]:
                y_true_list.append(1)
            else:
                y_true_list.append(0)

X_matrix = np.array(X_predict)
X_scaled = scaler.transform(X_matrix)
all_probabilities = model.predict(X_scaled, batch_size=256)
all_decisions_binary = (all_probabilities.flatten() > best_threshold1).astype(int)
y_true_array = np.array(y_true_list)
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
        abt_record, buy_record = stage1_pairs_data[idx]
        
        # Create the SLOW, HYBRID (embedding + jaro) feature vector
        hybrid_features = lib.create_hybrid_feature_vector(abt_record, buy_record)
        X_stage2_features.append(hybrid_features)
        
        # Get the true label for our final test
        abt_id = str(abt_record["id"])
        buy_id = str(buy_record["id"])
        y_true_list.append(1.0 if (abt_id, buy_id) in gt_lookup else 0.0)

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


 









