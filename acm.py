import pandas as pd
import numpy as np
import lib as lib
import random
import faiss
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.cluster import MiniBatchKMeans

N_PARTITIONS = 3
NUM_ITERATIONS_PER_PARTITION = 3
LABELS_PER_ITERATION = 300 
SEED_SIZE = 100              

VAL_SET_PROPORTION = 0.1
VAL_SET_MAX_SIZE = 20000  

PATH_RAW_A = './data/ACM.csv'
PATH_RAW_B = './data/DBLP.csv'
PATH_GT = './data/truth_ACM_DBLP.csv'
ID_COL_A = 'idACM'
ID_COL_B = 'idDBLP'
COLS_TO_USE = ['title', 'authors', 'venue', 'year'] 

print("--- Loading Raw Data and Oracle ---")
df_a_raw = pd.read_csv(PATH_RAW_A, encoding='unicode_escape')
df_b_raw = pd.read_csv(PATH_RAW_B, encoding='unicode_escape')
df_gt = pd.read_csv(PATH_GT, encoding="unicode_escape", keep_default_na=False)

truthD = dict()
a = 0
for i, r in df_gt.iterrows():
        idACM = r["idACM"]
        idDBLP = r["idDBLP"]
        if idACM in truthD:
            ids = truthD[idACM]
            ids.append(idDBLP)
            a += 1
        else:
            truthD[idACM] = [idDBLP]
matches = len(truthD.keys()) + a
print("No of matches=", matches)

gt_lookup = {
    (str(key), str(value))
    for key, value_list in truthD.items()
    for value in value_list
}
print(f"Loaded Oracle with {len(gt_lookup)} total matches.")

df_a, df_b = lib.bootstrap_embeddings_only(
      df_a_raw, df_b_raw, "source_a", "source_b", COLS_TO_USE
)

dblp_embeddings = np.array(df_b['v'].tolist()).astype('float32')
acm_embeddings = np.array(df_a['v'].tolist()).astype('float32')
df_a_whole  = df_a
SAMPLE_PROPORTION = 0.2
SAMPLE_SIZE= int(len(df_b) * SAMPLE_PROPORTION)
df_a = df_a.sample(n=SAMPLE_SIZE, random_state=42)

a_lookup = {row['text']: row for _, row in df_a.iterrows()}
b_lookup = {row['text']: row for _, row in df_b.iterrows()}

print(f"\n--- Partitioning data into {N_PARTITIONS} chunks using KMeans ---")
embeddings_a = np.array(df_a['v'].tolist()).astype('float32')
embeddings_b = np.array(df_b['v'].tolist()).astype('float32')
kmeans = MiniBatchKMeans(n_clusters=N_PARTITIONS, random_state=42, batch_size=256, n_init=3)
df_a['partition'] = kmeans.fit_predict(embeddings_a)

print("Building global FAISS index...")
d = embeddings_a.shape[1]
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 60
index.hnsw.efSearch = 64
faiss.normalize_L2(dblp_embeddings) 
index.add(dblp_embeddings)

master_clean_training_set = []
fast_validation_set = [] 
model, scaler = (None, None) 

for i in range(N_PARTITIONS):
    print(f"\n--- Processing Partition {i+1}/{N_PARTITIONS} ---")

    df_a_partition = df_a[df_a['partition'] == i]
    if len(df_a_partition) == 0:
        print("Partition is empty, skipping.")
        continue

    embeddings_a_partition = np.array(df_a_partition['v'].tolist()).astype('float32')
    faiss.normalize_L2(embeddings_a_partition) 

    print(f"Generating candidate pool for {len(df_a_partition)} records...")
    D, I = index.search(embeddings_a_partition, k=10)

    candidate_pool_text_partition = set()
    for a_idx, b_indices_list in enumerate(I):
        text_a = df_a_partition.iloc[a_idx]['text']
        for b_idx in b_indices_list:
            text_b = df_b.iloc[b_idx]['text']
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
         col="title"
    )
    print(f"Master Precision Model F1-Score: {best_f1:.4f}")
else:
    print("No clean labels gathered, skipping precision model.")


print("\n--- Starting Final Resolution (Held-Out Test Set, Classifier Metrics) ---")

train_val_ids = set(df_a['id'].astype(str).tolist()) 

faiss.normalize_L2(acm_embeddings)
D, I = index.search(acm_embeddings, k=5)

X_stage1_features = []
stage1_pairs_data = [] 
y_true_list_stage1 = [] 

print("Running Stage 1 on HELD-OUT TEST SET only...")
for acm_idx, dblp_indices in enumerate(I):
    acm_record = df_a_whole.iloc[acm_idx]
    
    if str(acm_record['id']) in train_val_ids:
        continue 

    for dblp_idx in dblp_indices:
        dblp_record = df_b.iloc[dblp_idx]

        stage1_pairs_data.append((acm_record, dblp_record))
        X_stage1_features.append(lib.create_pure_embedding_vector(acm_record, dblp_record))

        acm_id = str(acm_record["id"])
        dblp_id = str(dblp_record["id"])
        is_match = 1.0 if (acm_id, dblp_id) in gt_lookup else 0.0
        y_true_list_stage1.append(is_match)

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
        acm_record, dblp_record = stage1_pairs_data[idx]
        hybrid_features = lib.create_hybrid_feature_vector(acm_record, dblp_record, col="title")
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

real_total_labels = len(master_clean_training_set) + len(fast_validation_set)
active_queries = len(master_clean_training_set) - (SEED_SIZE * N_PARTITIONS) 

print("\n" + "="*30)
print("PAPER TABLE 2 DATA (ACM-DBLP)")
print("="*30)
print(f"Seed (B_seed): {SEED_SIZE * N_PARTITIONS}") 
print(f"Valid (|V|):   {len(fast_validation_set)}")
print(f"Active Loop:   {active_queries}") 
print(f"Total Budget:  {real_total_labels}")
print("="*30)
