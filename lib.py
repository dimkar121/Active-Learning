# --- Step 0: All Imports ---
import pandas as pd
import numpy as np
import re
import faiss
import joblib
import random
from tqdm import tqdm  # For progress bars
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, precision_recall_curve
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import math
from jellyfish import jaro_winkler_similarity

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)



import numpy as np
from jellyfish import jaro_winkler_similarity
# (Need other imports like tensorflow, sklearn, etc. for the trainer)

def create_hybrid_feature_vector(record_a, record_b, col="name" ):
    """
    Creates a rich feature vector by combining embeddings
    AND Jaro-Winkler similarity for the 'name' column.
    """
    
    # --- 1. Embedding Features (The "Semantic" Signal) ---
    v_a = np.array(record_a['v'])
    v_b = np.array(record_b['v'])
    
    v_diff = np.abs(v_a - v_b)  # Element-wise difference
    v_prod = v_a * v_b        # Element-wise product
    
    embedding_features = np.concatenate([v_a, v_b, v_diff, v_prod]) # (1536 dims)

    # --- 2. Structural Feature (The "Precision" Signal) ---
    
    # Jaro-Winkler on 'name'
    # We use .get() for safety in case the column is missing
    name_a = str(record_a.get(col, ''))
    name_b = str(record_b.get(col, ''))
    jaro_name_sim = jaro_winkler_similarity(name_a, name_b)
    
    structural_features = np.array([jaro_name_sim]) # (1 dim)
    
    # --- 3. Concatenate All Features ---
    final_vector = np.concatenate([embedding_features, structural_features])
    
    return final_vector




def train_precision_classifier(training_pairs, validation_pairs, a_lookup, b_lookup, col="name" ):
    """
    Trains the new MLP (Stage 2: Precision Model) using the
    HYBRID feature vector (embeddings + jaro).
    """
    
    # --- 1. Create Training Matrix X and y ---
    X_train_list = []
    y_train_list = []
    for pair in training_pairs:
        record_a = a_lookup.get(pair[0]) 
        record_b = b_lookup.get(pair[1]) 
        label = pair[2]                  
        
        if record_a is not None and record_b is not None:
            
            # --- THIS IS THE ONLY CHANGE ---
            features = create_hybrid_feature_vector(record_a, record_b, col)
            # --- END OF CHANGE ---
            
            X_train_list.append(features)
            y_train_list.append(label)

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    if len(X_train) == 0:
        print("No valid training data found.")
        return None, None, 0.0

    # --- 2. Create Validation Matrix X and y ---
    X_val_list = []
    y_val_list = []
    for pair in validation_pairs:
        record_a = a_lookup.get(pair[0])
        record_b = b_lookup.get(pair[1])
        label = pair[2]
        
        if record_a is not None and record_b is not None:
            # --- THIS IS THE ONLY CHANGE ---
            features = create_hybrid_feature_vector(record_a, record_b)
            # --- END OF CHANGE ---
            
            X_val_list.append(features)
            y_val_list.append(label)
    
    X_val = np.array(X_val_list)
    y_val = np.array(y_val_list)

    # --- 3. Scale Data ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # --- 4. Define and Train Model ---
    # The input dimension is now 1536 + 1 = 1537
    input_dim = X_train_scaled.shape[1] 
    
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # (Class weight calculation remains the same)
    total = len(y_train); neg = np.sum(y_train == 0); pos = np.sum(y_train == 1)
    weight_for_0 = (1 / neg) * (total / 2.0) if neg > 0 else 1
    weight_for_1 = (1 / pos) * (total / 2.0) if pos > 0 else 1

    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=20, batch_size=64,
        class_weight={0: weight_for_0, 1: weight_for_1},
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=0
    )
    
    # --- 5. Evaluate and Return ---
    preds_prob = model.predict(X_val_scaled).flatten()
    
    # (Find best threshold and F1 score - remains the same)
    precision, recall, thresholds = precision_recall_curve(y_val, preds_prob)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
    best_f1_index = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_index]
    best_threshold = thresholds[best_f1_index]

    print(f"--- Precision Model Training Complete ---")
    print(f"Best F1-Score: {best_f1:.4f} (at threshold {best_threshold:.4f})")
    
    return model, scaler, best_f1, best_threshold














# --- Step 1: Helper Functions (The Tools) ---

def create_pure_embedding_vector(record_a, record_b):
    """Creates a rich, FLAT feature vector using ONLY embedding interactions."""
    v_a = np.array(record_a['v'])
    v_b = np.array(record_b['v'])
    v_diff = np.abs(v_a - v_b)
    v_prod = v_a * v_b
    final_vector = np.concatenate([v_a, v_b, v_diff, v_prod])
    return final_vector




def train_classifier(training_pairs, validation_pairs, a_lookup, b_lookup):
    """
    Trains a new MLP classifier on the provided training pairs
    and evaluates it on the validation pairs.
    """
    
    # --- 1. Create Training Matrix X and y ---
    X_train_list = []
    y_train_list = []
    for pair in training_pairs:
        record_a = a_lookup.get(pair[0]) # pair[0] is text1
        record_b = b_lookup.get(pair[1]) # pair[1] is text2
        label = pair[2]                  # pair[2] is label
        
        if record_a is not None and record_b is not None:
            features = create_pure_embedding_vector(record_a, record_b)
            if features.shape[0] == 1536:
                X_train_list.append(features)
                y_train_list.append(label)

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)


    if len(X_train) == 0:
        print("No valid training data found.")
        return None, None, 0.0

    # --- 2. Create Validation Matrix X and y ---
    X_val_list = []
    y_val_list = []
    for pair in validation_pairs:
        record_a = a_lookup.get(pair[0])
        record_b = b_lookup.get(pair[1])
        label = pair[2]
        
        if record_a is not None and record_b is not None:
            features = create_pure_embedding_vector(record_a, record_b)
            if features.shape[0] == 1536:
                X_val_list.append(features)
                y_val_list.append(label)
    

    X_val = np.array(X_val_list)
    y_val = np.array(y_val_list)

    # --- 3. Scale Data ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # --- 4. Define and Train Model ---
    input_dim = X_train_scaled.shape[1]
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Calculate class weights for imbalance
    total = len(y_train)
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    weight_for_0 = (1 / neg) * (total / 2.0) if neg > 0 else 1
    weight_for_1 = (1 / pos) * (total / 2.0) if pos > 0 else 1

    model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=20,
        batch_size=64,
        class_weight={0: weight_for_0, 1: weight_for_1},
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=0 # Set to 1 to see training progress
    )
   
    # --- 5. Evaluate and Return ---
    preds_prob = model.predict(X_val_scaled).flatten()
    preds_binary = (preds_prob > 0.5).astype(int)
    f1 = f1_score(y_val, preds_binary)
    
    # --- START CHANGE ---
    # Find the best threshold instead of using 0.5
    precision, recall, thresholds = precision_recall_curve(y_val, preds_prob)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)    
    best_f1_index = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_index]
    
    # Handle edge case where the "best" index might be out of bounds for thresholds
    if best_f1_index < len(thresholds):
        best_threshold = thresholds[best_f1_index]
    else:
        # This happens if all predictions are 0, use 0.5 as a safe default
        best_threshold = 0.5 
    
    print(f"(Best F1: {best_f1:.4f} at Threshold: {best_threshold:.4f})")

    return model, scaler, f1, best_threshold




def get_candidate_pool(df_a, df_b, k=7):
    """
    Uses FAISS on the fine-tuned 'v' embeddings to create a
    large pool of high-potential candidate pairs.
    """
    print(f"\nBuilding candidate pool: Finding top {k} neighbors for {len(df_a)} records...")
    
    # Get embeddings from dataframes
    embeddings_a = np.array(df_a['v'].tolist()).astype('float32')
    embeddings_b = np.array(df_b['v'].tolist()).astype('float32')
    
    # FAISS requires normalized vectors for cosine similarity (Inner Product)
    faiss.normalize_L2(embeddings_a)
    faiss.normalize_L2(embeddings_b)
    
    d = embeddings_b.shape[1]
   
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 64

    index.add(embeddings_b)
    
    # Search for the k nearest B-records for every A-record
    distances, indices = index.search(embeddings_a, k)
    
    candidate_pairs = set() # Use a set to avoid duplicates
    for a_idx, b_indices_list in enumerate(indices):
        text_a = df_a.iloc[a_idx]['text']
        for b_idx in b_indices_list:
            text_b = df_b.iloc[b_idx]['text']
            candidate_pairs.add((text_a, text_b))
            
    print(f"Created a candidate pool of {len(candidate_pairs)} pairs.")
    return list(candidate_pairs)

def query_oracle(pairs_to_label, a_lookup, b_lookup, gt_lookup, id_col_a, id_col_b):
    """
    Simulates a human expert by checking the ground truth.
    Returns a list of perfectly labeled pairs.
    """
    clean_labels = []
    for text_a, text_b in pairs_to_label:
        record_a = a_lookup.get(text_a)
        record_b = b_lookup.get(text_b)
        
        if record_a is None or record_b is None:
            continue
            
        id_a = str(record_a[id_col_a])
        id_b = str(record_b[id_col_b])
        
        if (id_a, id_b) in gt_lookup:
            label = 1.0
        else:
            label = 0.0
        
        clean_labels.append((text_a, text_b, label))
        
    return clean_labels





def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text





def bootstrap_embeddings_only(df_a, df_b, source_a_name, source_b_name, cols):
    """
    This is Iteration 0.
    1. Generates initial 'v' embeddings from a pre-trained model.
    2. Saves the .pqt files.
    """
    print("--- Phase 1: Bootstrapping Embeddings ---")
    
    # 1. Preprocess text
    for col in cols:
        df_a[col] = df_a[col].apply(preprocess_text)
        df_b[col] = df_b[col].apply(preprocess_text)

    df_a['source'] = source_a_name
    df_b['source'] = source_b_name
    df_a['text'] = df_a[cols].astype(str).agg(' [SEP] '.join, axis=1)
    df_b['text'] = df_b[cols].astype(str).agg(' [SEP] '.join, axis=1)
    
    # 2. Load pre-trained model and encode
    print("Loading pre-trained SBERT and encoding...")
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    
    df_a['v'] = [emb.tolist() for emb in model.encode(
        df_a['text'].tolist(), show_progress_bar=True, normalize_embeddings=True
    )]
    df_b['v'] = [emb.tolist() for emb in model.encode(
        df_b['text'].tolist(), show_progress_bar=True, normalize_embeddings=True
    )]

    # 3. Save the .pqt files
    print("Saving bootstrapped .pqt files...")
    df_a.to_parquet('./data/df_a_bootstrapped.pqt', engine="pyarrow")
    df_b.to_parquet('./data/df_b_bootstrapped.pqt', engine="pyarrow")
    print("Bootstrapping complete.")
    return df_a, df_b

















def bootstrap_and_get_noisy_labels(df_a, df_b, source_a_name, source_b_name, cols):
    """
    This is Iteration 0.
    1. Generates initial 'v' embeddings from a pre-trained model.
    2. Saves the .pqt files (your question).
    3. Runs unsupervised clustering.
    4. Returns a list of "noisy" training pairs.
    """
    print("--- Phase 1: Bootstrapping ---")
    
    # 1. Preprocess text
    for col in cols:
        df_a[col] = df_a[col].apply(preprocess_text)
        df_b[col] = df_b[col].apply(preprocess_text)

    df_a['source'] = source_a_name
    df_b['source'] = source_b_name
    df_a['text'] = df_a[cols].astype(str).agg(' [SEP] '.join, axis=1)
    df_b['text'] = df_b[cols].astype(str).agg(' [SEP] '.join, axis=1)
    
    # 2. Load pre-trained model and encode
    print("Loading pre-trained SBERT and encoding...")
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    
    # Create the initial 'v' column
    df_a['v'] = [emb.tolist() for emb in model.encode(
        df_a['text'].tolist(), show_progress_bar=True, normalize_embeddings=True
    )]
    df_b['v'] = [emb.tolist() for emb in model.encode(
        df_b['text'].tolist(), show_progress_bar=True, normalize_embeddings=True
    )]

    # 3. THIS ANSWERS YOUR QUESTION: Save the .pqt files
    print("Saving bootstrapped .pqt files...")
    df_a.to_parquet('./data/df_a_bootstrapped.pqt', engine="pyarrow")
    df_b.to_parquet('./data/df_b_bootstrapped.pqt', engine="pyarrow")
    print("  ... df_a_bootstrapped.pqt saved.")
    print("  ... df_b_bootstrapped.pqt saved.")

    # 4. Run Unsupervised Clustering (FAISS + Leiden)
    print("Running unsupervised clustering...")
    df_all = pd.concat([df_a, df_b], ignore_index=True)
    embeddings = np.array(df_all['v'].tolist()).astype('float32')
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d) # Using Inner Product (cosine similarity)
    index.add(embeddings)
    k = 5 # Number of nearest neighbors for graph
    distances, indices = index.search(embeddings, k)

    edges = []
    for i in range(len(indices)):
        for j in indices[i][1:]:
            edges.append((i, j))

    graph = ig.Graph(edges, directed=False)
    
    # --- This is the key parameter to tune! ---
    # Start with a high resolution to get small, pure clusters
    partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=2.5)
    cluster_labels = np.array(partition.membership)
    df_all['cluster'] = cluster_labels
    print(f"Found {len(set(cluster_labels))} initial clusters.")

    # 5. Mine Noisy Pairs from Clusters
    noisy_training_pairs = []
    
    # --- Mine Hard Negatives ---
    for i in range(len(indices)):
        anchor_cluster = df_all.iloc[i]['cluster']
        anchor_text = df_all.iloc[i]['text']
        
        for neighbor_idx in indices[i][1:]: # 4 neighbors
            neighbor_cluster = df_all.iloc[neighbor_idx]['cluster']
            neighbor_text = df_all.iloc[neighbor_idx]['text']
            
            if anchor_cluster != neighbor_cluster:
                noisy_training_pairs.append((anchor_text, neighbor_text, 0.0)) # label = No Match

    # --- Mine Positives ---
    num_negatives = len(noisy_training_pairs)
    positive_count = 0
    
    for cluster_id in pd.Series(cluster_labels).unique():
        if positive_count >= num_negatives: # Balance the dataset
            break
            
        cluster_df = df_all[df_all['cluster'] == cluster_id]
        
        # Get cross-source pairs
        a_records = cluster_df[cluster_df['source'] == source_a_name]
        b_records = cluster_df[cluster_df['source'] == source_b_name]
        
        if not a_records.empty and not b_records.empty:
            # Simple: just pair the first A with all B's
            anchor_text = a_records.iloc[0]['text']
            for b_text in b_records['text']:
                noisy_training_pairs.append((anchor_text, b_text, 1.0)) # label = Match
                positive_count += 1
                if positive_count >= num_negatives:
                    break

    print(f"Bootstrapping complete. Mined {len(noisy_training_pairs)} noisy pairs.")
    return df_a, df_b, noisy_training_pairs


def fine_tune_sbert_model(training_pairs, base_model_name, save_path):
    """
    Fine-tunes an SBERT model on a list of clean (text1, text2, label) pairs.

    Args:
        training_pairs: A list of tuples, e.g., [('text_a1', 'text_b1', 1.0), ('text_a2', 'text_b2', 0.0)]
        base_model_name: The pre-trained model to start from, e.g., 'all-MiniLM-L6-v2'
        save_path: The directory path to save the new fine-tuned model.
    """
    print(f"--- Starting SBERT Fine-Tuning ---")
    print(f"Loading base model: {base_model_name}")
    
    # 1. Load the pre-trained base model
    model = SentenceTransformer(base_model_name)

    # 2. Format the training data into InputExamples
    print(f"Formatting {len(training_pairs)} pairs into InputExamples...")
    train_examples = []
    for (text1, text2, label) in training_pairs:
        # The label must be a float
        train_examples.append(InputExample(texts=[text1, text2], label=float(label)))

    # 3. Create the DataLoader
    # This will batch and shuffle the data during training
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

    # 4. Define the loss function
    # CosineSimilarityLoss is perfect. It will try to make:
    # - label=1.0 pairs have a cosine similarity of 1.0
    # - label=0.0 pairs have a cosine similarity of 0.0 (or less)
    train_loss = losses.CosineSimilarityLoss(model)

    # 5. Define training parameters
    num_epochs = 1  # 1 epoch is often enough for fine-tuning
    
    # Calculate warmup steps (10% of training data)
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    print(f"Total training steps: {len(train_dataloader) * num_epochs}")
    print(f"Warmup steps: {warmup_steps}")

    # 6. Start the fine-tuning
    print("--- Starting model.fit() ---")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=save_path,
        show_progress_bar=True
    )
    
    print(f"--- Fine-Tuning Complete ---")
    print(f"New model saved to: {save_path}")
