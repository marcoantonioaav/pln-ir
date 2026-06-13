import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
import os
import gc

from metrics import PerformanceTracker, calculate_recall_at_k, calculate_mean_distance, calculate_1nn_distance_diff

try:
    import initialization_cpp
except ImportError:
    print("Warning: initialization_cpp not found.")
    initialization_cpp = None

def compute_ground_truth_cosine(corpus_ds, queries, k):
    ground_truth = []
    # Compute dot products in chunks if dataset is too large
    # but since queries are small, we can matrix multiply
    for q in tqdm(queries, desc="Computing Ground Truth"):
        sim = []
        chunk_size = 50000
        total_size = corpus_ds.shape[0]
        for start_idx in range(0, total_size, chunk_size):
            end_idx = min(start_idx + chunk_size, total_size)
            chunk = corpus_ds[start_idx:end_idx]
            sim.append(np.dot(chunk, q))
        sim = np.concatenate(sim)
        nearest_indices = np.argsort(-sim)[:k]
        ground_truth.append(nearest_indices.tolist())
    return ground_truth

def run_dataset_experiment(dataset_name, corpus_path, queries_path, output_dir, max_queries=1000):
    print(f"=== Running Experiment on {dataset_name} ===")
    corpus_f = h5py.File(corpus_path, 'r')
    f = corpus_f
    total_size = f['embeddings'].shape[0]
    load_size = total_size
    print(f"Dataset total size is {total_size}. Loading all points...")
    with h5py.File(queries_path, 'r') as f_q:
        q_total_size = f_q['embeddings'].shape[0]
        q_load_size = min(q_total_size, max_queries)
        print(f"Loading {q_load_size} queries...")
        q_dataset = f_q['embeddings'][:q_load_size]
        
    k_search = 100
        
    # Read queries once since they are small enough
    queries = q_dataset
    ground_truth = compute_ground_truth_cosine(corpus_f['embeddings'], queries, k=100)
    
    approaches = {
        "VP-tree": [
            ("VP-tree (max_leaves=100)", initialization_cpp.VPTreeInit(100, 1.0, 1.0, "cosine")),
            ("VP-tree (max_leaves=500)", initialization_cpp.VPTreeInit(500, 1.0, 1.0, "cosine")),
            ("VP-tree (max_leaves=1000)", initialization_cpp.VPTreeInit(1000, 1.0, 1.0, "cosine")),
            ("VP-tree (max_leaves=2000)", initialization_cpp.VPTreeInit(2000, 1.0, 1.0, "cosine")),
            ("VP-tree (max_leaves=5000)", initialization_cpp.VPTreeInit(5000, 1.0, 1.0, "cosine"))
        ],
        "Stacked NSW": [
            ("Stacked NSW (ef=10)", initialization_cpp.StackedNSWInit(16, 200, 10, "cosine")),
            ("Stacked NSW (ef=50)", initialization_cpp.StackedNSWInit(16, 200, 50, "cosine")),
            ("Stacked NSW (ef=100)", initialization_cpp.StackedNSWInit(16, 200, 100, "cosine")),
            ("Stacked NSW (ef=200)", initialization_cpp.StackedNSWInit(16, 200, 200, "cosine")),
            ("Stacked NSW (ef=500)", initialization_cpp.StackedNSWInit(16, 200, 500, "cosine"))
        ],
        "LSH": [
            ("LSH (probes=10)", initialization_cpp.LSHInit(50, 16, 10, "cosine")),
            ("LSH (probes=50)", initialization_cpp.LSHInit(50, 16, 50, "cosine")),
            ("LSH (probes=100)", initialization_cpp.LSHInit(50, 16, 100, "cosine")),
            ("LSH (probes=500)", initialization_cpp.LSHInit(50, 16, 500, "cosine"))
        ],
        "t KD-Trees": [
            ("t KD-Trees (checks=100)", initialization_cpp.FlannKDTreeInit(4, 100, "cosine")),
            ("t KD-Trees (checks=500)", initialization_cpp.FlannKDTreeInit(4, 500, "cosine")),
            ("t KD-Trees (checks=1000)", initialization_cpp.FlannKDTreeInit(4, 1000, "cosine")),
            ("t KD-Trees (checks=2000)", initialization_cpp.FlannKDTreeInit(4, 2000, "cosine")),
            ("t KD-Trees (checks=5000)", initialization_cpp.FlannKDTreeInit(4, 5000, "cosine")),
        ],
        "t K-Means": [
            ("t K-Means (checks=100)", initialization_cpp.FlannKMeansInit(1, 32, 11, 100, "cosine")),
            ("t K-Means (checks=500)", initialization_cpp.FlannKMeansInit(1, 32, 11, 500, "cosine")),
            ("t K-Means (checks=1000)", initialization_cpp.FlannKMeansInit(1, 32, 11, 1000, "cosine")),
            ("t K-Means (checks=2000)", initialization_cpp.FlannKMeansInit(1, 32, 11, 2000, "cosine")),
            ("t K-Means (checks=5000)", initialization_cpp.FlannKMeansInit(1, 32, 11, 5000, "cosine")),
        ],
        "Random": [
            ("Random Points", initialization_cpp.RandomPointsInit(42, "cosine"))
        ],
        "Medoid": [
            ("Medoid", initialization_cpp.MedoidInit("cosine"))
        ]
    }
    
    os.makedirs(output_dir, exist_ok=True)
    tracker = PerformanceTracker()
    
    for approach_family, instances in approaches.items():
        results_list = []
        for name, approach in instances:
            print(f"Testing {name} on {dataset_name}...")
            
            tracker.start()
            
            chunk_size = 50000
            for start_idx in tqdm(range(0, load_size, chunk_size), desc=f"Loading chunks for {name}"):
                end_idx = min(start_idx + chunk_size, load_size)
                chunk = corpus_f['embeddings'][start_idx:end_idx].astype(np.float32)
                approach.add_items(chunk.tolist())
                
            approach.build_index()
            build_time = tracker.stop()
            mem_footprint = approach.get_memory_usage() / (1024 * 1024)
            index_size = approach.get_index_size() / (1024 * 1024)
            
            approach.reset_distance_computations()
            tracker.start()
            search_results_indices = []
            
            for i in tqdm(range(len(queries)), desc=f"Querying {name}"):
                results = approach.search(queries[i].tolist(), k_search)
                search_results_indices.append([r.index for r in results])
                
            search_time = tracker.stop()
            total_dist_comps = approach.get_distance_computations()
            avg_dist_comps = total_dist_comps / len(queries)
            
            avg_recall = {1: 0.0, 10: 0.0, 25: 0.0, 100: 0.0}
            avg_mean_dist = 0.0
            avg_1nn_diff = 0.0
            
            for i in range(len(queries)):
                rec = calculate_recall_at_k(search_results_indices[i], ground_truth[i], k_values=[1, 10, 25, 100])
                for k in avg_recall:
                    avg_recall[k] += rec.get(k, 0.0)
                avg_mean_dist += calculate_mean_distance(search_results_indices[i], queries[i], corpus_f['embeddings'])
                avg_1nn_diff += calculate_1nn_distance_diff(search_results_indices[i][0], ground_truth[i][0], queries[i], corpus_f['embeddings'])
                
            for k in avg_recall:
                avg_recall[k] /= len(queries)
            avg_mean_dist /= len(queries)
            avg_1nn_diff /= len(queries)
            
            results_list.append({
                "Parameter Setup": name,
                "Build Time (s)": build_time,
                "Memory Footprint (MB)": mem_footprint,
                "Index Size (MB)": index_size,
                "Search Time (ms/q)": (search_time / len(queries)) * 1000,
                "Dist Comps": avg_dist_comps,
                "Recall@1": avg_recall[1],
                "Recall@10": avg_recall[10],
                "Recall@25": avg_recall[25],
                "Recall@100": avg_recall[100],
                "Mean Dist": avg_mean_dist,
                "1-NN Diff": avg_1nn_diff
            })
            
            del approach
            gc.collect()
            
        df = pd.DataFrame(results_list)
        df.to_csv(f"{output_dir}/results_{dataset_name}_{approach_family.replace(' ', '_')}.csv", index=False)
        
    corpus_f.close()

if __name__ == "__main__":
    msmarco_corpus = "/home/marco/Text-Dense-Retrieval-Evaluation/embeddings/msmarco/Octen-Embedding-0.6B_corpus.h5"
    msmarco_queries = "/home/marco/Text-Dense-Retrieval-Evaluation/embeddings/msmarco/Octen-Embedding-0.6B_queries.h5"
    
    nq_corpus = "/home/marco/Text-Dense-Retrieval-Evaluation/embeddings/nq/Octen-Embedding-0.6B_corpus.h5"
    nq_queries = "/home/marco/Text-Dense-Retrieval-Evaluation/embeddings/nq/Octen-Embedding-0.6B_queries.h5"
    
    output_dir = "results"
    
    if os.path.exists(msmarco_corpus):
        run_dataset_experiment("MSMARCO", msmarco_corpus, msmarco_queries, output_dir)
    else:
        print(f"Skipping MSMARCO, {msmarco_corpus} not found.")
        
    if os.path.exists(nq_corpus):
        run_dataset_experiment("NQ", nq_corpus, nq_queries, output_dir)
    else:
        print(f"Skipping NQ, {nq_corpus} not found.")
