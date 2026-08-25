
import os

DEFAULT_EMBEDDING_DEVICE = os.getenv("OPENROUTER_EMBEDDING_DEVICE", "cpu").strip().lower() or "cpu"
if DEFAULT_EMBEDDING_DEVICE == "cpu":
    
    
    
    
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.cluster import AgglomerativeClustering

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  
    SentenceTransformer = None


PRIMARY_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
PRIMARY_EMBEDDING_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
DEFAULT_EMBEDDING_MODEL = PRIMARY_EMBEDDING_MODEL
DEFAULT_EMBEDDING_REVISION = PRIMARY_EMBEDDING_REVISION


def get_hf_cache_hint(model_name):
    repo_id = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    return os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "huggingface",
        "hub",
        f"models--{repo_id.replace('/', '--')}",
    )


class SemanticScorer:
    def __init__(self, model_name=None, model_revision=None):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for SemanticScorer. "
                "Install it with pip install sentence-transformers."
            )

        model_name = model_name or DEFAULT_EMBEDDING_MODEL
        if model_revision is None and model_name == PRIMARY_EMBEDDING_MODEL:
            model_revision = DEFAULT_EMBEDDING_REVISION
        self.model_name = model_name
        self.model_revision = model_revision
        self.embedding_device = DEFAULT_EMBEDDING_DEVICE
        self.model_note = (
            f"Embedding model is {self.model_name} at revision "
            f"{self.model_revision or 'repository default'} on {self.embedding_device}."
        )
        cache_hint = get_hf_cache_hint(model_name)
        prefer_local_only = os.path.isdir(cache_hint)
        load_kwargs = {
            "device": self.embedding_device,
            "local_files_only": prefer_local_only,
        }
        if model_revision:
            load_kwargs["revision"] = model_revision

        try:
            self.model = SentenceTransformer(model_name, **load_kwargs)
        except Exception:
            if not prefer_local_only:
                raise
            load_kwargs["local_files_only"] = False
            self.model = SentenceTransformer(model_name, **load_kwargs)

    def calculate_originality(self, target_concept, idea):
        """
        Calculates the semantic originality (distance) between the concept and the idea.
        Originality = 1 - Cosine Similarity.
        """
        embeddings = self.model.encode([target_concept, idea])
        cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        distance = 1.0 - cos_sim
        return float(max(0.0, min(2.0, distance)))

    def deduplicate_ideas(self, ideas, similarity_threshold=0.85):
        
        if not ideas:
            return {
                "unique_indices": [],
                "unique_ideas": [],
                "duplicate_map": {},
                "num_removed": 0
            }
        
        if len(ideas) == 1:
            return {
                "unique_indices": [0],
                "unique_ideas": ideas[:],
                "duplicate_map": {0: 0},
                "num_removed": 0
            }
        
        
        embeddings = self.model.encode(ideas)
        
        
        sim_matrix = cosine_similarity(embeddings)
        
        
        
        
        unique_indices = []
        duplicate_map = {}  
        
        for i in range(len(ideas)):
            is_duplicate = False
            for kept_idx in unique_indices:
                if sim_matrix[i][kept_idx] >= similarity_threshold:
                    
                    duplicate_map[i] = kept_idx
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_indices.append(i)
                duplicate_map[i] = i  
        
        unique_ideas = [ideas[i] for i in unique_indices]
        num_removed = len(ideas) - len(unique_indices)
        
        return {
            "unique_indices": unique_indices,
            "unique_ideas": unique_ideas,
            "duplicate_map": duplicate_map,
            "num_removed": num_removed
        }

    def calculate_flexibility(self, ideas, distance_threshold=0.55):
        """
        Calculates cognitive flexibility using semantic clustering.
        Based on psychological methods of semantic clustering for divergent thinking.
        
        :param ideas: List of cleaned idea strings.
        :param distance_threshold: The cosine distance threshold used to form clusters.
                                   A slightly higher threshold prevents near-ceiling cluster
                                   counts on diverse LLM outputs.
        :return: Dictionary containing clustering plus continuous dispersion metrics.
        """
        if not ideas:
            return {
                "num_clusters": 0,
                "category_switches": 0,
                "labels": [],
                "cluster_ratio": 0.0,
                "switch_rate": 0.0,
                "mean_pairwise_distance": 0.0,
                "mean_adjacent_distance": 0.0,
                "cluster_entropy": 0.0,
            }
        if len(ideas) == 1:
            return {
                "num_clusters": 1,
                "category_switches": 0,
                "labels": [0],
                "cluster_ratio": 1.0,
                "switch_rate": 0.0,
                "mean_pairwise_distance": 0.0,
                "mean_adjacent_distance": 0.0,
                "cluster_entropy": 0.0,
            }

        embeddings = self.model.encode(ideas)
        dist_matrix = cosine_distances(embeddings)
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='average'
        )
        
        labels = clustering.fit_predict(dist_matrix)
        
        num_clusters = len(set(labels))
        switches = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
        cluster_ratio = num_clusters / len(ideas) if ideas else 0.0
        switch_rate = switches / (len(ideas) - 1) if len(ideas) > 1 else 0.0

        upper_triangle = dist_matrix[np.triu_indices(len(ideas), k=1)]
        mean_pairwise_distance = float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else 0.0

        adjacent_distances = [float(dist_matrix[i - 1][i]) for i in range(1, len(ideas))]
        mean_adjacent_distance = float(np.mean(adjacent_distances)) if adjacent_distances else 0.0

        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum() if counts.sum() > 0 else np.array([])
        entropy = float(-np.sum(probabilities * np.log2(probabilities))) if len(probabilities) > 0 else 0.0
        max_entropy = float(np.log2(len(unique_labels))) if len(unique_labels) > 1 else 0.0
        cluster_entropy = (entropy / max_entropy) if max_entropy > 0 else 0.0

        return {
            "num_clusters": num_clusters,
            "category_switches": switches,
            "labels": labels.tolist(),
            "cluster_ratio": round(cluster_ratio, 4),
            "switch_rate": round(switch_rate, 4),
            "mean_pairwise_distance": round(mean_pairwise_distance, 4),
            "mean_adjacent_distance": round(mean_adjacent_distance, 4),
            "cluster_entropy": round(cluster_entropy, 4),
        }
