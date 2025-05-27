import torch
import torch.nn.functional as F
import numpy as np

class PathReasoner:
    """
    A reasoning model that leverages pretrained KG embeddings to answer
    downstream questions by traversing multi-hop paths in the embedding space.
    """
    def __init__(self, model, processor, device=None):
        """
        Args:
            model: Trained KG embedding model with attributes `ent_emb` and `relation_emb`.
            processor: KGDataProcessor with `entities`, `relations`, `entity2id`, `relation2id`, and `triplets`.
            device: torch.device, defaults to model's device.
        """
        self.model = model
        self.processor = processor
        self.entity2id = processor.entity2id
        self.id2entity = {v: k for k, v in processor.entity2id.items()}
        self.relation2id = processor.relation2id
        self.device = device or next(model.parameters()).device
        # Precompute all entity embeddings
        with torch.no_grad():
            self.ent_embeddings = model.ent_emb.weight.data.to(self.device)

    def _cosine_similarity(self, vec, mat):
        """Compute cosine similarity between vec [D] and mat [N, D]."""
        vec_norm = vec / (vec.norm() + 1e-8)
        mat_norm = mat / (mat.norm(dim=1, keepdim=True) + 1e-8)
        return mat_norm @ vec_norm

    def find_neighborhood_for_poi(self, poi_names, top_k=10):
        """
        Q1: Given a list of POI category names, find top_k communities whose embeddings
        are most similar to the average POI embedding.
        """
        # Get POI embeddings
        poi_ids = []
        for poi in poi_names:
            if poi not in self.entity2id:
                raise ValueError(f"Unknown POI entity: {poi}")
            poi_ids.append(self.entity2id[poi])
        poi_embs = self.ent_embeddings[poi_ids]  # [P, D]
        # Query vector = mean POI embedding
        query = poi_embs.mean(dim=0)            # [D]
        # Filter community entities (heuristic: names containing 'community')
        comm_ids = [idx for name, idx in self.entity2id.items() if 'community' in name.lower()]
        comm_embs = self.ent_embeddings[comm_ids]  # [C, D]
        # Similarities
        sims = self._cosine_similarity(query, comm_embs)  # [C]
        topk = torch.topk(sims, k=top_k)
        results = [(self.id2entity[comm_ids[i]], float(sims[i])) for i in topk.indices]
        return results

    def find_origins_to_home(self, home_community, time_bin, top_k=10):
        """
        Q2: Given a home community and a time_bin, find origin communities with highest
        likelihood of trips at that time_bin terminating at home.
        This uses embedding arithmetic and similarity scoring.
        """
        # Build a query vector: e_home - e_time
        # Embedding for relation 'starts_at'
        rel_id = self.relation2id.get('starts_at')
        if rel_id is None:
            raise ValueError("Relation 'starts_at' not in KG.")
        time_id = self.entity2id.get(time_bin)
        home_id = self.entity2id.get(home_community)
        if time_id is None or home_id is None:
            raise ValueError("Unknown time_bin or community.")
        e_home = self.ent_embeddings[home_id]
        e_time = self.ent_embeddings[time_id]
        e_rel = self.model.relation_emb.weight[rel_id]
        # Score candidate origins by: sim(e_origin + e_rel + e_time, e_home)
        # So query = e_home - e_rel - e_time
        query = e_home - e_rel - e_time
        # Candidate origins: filter by 'community' in name
        origin_ids = [idx for name, idx in self.entity2id.items() if 'community' in name.lower()]
        origin_embs = self.ent_embeddings[origin_ids]
        sims = self._cosine_similarity(query, origin_embs)
        topk = torch.topk(sims, k=top_k)
        results = [(self.id2entity[origin_ids[i]], float(sims[i])) for i in topk.indices]
        return results

    def recommend_reroute(self, current_time_bin, top_k=10):
        """
        Q3: Recommend re-route areas for idle drivers at current time_bin based on
        demand density.
        Strategy: similarity between time-based query and community embeddings.
        """
        # relation 'starts_at' again
        rel_id = self.relation2id.get('starts_at')
        time_id = self.entity2id.get(current_time_bin)
        if rel_id is None or time_id is None:
            raise ValueError("Missing 'starts_at' relation or time_bin entity.")
        # Query = e_time + e_rel
        e_time = self.ent_embeddings[time_id]
        e_rel  = self.model.relation_emb.weight[rel_id]
        query  = e_time + e_rel
        comm_ids = [idx for name, idx in self.entity2id.items() if 'community' in name.lower()]
        comm_embs = self.ent_embeddings[comm_ids]
        sims = self._cosine_similarity(query, comm_embs)
        topk = torch.topk(sims, k=top_k)
        results = [(self.id2entity[comm_ids[i]], float(sims[i])) for i in topk.indices]
        return results

# Example usage:
# reasoner = PathReasoner(model, processor)
# q1 = reasoner.find_neighborhood_for_poi(['place_of_worship', 'supermarket', 'salon'])
# q2 = reasoner.find_origins_to_home('community_42', '18:00')
# q3 = reasoner.recommend_reroute('18:00')
