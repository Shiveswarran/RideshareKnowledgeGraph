import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class KGModels(nn.Module):
    def __init__(self, model_name, num_entities, num_relations, embedding_dim, **kwargs):
        """
        Initialize a KG embedding model.

        Args:
            model_name (str): One of ['TransE', 'RotatE', 'ReflexE', 'MuRE', 'ComplexE',
                'MuRP', 'RotH', 'RefH', 'ConvE', 'TuckER'].
            num_entities (int): Number of entities.
            num_relations (int): Number of relations.
            embedding_dim (int): Embedding dimension.
            kwargs: Additional keyword arguments (e.g., for ConvE: conv_channels, kernel_size).
        """
        super(KGModels, self).__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations

        if model_name in ['TransE', 'RotatE', 'ReflexE']:
            # Standard real-value embeddings.
            self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
            self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        if model_name == 'MuRE':
            # MuRE uses a weight and a bias part per relation.
            self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
            self.relation_weight = nn.Embedding(num_relations, embedding_dim)
            self.relation_bias = nn.Embedding(num_relations, embedding_dim)
        
        if model_name == 'ComplexE':
            # ComplEx requires complex embeddings: use double dimension.
            self.entity_embeddings = nn.Embedding(num_entities, 2 * embedding_dim)
            self.relation_embeddings = nn.Embedding(num_relations, 2 * embedding_dim)
        
        if model_name == 'MuRP':
            # Similar to MuRE but with normalization (this is one possible formulation).
            self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
            self.relation_weight = nn.Embedding(num_relations, embedding_dim)
            self.relation_bias = nn.Embedding(num_relations, embedding_dim)
        
        if model_name in ['RotH', 'RefH']:
            # Hyperbolic models: here we use a similar architecture as RotatE but later apply a hyperbolic transformation.
            # One common trick is to split the embedding into two halves.
            self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
            self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        if model_name == 'ConvE':
            # ConvE uses convolution layers.
            self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
            self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
            self.conv_channels = kwargs.get('conv_channels', 32)
            self.kernel_size = kwargs.get('kernel_size', 3)
            # For ConvE, we reshape each embedding to a square.
            self.embedding_height = int(math.sqrt(embedding_dim))
            self.embedding_width = self.embedding_height
            self.conv_layer = nn.Conv2d(2, self.conv_channels, kernel_size=self.kernel_size, padding=1)
            self.fc = nn.Linear(self.conv_channels * self.embedding_height * self.embedding_width, embedding_dim)
        
        if model_name == 'TuckER':
            # TuckER uses a core tensor to model interactions.
            self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
            self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
            # Core tensor of shape (embedding_dim, embedding_dim, embedding_dim)
            self.core_tensor = nn.Parameter(torch.randn(embedding_dim, embedding_dim, embedding_dim))
    
    def forward(self, head_idx, relation_idx, tail_idx):
        """
        Dispatch the forward pass to the corresponding scoring function.
        Args:
            head_idx, relation_idx, tail_idx: torch.LongTensor indices.
        Returns:
            Score (or distance) for each triplet.
        """
        if self.model_name == 'TransE':
            return self.transE_score(head_idx, relation_idx, tail_idx)
        elif self.model_name == 'RotatE':
            return self.rotatE_score(head_idx, relation_idx, tail_idx)
        elif self.model_name == 'ReflexE':
            return self.reflexE_score(head_idx, relation_idx, tail_idx)
        elif self.model_name == 'MuRE':
            return self.mure_score(head_idx, relation_idx, tail_idx)
        elif self.model_name == 'ComplexE':
            return self.complexE_score(head_idx, relation_idx, tail_idx)
        elif self.model_name == 'MuRP':
            return self.murp_score(head_idx, relation_idx, tail_idx)
        elif self.model_name == 'RotH':
            return self.roth_score(head_idx, relation_idx, tail_idx)
        elif self.model_name == 'RefH':
            return self.refh_score(head_idx, relation_idx, tail_idx)
        elif self.model_name == 'ConvE':
            return self.conve_score(head_idx, relation_idx, tail_idx)
        elif self.model_name == 'TuckER':
            return self.tucker_score(head_idx, relation_idx, tail_idx)
        else:
            raise NotImplementedError(f"{self.model_name} is not implemented.")

    def transE_score(self, head_idx, relation_idx, tail_idx):
        """
        TransE: Score = || h + r - t || (L1 norm).
        """
        h = self.entity_embeddings(head_idx)
        r = self.relation_embeddings(relation_idx)
        t = self.entity_embeddings(tail_idx)
        score = torch.norm(h + r - t, p=1, dim=-1)
        return score

    def rotatE_score(self, head_idx, relation_idx, tail_idx):
        h = self.entity_embeddings(head_idx)
        t = self.entity_embeddings(tail_idx)
        h_re, h_im = torch.chunk(h, 2, dim=-1)
        t_re, t_im = torch.chunk(t, 2, dim=-1)
        r = self.relation_embeddings(relation_idx)
        # Slice the relation embedding to match the dimension of h_re (which is D/2).
        r = r[:, :h_re.shape[1]]
        # Map relation to phase representation.
        r_re = torch.cos(r)
        r_im = torch.sin(r)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im
        score = torch.norm(torch.cat([diff_re, diff_im], dim=-1), p=1, dim=-1)
        return score


    def reflexE_score(self, head_idx, relation_idx, tail_idx):
        """
        ReflexE: A sample bilinear scoring function.
        Here we use: score = - (h ∘ r ∘ t). Sum over the element-wise multiplication.
        """
        h = self.entity_embeddings(head_idx)
        r = self.relation_embeddings(relation_idx)
        t = self.entity_embeddings(tail_idx)
        score = -torch.sum(h * r * t, dim=-1)
        return score

    def mure_score(self, head_idx, relation_idx, tail_idx):
        """
        MuRE: Represent relation as (r_weight, r_bias).
        Score = || (h ∘ r_weight + r_bias) - t || (L1 norm).
        """
        h = self.entity_embeddings(head_idx)
        t = self.entity_embeddings(tail_idx)
        r_w = self.relation_weight(relation_idx)
        r_b = self.relation_bias(relation_idx)
        score = torch.norm(h * r_w + r_b - t, p=1, dim=-1)
        return score

    def complexE_score(self, head_idx, relation_idx, tail_idx):
        """
        ComplEx: Score = Re(<h, r, conjugate(t)>).
        After splitting embeddings into real and imaginary parts.
        """
        x = self.entity_embeddings(head_idx)
        r = self.relation_embeddings(relation_idx)
        y = self.entity_embeddings(tail_idx)
        x_re, x_im = torch.chunk(x, 2, dim=-1)
        r_re, r_im = torch.chunk(r, 2, dim=-1)
        y_re, y_im = torch.chunk(y, 2, dim=-1)
        score = (x_re * r_re * y_re +
                 x_re * r_im * y_im +
                 x_im * r_re * y_im -
                 x_im * r_im * y_re).sum(dim=-1)
        return -score  # Negate for margin-based losses

    def murp_score(self, head_idx, relation_idx, tail_idx):
        """
        MuRP: A variant of MuRE in polar coordinate space.
        Here we normalize the entity embeddings before applying a MuRE-like operation.
        Score = || (normalize(h) ∘ r_weight + r_bias) - normalize(t) ||.
        """
        h = self.entity_embeddings(head_idx)
        t = self.entity_embeddings(tail_idx)
        r_w = self.relation_weight(relation_idx)
        r_b = self.relation_bias(relation_idx)
        h_norm = F.normalize(h, p=2, dim=-1)
        t_norm = F.normalize(t, p=2, dim=-1)
        score = torch.norm(h_norm * r_w + r_b - t_norm, p=1, dim=-1)
        return score

    def roth_score(self, head_idx, relation_idx, tail_idx):
        """
        RotH: A hyperbolic variant of RotatE.
        Use the same rotation operation, then transform the Euclidean distance into a hyperbolic distance.
        For demonstration, we use: score = acosh(1 + d^2) where d is Euclidean distance.
        """
        h = self.entity_embeddings(head_idx)
        t = self.entity_embeddings(tail_idx)
        h_re, h_im = torch.chunk(h, 2, dim=-1)
        t_re, t_im = torch.chunk(t, 2, dim=-1)
        r = self.relation_embeddings(relation_idx)
        r_re = torch.cos(r)
        r_im = torch.sin(r)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        diff = torch.cat([hr_re - t_re, hr_im - t_im], dim=-1)
        euclidean_dist = torch.norm(diff, p=2, dim=-1)
        score = torch.acosh(1 + euclidean_dist**2)
        return score

    def refh_score(self, head_idx, relation_idx, tail_idx):
        """
        RefH: A hyperbolic variant of ReflexE.
        Compute a simple bilinear form then map using acosh.
        """
        h = self.entity_embeddings(head_idx)
        r = self.relation_embeddings(relation_idx)
        t = self.entity_embeddings(tail_idx)
        bilinear = torch.sum(h * r * t, dim=-1)
        score = torch.acosh(1 + torch.abs(bilinear))
        return score

    def conve_score(self, head_idx, relation_idx, tail_idx):
        """
        ConvE: Convolution-based scoring.
        Reshape head and relation embeddings to 2D, concatenate, process with conv and fc layers,
        and then compute the dot product with tail embedding.
        """
        h = self.entity_embeddings(head_idx)  # [batch, embedding_dim]
        r = self.relation_embeddings(relation_idx)
        t = self.entity_embeddings(tail_idx)
        batch_size = h.shape[0]
        h_2d = h.view(batch_size, 1, self.embedding_height, self.embedding_width)
        r_2d = r.view(batch_size, 1, self.embedding_height, self.embedding_width)
        hr_cat = torch.cat([h_2d, r_2d], dim=1)  # [batch, 2, H, W]
        conv_out = self.conv_layer(hr_cat)             # [batch, conv_channels, H, W]
        conv_out = conv_out.view(batch_size, -1)
        fc_out = self.fc(conv_out)                       # [batch, embedding_dim]
        score = torch.sum(fc_out * t, dim=-1)
        return score

    def tucker_score(self, head_idx, relation_idx, tail_idx):
        """
        TuckER: Compute score = h^T * W_r * t, where W_r is relation-specific.
        W_r is obtained by contracting the core tensor with the relation embedding.
        """
        h = self.entity_embeddings(head_idx)  # [batch, d]
        r = self.relation_embeddings(relation_idx)  # [batch, d]
        t = self.entity_embeddings(tail_idx)  # [batch, d]
        # Compute relation-specific transformation: contract r with core_tensor over first dimension.
        # Resulting W_r has shape [batch, d, d]
        W_r = torch.tensordot(r, self.core_tensor, dims=([1], [0]))
        h_W = torch.bmm(h.unsqueeze(1), W_r)         # [batch, 1, d]
        score = torch.bmm(h_W, t.unsqueeze(2)).squeeze()  # [batch]
        return score

class GIE(nn.Module):
    """
    Geometric Interactive Embedding (GIE) Model.
    
    GIE applies a geometric transformation on entity embeddings under the guidance 
    of relation embeddings. In this implementation, the relation embedding is used 
    as an interaction operator that scales the entity embeddings via a sigmoid activation.
    
    The transformation on an entity embedding x given a relation r is defined as:
    
        f(x, r) = x * sigmoid(r)
    
    The score for a triplet (h, r, t) is then computed as:
    
        score = γ - || f(h, r) - f(t, r) ||₂
        
    where γ is a margin hyperparameter.
    """
    def __init__(self, num_entities, num_relations, embedding_dim, gamma=12.0, epsilon=2.0):
        """
        Initialize the GIE model.

        Args:
            num_entities (int): Total number of entities.
            num_relations (int): Total number of relations.
            embedding_dim (int): Dimension for both entity and relation embeddings.
            gamma (float): Margin hyperparameter, controls the score scale.
            epsilon (float): Small constant to adjust the embedding range.
        """
        super(GIE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.epsilon = epsilon

        # Compute an embedding range based on gamma and epsilon
        self.embedding_range = (gamma + epsilon) / embedding_dim

        # Initialize the entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        nn.init.uniform_(self.entity_embeddings.weight,
                         -self.embedding_range, self.embedding_range)
        nn.init.uniform_(self.relation_embeddings.weight,
                         -self.embedding_range, self.embedding_range)

    def forward(self, head_idx, relation_idx, tail_idx):
        """
        Forward pass for GIE.

        Args:
            head_idx (torch.LongTensor): Indices for the head entities.
            relation_idx (torch.LongTensor): Indices for the relations.
            tail_idx (torch.LongTensor): Indices for the tail entities.

        Returns:
            score (torch.Tensor): A tensor containing a score for each triplet.
        """
        # Retrieve the embeddings for head, relation, and tail.
        head = self.entity_embeddings(head_idx)         # shape: [batch_size, embedding_dim]
        relation = self.relation_embeddings(relation_idx) # shape: [batch_size, embedding_dim]
        tail = self.entity_embeddings(tail_idx)           # shape: [batch_size, embedding_dim]

        # Geometric interactive transformation:
        # Each entity embedding is scaled by a factor derived from the relation embedding.
        # You can experiment with alternative interactive functions if needed.
        transformed_head = head * torch.sigmoid(relation)
        transformed_tail = tail * torch.sigmoid(relation)

        # Compute the Euclidean (L2) distance between the transformed head and tail.
        distance = torch.norm(transformed_head - transformed_tail, p=2, dim=-1)
        
        # Define the triplet score: higher scores indicate more plausible triplets.
        # Here we subtract the computed distance from the margin γ.
        score = self.gamma - distance
        return score
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# class MotifAttentionKG(nn.Module):
#     """
#     MotifAttentionKG implements a motif-driven KGE with explicit motif influence.
    
#     For each triplet (h, r, t), the model computes:
#       - Standard embeddings: e_h, e_r, e_t,
#       - A motif influence φ(h, r, t) via a feed-forward network on [e_h; e_r; e_t],
#       - A score using a TransE-like formulation:
#             score = || φ(h, r, t) ⊙ (e_h + e_r - e_t) ||_2,
#       - And a motif consistency loss:
#             L_motif = ||(e_h + e_r - e_t) - φ(h, r, t)||_2².
#     """
#     def __init__(self, num_entities, num_relations, embedding_dim):
#         super(MotifAttentionKG, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.num_entities = num_entities
#         self.num_relations = num_relations

#         # Standard embedding layers.
#         self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
#         self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
#         # Initialize using uniform distribution.
#         bound = 6.0 / math.sqrt(embedding_dim)
#         nn.init.uniform_(self.entity_embeddings.weight, -bound, bound)
#         nn.init.uniform_(self.relation_embeddings.weight, -bound, bound)
        
#         # Motif influence network φ: input is [e_h; e_r; e_t] of size 3*embedding_dim,
#         # output is a vector of size embedding_dim.
#         self.phi_net = nn.Sequential(
#             nn.Linear(3 * embedding_dim, embedding_dim),
#             nn.ReLU(),
#             nn.Linear(embedding_dim, embedding_dim),
#             nn.Sigmoid()  # Force outputs to be in (0, 1); adjust activation if needed.
#         )
    
#     def forward(self, head_idx, relation_idx, tail_idx):
#         # Look up embeddings.
#         e_h = self.entity_embeddings(head_idx)      # shape: [batch, embedding_dim]
#         e_r = self.relation_embeddings(relation_idx)  # shape: [batch, embedding_dim]
#         e_t = self.entity_embeddings(tail_idx)        # shape: [batch, embedding_dim]
        
#         # Compute motif influence φ(h, r, t)
#         phi_input = torch.cat([e_h, e_r, e_t], dim=-1)  # shape: [batch, 3 * embedding_dim]
#         phi = self.phi_net(phi_input)                   # shape: [batch, embedding_dim]
        
#         # Compute standard difference.
#         diff = e_h + e_r - e_t  # shape: [batch, embedding_dim]
        
#         # Compute modulated difference with elementwise multiplication.
#         modulated = phi * diff
#         # Compute TransE-like score using L2 norm (lower scores indicate better plausibility).
#         score = torch.norm(modulated, p=2, dim=-1)
        
#         # Motif consistency loss: enforce that φ(h,r,t) approximates diff.
#         motif_loss = torch.norm(diff - phi, p=2, dim=-1) ** 2
        
#         # Return both the score and the motif consistency loss.
#         return score, motif_loss

class MultiHeadDensityMotifAttentionKG(nn.Module):
    """
    Motif-driven KGE with multi-head attention over learned motif prototypes,
    modulated by a density factor per entity.
    """
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 num_heads: int = 4,
                 num_motifs: int = 8):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # 1) Standard embeddings
        self.entity_embeddings   = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        # Density scalar per entity
        self.density_weights     = nn.Embedding(num_entities, 1)

        # 2) Learned motif prototypes: one per head
        # Shape: [num_heads, num_motifs, head_dim]
        self.motif_prototypes = nn.Parameter(torch.randn(num_heads, num_motifs, self.head_dim))

        # 3) Linear maps for Q, K, V per head
        self.q_linears = nn.ModuleList([
            nn.Linear(embedding_dim, self.head_dim, bias=False)
            for _ in range(num_heads)
        ])
        self.k_linears = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim, bias=False)
            for _ in range(num_heads)
        ])
        self.v_linears = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim, bias=False)
            for _ in range(num_heads)
        ])

        # 4) Output projection back to embedding_dim
        self.out_proj = nn.Linear(embedding_dim + embedding_dim, embedding_dim)

        # Initialization
        bound = 6.0 / math.sqrt(embedding_dim)
        nn.init.uniform_(self.entity_embeddings.weight,   -bound, bound)
        nn.init.uniform_(self.relation_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.density_weights.weight,     -1.0,   1.0)

    def forward(self, h_idx, r_idx, t_idx):
        # a) Lookup embeddings
        e_h = self.entity_embeddings(h_idx)    # [B, D]
        e_r = self.relation_embeddings(r_idx)  # [B, D]
        e_t = self.entity_embeddings(t_idx)    # [B, D]
        diff = e_h + e_r - e_t                 # [B, D]

        # b) Compute density factor
        d_h = torch.sigmoid(self.density_weights(h_idx)).squeeze(-1)  # [B]
        d_t = torch.sigmoid(self.density_weights(t_idx)).squeeze(-1)  # [B]
        density = ((d_h + d_t) / 2).unsqueeze(-1)                     # [B, 1]

        # c) Multi-head motif attention
        head_outputs = []
        for i in range(self.num_heads):
            # Query = linear(diff) -> [B, head_dim]
            q = self.q_linears[i](diff)
            # Keys = linear(prototypes) -> [num_motifs, head_dim]
            k = self.k_linears[i](self.motif_prototypes[i])
            # Values = linear(prototypes)
            v = self.v_linears[i](self.motif_prototypes[i])

            # Attention scores: (q · kᵀ) / sqrt(head_dim) plus density bias
            # q: [B, H], k: [M, H] => scores [B, M]
            scores = (q @ k.t()) / math.sqrt(self.head_dim)
            scores = scores + density  # broadcast [B,1] -> [B,M]
            weights = F.softmax(scores, dim=-1)     # [B, M]

            # Weighted sum of v: [B, M] @ [M, H] -> [B, H]
            head_out = weights @ v
            head_outputs.append(head_out)

        # d) Concatenate heads: [B, num_heads*head_dim] == [B, D]
        motif_attn = torch.cat(head_outputs, dim=-1)

        # e) Motif consistency loss
        motif_loss = torch.norm(diff - motif_attn, p=2, dim=-1).pow(2)  # [B]

        # f) Combine diff and motif_attn, project and modulate by density
        combined = torch.cat([diff, motif_attn], dim=-1)  # [B, 2D]
        projected = self.out_proj(combined)               # [B, D]
        final = projected * density                        # [B, D]

        # g) Final score: L2 norm
        score = torch.norm(final, p=2, dim=-1)  # [B]

        return score, motif_loss
    

# class UrbanRidehailKG(nn.Module):
#     """
#     Enhanced urban ridehail KG embedding model with:
#     - Multi-scale density modeling (entity and relation-aware)
#     - Hierarchical attention over motifs
#     - Self-adversarial negative sampling
#     - Enhanced scoring function
#     """
#     def __init__(self, num_entities, num_relations,
#                  embedding_dim=100, num_heads=4, num_motifs=8, gamma=12.0):
#         super().__init__()
#         self.num_entities  = num_entities
#         self.num_relations = num_relations
#         self.embedding_dim = embedding_dim
#         self.num_heads     = num_heads
#         self.head_dim      = embedding_dim // num_heads
#         self.gamma         = gamma

#         # 1) Core embeddings
#         self.entity_emb   = nn.Embedding(num_entities, embedding_dim)
#         self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        
#         # 2) Density modeling (multi-scale)
#         self.entity_density = nn.Sequential(
#             nn.Embedding(num_entities, embedding_dim),
#             nn.Linear(embedding_dim, 1),
#             nn.Sigmoid()
#         )
#         self.relation_density = nn.Sequential(
#             nn.Embedding(num_relations, embedding_dim),
#             nn.Linear(embedding_dim, 1),
#             nn.Sigmoid()
#         )

#         # 3) Hierarchical motif attention
#         self.local_motifs  = nn.Parameter(torch.randn(num_motifs, self.head_dim))
#         self.global_motifs = nn.Parameter(torch.randn(num_motifs, self.head_dim))
#         self.query_proj = nn.Linear(embedding_dim, embedding_dim)
#         self.key_proj   = nn.ModuleList([
#             nn.Linear(self.head_dim, self.head_dim) for _ in range(2)
#         ])

#         # 4) Self-adversarial parameters
#         self.adv_weight = nn.Parameter(torch.tensor(1.0))
#         self.adv_bias   = nn.Parameter(torch.tensor(0.0))

#         # 5) Output transformation
#         self.output_proj = nn.Sequential(
#             nn.Linear(2 * embedding_dim, embedding_dim),
#             nn.Tanh()
#         )

#         self._init_weights()

#     def _init_weights(self):
#         nn.init.xavier_uniform_(self.entity_emb.weight)
#         nn.init.xavier_uniform_(self.relation_emb.weight)
#         nn.init.xavier_uniform_(self.local_motifs)
#         nn.init.xavier_uniform_(self.global_motifs)

#     def _compute_density(self, h_idx, r_idx, t_idx):
#         h_d = self.entity_density(h_idx)   # [B,1]
#         t_d = self.entity_density(t_idx)   # [B,1]
#         r_d = self.relation_density(r_idx) # [B,1]
#         return (h_d + t_d + r_d) / 3.0     # [B,1]

#     def _hierarchical_attention(self, h_emb, r_emb, t_emb, density):
#         B,D = h_emb.size()
#         query = self.query_proj(h_emb + r_emb + t_emb)
#         query = query.view(B, self.num_heads, self.head_dim)  # [B,H,d]
#         # keys
#         local_k  = self.key_proj[0](self.local_motifs)   # [M,d]
#         global_k = self.key_proj[1](self.global_motifs)  # [M,d]
#         # scores
#         l_scores = (query @ local_k.t()) / math.sqrt(self.head_dim)  # [B,H,M]
#         g_scores = (query @ global_k.t()) / math.sqrt(self.head_dim)
#         # density bias
#         l_scores = l_scores + density.unsqueeze(-1)
#         g_scores = g_scores + density.unsqueeze(-1)
#         # weights
#         l_w = F.softmax(l_scores, dim=-1)
#         g_w = F.softmax(g_scores, dim=-1)
#         return (l_w + g_w) / 2.0  # [B,H,M]

#     def forward(self, h_idx, r_idx, t_idx):
#         h = self.entity_emb(h_idx); r = self.relation_emb(r_idx); t = self.entity_emb(t_idx)
#         density = self._compute_density(h_idx, r_idx, t_idx)  # [B,1]
#         # hierarchical attention
#         attn_w = self._hierarchical_attention(h, r, t, density)  # [B,H,M]
#         # motif repr
#         motifs = (self.local_motifs + self.global_motifs).unsqueeze(0)  # [1,M,d]
#         rep = (attn_w.unsqueeze(-1) * motifs).sum(dim=2)  # [B,H,d]
#         motif_repr = rep.view(-1, self.embedding_dim)    # [B,D]
#         # combine
#         diff = h + r - t
#         combined = torch.cat([diff, motif_repr], dim=-1)  # [B,2D]
#         transformed = self.output_proj(combined)          # [B,D]
#         score = transformed.norm(p=2, dim=-1)             # [B]
#         score = self.adv_weight * score + self.adv_bias
#         motif_loss = (diff - motif_repr).norm(p=2, dim=-1).pow(2)
#         return score, motif_loss
    
    

# models.py (overwrite the old UrbanRidehailKG)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class UrbanRidehailKG(nn.Module):
    """
    Enhanced urban ridehail KG embedding with ablation flags:
      - use_density:           disable learned density
      - use_motif_attention:   disable motif attention
      - use_projection:        disable final projection layer
      - use_relation_attention:disable relation in the attention context
      - fix_prototypes:        freeze prototype parameters
    """
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int = 100,
                 num_heads: int = 4,
                 num_motifs: int = 8,
                 gamma: float = 12.0,
                 use_density: bool = True,
                 use_motif_attention: bool = True,
                 use_projection: bool = True,
                 use_relation_attention: bool = True,
                 fix_prototypes: bool = False):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.num_entities            = num_entities
        self.num_relations           = num_relations
        self.embedding_dim           = embedding_dim
        self.num_heads               = num_heads
        self.head_dim                = embedding_dim // num_heads
        self.gamma                   = gamma

        # Ablation flags
        self.use_density             = use_density
        self.use_motif_attention     = use_motif_attention
        self.use_projection          = use_projection
        self.use_relation_attention  = use_relation_attention

        # 1) Core embeddings
        self.entity_emb   = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        # 2) Density networks
        self.entity_density   = nn.Sequential(
            nn.Embedding(num_entities, embedding_dim),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        self.relation_density = nn.Sequential(
            nn.Embedding(num_relations, embedding_dim),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

        # 3) Motif prototypes & projections
        self.local_motifs  = nn.Parameter(torch.randn(num_motifs, self.head_dim))
        self.global_motifs = nn.Parameter(torch.randn(num_motifs, self.head_dim))
        if fix_prototypes:
            self.local_motifs.requires_grad_(False)
            self.global_motifs.requires_grad_(False)

        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj   = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim),
            nn.Linear(self.head_dim, self.head_dim),
        ])

        # 4) Self-adversarial components
        self.adv_weight = nn.Parameter(torch.tensor(1.0))
        self.adv_bias   = nn.Parameter(torch.tensor(0.0))

        # 5) Output projection
        if self.use_projection:
            self.output_proj = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                nn.Tanh()
            )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.local_motifs)
        nn.init.xavier_uniform_(self.global_motifs)

    def _compute_density(self, h_idx, r_idx, t_idx):
        if not self.use_density:
            B = h_idx.size(0)
            return torch.ones(B, 1, device=h_idx.device)
        h_d = self.entity_density(h_idx)
        t_d = self.entity_density(t_idx)
        r_d = self.relation_density(r_idx)
        return (h_d + t_d + r_d) / 3.0

    def _hierarchical_attention(self, h_emb, r_emb, t_emb, density):
        B, _ = h_emb.size()
        ctx = h_emb + (r_emb if self.use_relation_attention else 0) + t_emb
        query = self.query_proj(ctx).view(B, self.num_heads, self.head_dim)
        local_k  = self.key_proj[0](self.local_motifs)   # [M, d]
        global_k = self.key_proj[1](self.global_motifs)  # [M, d]
        l_scores = (query @ local_k.t()) / math.sqrt(self.head_dim)
        g_scores = (query @ global_k.t()) / math.sqrt(self.head_dim)
        l_scores = l_scores + density.unsqueeze(-1)
        g_scores = g_scores + density.unsqueeze(-1)
        l_w = F.softmax(l_scores, dim=-1)
        g_w = F.softmax(g_scores, dim=-1)
        return (l_w + g_w) / 2.0

    def forward(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)

        density = self._compute_density(h_idx, r_idx, t_idx)
        diff    = h + r - t

        if self.use_motif_attention:
            attn_w = self._hierarchical_attention(h, r, t, density)  # [B,H,M]
            motifs = (self.local_motifs + self.global_motifs).unsqueeze(0)  # [1,M,d]
            rep    = (attn_w.unsqueeze(-1) * motifs).sum(dim=2)  # [B,H,d]
            motif_repr = rep.view(-1, self.embedding_dim)
        else:
            motif_repr = torch.zeros_like(diff)

        motif_loss = (diff - motif_repr).norm(p=2, dim=-1).pow(2)

        if self.use_projection:
            combined    = torch.cat([diff, motif_repr], dim=-1)
            transformed = self.output_proj(combined)
        else:
            transformed = diff

        score = transformed.norm(p=2, dim=-1)
        score = self.adv_weight * score + self.adv_bias

        return score, motif_loss
