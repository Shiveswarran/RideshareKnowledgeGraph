import torch
import torch.optim as optim
import torch.nn.functional as F

import csv
import logging
import os
import sys
import numpy as np
from scipy.stats import pearsonr

def setup_logger(log_dir, log_filename="training.log", level=logging.INFO):
    """
    Sets up a logger that writes to both a file and the console.
    
    Args:
        log_dir (str): Directory to save the log file.
        log_filename (str): Name of the log file.
        level (int): Logging level (e.g., logging.INFO).
        
    Returns:
        logger: A configured logger.
    """
    # Create log directory if it does not exist.
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    
    # Create a logger instance.
    logger = logging.getLogger("KG_Training")
    logger.setLevel(level)
    
    # Remove any existing handlers.
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create a file handler that logs even debug messages.
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    
    # Create a console handler with a higher log level.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Define a formatter and set it for both handlers.
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger.
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_optimizer(parameters, optimizer_name='Adam', lr=0.001, weight_decay=0):
    """
    Select and return an optimizer based on the given name.
    
    Args:
        parameters: Iterable, model parameters to optimize.
        optimizer_name: str, one of ['Adam', 'Adagrad', 'SGD', 'SparseAdam', 'RAdam', 'RSGD'].
        lr: float, learning rate.
        weight_decay: float, weight decay coefficient (for regular L2 penalty).
        
    Returns:
        optimizer: PyTorch optimizer instance.
    """
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SparseAdam':
        optimizer = optim.SparseAdam(parameters, lr=lr)
    elif optimizer_name == 'RAdam':
        try:
            from torch.optim import RAdam  # May require a recent version of PyTorch.
            optimizer = RAdam(parameters, lr=lr, weight_decay=weight_decay)
        except ImportError:
            raise ImportError("RAdam optimizer not found; please update PyTorch or install a compatible package.")
    elif optimizer_name == 'RSGD':
        # Riemannian SGD requires specialized implementations (e.g., via geoopt)
        # Here we fall back to standard SGD and print a warning.
        optimizer = optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
        print("Warning: RSGD selected but using standard SGD as a placeholder.")
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_name}")
    return optimizer

def negative_sampling(triplets, num_entities, num_samples=1):
    """
    Generate negative samples by corrupting the tail of each triplet.
    
    Args:
        triplets: list of tuples (head, relation, tail) representing positive triplets.
        num_entities: int, total number of unique entities.
        num_samples: int, the number of negative samples to generate per positive triplet.
    
    Returns:
        neg_triplets: list of negative triplets.
    """
    neg_triplets = []
    for (h, r, t) in triplets:
        for _ in range(num_samples):
            # Replace the tail entity with a random entity.
            corrupt_tail = torch.randint(0, num_entities, (1,)).item()
            neg_triplets.append((h, r, corrupt_tail))
    return neg_triplets

def l2_regularizer(tensor):
    """
    Compute the L2 (squared) norm of a tensor.
    
    Args:
        tensor: torch.Tensor
        
    Returns:
        A scalar representing the L2 regularization term.
    """
    return torch.norm(tensor, p=2) ** 2


def l3_regularizer(tensor):
    """
    Compute the L3 (cubic) norm of a tensor.
    
    Args:
        tensor: torch.Tensor
        
    Returns:
        A scalar representing the L3 regularization term.
    """
    return torch.norm(tensor, p=3) ** 3


def apply_regularizer(model, reg_type='N2'):
    """
    Apply a regularization penalty over all parameters of the model.
    
    Args:
        model: PyTorch model.
        reg_type: str, 'N2' applies L2 regularization and 'N3' applies L3 regularization.
    
    Returns:
        A scalar representing the total regularization loss.
    """
    reg_loss = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if reg_type == 'N2':
                reg_loss += l2_regularizer(param)
            elif reg_type == 'N3':
                reg_loss += l3_regularizer(param)
            else:
                raise ValueError("Unsupported regularizer type. Use 'N2' or 'N3'.")
    return reg_loss

def save_embeddings(file_path, entity_embeddings, entity2id):
    """
    Save learned entity embeddings to a CSV file.
    
    Args:
        file_path: str, path to save the CSV file.
        entity_embeddings: torch.Tensor or numpy array of shape (num_entities, embedding_dim).
        entity2id: dict, mapping from entity name to its integer ID.
    
    Output:
        A CSV file where each row maps an entity to its embedding vector.
    """
    # Reverse the mapping to get id -> entity.
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    num_entities, emb_dim = entity_embeddings.shape
    header = ['entity'] + [f'emb_{i}' for i in range(emb_dim)]
    
    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx in range(num_entities):
            entity = id2entity.get(idx, f"unknown_{idx}")
            # Assuming entity_embeddings is a tensor
            emb = entity_embeddings[idx].tolist()
            writer.writerow([entity] + emb)
    print(f"Embeddings saved to {file_path}")
    
# def evaluate_ranking(model, dataset, device):
#     """
#     Evaluate the knowledge graph embedding model using ranking metrics.
    
#     For each triplet in the dataset, this function replaces the tail with all candidate entities,
#     computes the score for each candidate triplet, and then determines the rank of the correct tail.
#     Lower scores (i.e. distances or energies) indicate better (more plausible) triplets.
    
#     Args:
#         model: A PyTorch model with a method .forward(head, relation, tail) that returns a score for each triplet.
#                The model is assumed to have an attribute `num_entities` (total number of unique entities).
#         dataset: A dataset object with an attribute `triplets`, a list of (head, relation, tail) triplets.
#         device: The device (e.g., torch.device("cuda") or torch.device("cpu")) to run evaluation on.
    
#     Returns:
#         metrics: A dictionary containing:
#             - 'MRR': Mean Reciprocal Rank
#             - 'Hits@1': Proportion of triplets with rank <= 1
#             - 'Hits@3': Proportion of triplets with rank <= 3
#             - 'Hits@10': Proportion of triplets with rank <= 10
#     """
#     model.eval()
#     ranks = []

#     with torch.no_grad():
#         # Convert triplets to a tensor and move to device.
#         all_triplets = torch.tensor(dataset.triplets, dtype=torch.long, device=device)
#         head_all = all_triplets[:, 0]
#         relation_all = all_triplets[:, 1]
#         tail_all = all_triplets[:, 2]
#         num_triplets = all_triplets.shape[0]
#         num_entities = model.num_entities

#         for i in range(num_triplets):
#             # For each triplet, repeat the head and relation for all candidate tail entities.
#             h = head_all[i].unsqueeze(0).repeat(num_entities)
#             r = relation_all[i].unsqueeze(0).repeat(num_entities)
#             candidate_tails = torch.arange(num_entities, device=device)
            
#             # Compute scores for all candidate triplets.
#             scores = model(h, r, candidate_tails)  # Expected shape: [num_entities]
            
#             # Get the score for the correct tail.
#             target_score = model(head_all[i].unsqueeze(0), relation_all[i].unsqueeze(0), tail_all[i].unsqueeze(0)).item()
            
#             # Since lower scores indicate better triplets, count the number of candidate scores <= target score.
#             rank = torch.sum((scores <= target_score)).item()
#             ranks.append(rank)

#     ranks = np.array(ranks, dtype=np.float32)
#     mrr = np.mean(1.0 / ranks)
#     hits1 = np.mean(ranks <= 1)
#     hits3 = np.mean(ranks <= 3)
#     hits10 = np.mean(ranks <= 10)

#     metrics = {
#         'MRR': mrr,
#         'Hits@1': hits1,
#         'Hits@3': hits3,
#         'Hits@10': hits10
#     }
#     return metrics

def evaluate_ranking(model, dataset, device):
    """
    Evaluate the KG embedding model using ranking metrics (MRR, Hits@1/3/10),
    correctly unpacking models that return (score, motif_loss).
    """
    model.eval()
    ranks = []

    with torch.no_grad():
        all_triplets = torch.tensor(dataset.triplets, dtype=torch.long, device=device)
        head_all = all_triplets[:, 0]
        rel_all  = all_triplets[:, 1]
        tail_all = all_triplets[:, 2]
        num_entities = model.num_entities

        for i in range(len(head_all)):
            h_i = head_all[i].unsqueeze(0).repeat(num_entities)
            r_i = rel_all[i].unsqueeze(0).repeat(num_entities)
            candidates = torch.arange(num_entities, device=device)

            # compute all candidate scores
            out = model(h_i, r_i, candidates)
            scores = out[0] if isinstance(out, (tuple, list)) else out

            # compute target score
            tgt_out = model(head_all[i].unsqueeze(0),
                            rel_all[i].unsqueeze(0),
                            tail_all[i].unsqueeze(0))
            tgt_score = (tgt_out[0] if isinstance(tgt_out, (tuple, list)) else tgt_out).item()

            # rank = number of candidates with score <= target
            rank = int(torch.sum((scores <= tgt_score)).item())
            ranks.append(max(rank, 1))

    ranks = np.array(ranks, dtype=np.float32)
    return {
        'MRR': np.mean(1.0 / ranks),
        'Hits@1': np.mean(ranks <= 1),
        'Hits@3': np.mean(ranks <= 3),
        'Hits@10': np.mean(ranks <= 10)
    }


def density_aware_negative_sampling(triplets, entity_density, num_samples=1, temperature=1.0, device='cpu'):
    """
    Sample negatives weighted by learned entity_density network.
    entity_density: the nn.Sequential that outputs a [num_entities x embedding_dim] -> [1] scalar.
    """
    negs = []
    ent_ids = torch.arange(entity_density[0].num_embeddings, device=device)
    dens_vals = entity_density(ent_ids).squeeze()  # [E]
    for (h,r,t) in triplets:
        probs = F.softmax(dens_vals / temperature, dim=0)
        for _ in range(num_samples):
            nt = torch.multinomial(probs, 1).item()
            negs.append((h, r, nt))
    return negs



def evaluate_density_metrics(model, dataset, device):
    """
    Compute Pearson correlation between predicted densities
    and simple count-based densities as a sanity check.
    """
    if not hasattr(model, 'entity_density'):
        return {}
    model.eval()
    with torch.no_grad():
        H = torch.tensor([h for h,_,_ in dataset.triplets], device=device)
        T = torch.tensor([t for _,_,t in dataset.triplets], device=device)
        pred_h = model.entity_density(H).squeeze().cpu().numpy()
        pred_t = model.entity_density(T).squeeze().cpu().numpy()
        pred = (pred_h + pred_t)/2
        # actual counts
        counts = np.bincount(np.concatenate([H.cpu().numpy(), T.cpu().numpy()]))
        counts = counts / counts.max()
        actual = (counts[H.cpu().numpy()]+counts[T.cpu().numpy()])/2
        corr, _ = pearsonr(pred, actual)
    return {"Density_Correlation": corr}

