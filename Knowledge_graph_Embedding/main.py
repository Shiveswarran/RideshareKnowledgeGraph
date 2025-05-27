import argparse
import os
import json
import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim


from kg_data_processor import KGDataProcessor
from helpers import get_optimizer, negative_sampling, apply_regularizer, save_embeddings
from models import KGModels
from models import GIE
from models import MultiHeadDensityMotifAttentionKG
from models import UrbanRidehailKG

from helpers import density_aware_negative_sampling
from helpers import evaluate_density_metrics
# ------------------------------------------------------------------------

# Logging setup
from helpers import setup_logger

# Initialize logger; logs will be saved to the 'logs' folder.
logger = setup_logger(log_dir="logs")
logger.info("Starting KG embedding training...")

# Define a simple PyTorch dataset for triplets.
class TripletDataset(Dataset):
    def __init__(self, triplets):
        """
        Args:
            triplets: list of tuples (head, relation, tail)
        """
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]  # returns a tuple of (head, relation, tail)

def save_config(config, output_dir):
    """
    Save configuration parameters to a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=4)
    logger.info(f"Configuration saved to {config_path}")

def train_step(model, batch, optimizer, margin, num_entities, device,
               motif_lambda=0.1, reg_type=None, reg_weight=0.0,
               neg_sampling="uniform", temperature=1.0):
    ...

    """
    A single training step that computes the margin ranking loss along with an additional motif consistency loss.
    
    For models that return (score, motif_loss), the motif loss is averaged over the batch and multiplied
    by motif_lambda before being added to the ranking loss.
    """
    model.train()
    # Ensure the batch is a tensor on the correct device.
    if isinstance(batch, (tuple, list)) and isinstance(batch[0], torch.Tensor):
        batch = torch.stack([x.to(device) for x in batch], dim=1)
    else:
        batch = torch.tensor(batch, dtype=torch.long, device=device)
        
    # Unpack the batch.
    head, relation, tail = batch[:, 0], batch[:, 1], batch[:, 2]
    
    # Compute positive scores and motif loss.
    pos_result = model(head, relation, tail)
    # If the model returns a tuple, unpack it.
    if isinstance(pos_result, (tuple, list)):
        pos_scores, pos_motif_loss = pos_result
    else:
        pos_scores = pos_result
        pos_motif_loss = torch.zeros_like(pos_scores)
        
    
    
    # Generate negative triplets by corrupting the tail.
    
    # Negative sampling
    if neg_sampling == 'density':
        neg_triplets = density_aware_negative_sampling(
            batch.tolist(), 
            model.entity_density,  # Access the density network
            num_samples=1,
            temperature=temperature,
            device=device
        )
    else:
        neg_triplets = negative_sampling(batch.tolist(), num_entities, num_samples=1)
    
    #neg_triplets = negative_sampling(batch.tolist(), num_entities, num_samples=1)
    
    neg_triplets = torch.tensor(neg_triplets, dtype=torch.long, device=device)
    n_head, n_relation, n_tail = neg_triplets[:, 0], neg_triplets[:, 1], neg_triplets[:, 2]
    neg_result = model(n_head, n_relation, n_tail)
    if isinstance(neg_result, (tuple, list)):
        neg_scores, _ = neg_result
    else:
        neg_scores = neg_result
    
    # Margin ranking loss: we want pos_scores to be less than neg_scores by at least the margin.
    loss_fn = nn.MarginRankingLoss(margin=margin)
    target = -torch.ones_like(pos_scores, device=device)
    ranking_loss = loss_fn(pos_scores, neg_scores, target)
    
    # Combine ranking loss with motif consistency loss.
    loss = ranking_loss + motif_lambda * pos_motif_loss.mean()
    
    # Optionally add regularization.
    if reg_type is not None and reg_weight > 0:
        reg_loss = apply_regularizer(model, reg_type=reg_type)
        loss += reg_weight * reg_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def valid_step(model, dataset, device, margin):
    """
    Evaluate the model on a dataset computing ranking metrics: MRR, Hits@1, Hits@3, and Hits@10.
    
    For each triplet in the dataset, the tail is replaced with all candidate entities.
    """
    model.eval()
    with torch.no_grad():
        all_triplets = torch.tensor(dataset.triplets, dtype=torch.long, device=device)
        head_all, relation_all, tail_all = all_triplets[:, 0], all_triplets[:, 1], all_triplets[:, 2]
        num_triplets = all_triplets.shape[0]
        num_entities = model.num_entities

        ranks = []
        for i in range(num_triplets):
            # For each triplet, repeat the head and relation for all candidate tail entities.
            h = head_all[i].unsqueeze(0).repeat(num_entities)
            r = relation_all[i].unsqueeze(0).repeat(num_entities)
            candidate_tails = torch.arange(num_entities, device=device)
            scores = model(h, r, candidate_tails)
            # If model returns a tuple, extract the score tensor.
            if isinstance(scores, (tuple, list)):
                scores = scores[0]
            # Get the score for the true tail.
            result = model(head_all[i].unsqueeze(0), relation_all[i].unsqueeze(0), tail_all[i].unsqueeze(0))
            if isinstance(result, (tuple, list)):
                target_score = result[0].item()
            else:
                target_score = result.item()
            # Compute rank: number of candidate scores that are less than or equal to the target score.
            rank = torch.sum((scores <= target_score)).item()
            # Ensure rank is at least 1 to avoid division by zero.
            ranks.append(max(rank, 1))
        
        ranks = np.array(ranks, dtype=np.float32)
        mrr = np.mean(1.0 / ranks)
        hits1 = np.mean(ranks <= 1)
        hits3 = np.mean(ranks <= 3)
        hits10 = np.mean(ranks <= 10)
        metrics = {'MRR': mrr, 'Hits@1': hits1, 'Hits@3': hits3, 'Hits@10': hits10}
    return metrics


def main():
    parser = argparse.ArgumentParser(description="KG Embedding Training and Evaluation with Early Stopping")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to CSV file with columns: subject, relation, object")
    parser.add_argument("--model_name", type=str, default="TransE",
                        choices=["TransE", "RotatE", "ReflexE", "MuRE", "ComplexE", "MuRP",
                                 "RotH", "RefH", "ConvE", "TuckER", "GIE","MultiHeadDensityMotifAttentionKG","UrbanRidehailKG"],
                        help="KG embedding model to use")
    parser.add_argument("--motif_lambda", type=float, default=0.1,
                    help="Weight for motif consistency loss (only used for motif-based models)")
    parser.add_argument("--embedding_dim", type=int, default=100,
                        help="Dimension of embeddings")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Margin for ranking loss")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--patience", type=int, default=10,
                        help="Number of epochs with no improvement before early stopping")
    parser.add_argument("--reg_type", type=str, default=None, choices=[None, "N2", "N3"],
                        help="Type of regularization to apply")
    parser.add_argument("--reg_weight", type=float, default=0.0,
                        help="Regularization weight")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save model and configuration")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Enable GPU computation if available")
    parser.add_argument("--neg_sampling", type=str, default="uniform", choices=["uniform","density"], 
                        help="sampling strategy")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature for density sampling")
    parser.add_argument("--gamma", type=float, default=12.0,
                        help="margin/adversarial weight")
     # Ablation flags for MH-DM-KGE
    parser.add_argument("--no_density", action="store_true",
                       help="disable density modulation")
    parser.add_argument("--no_motif_loss", action="store_true",
                        help="set motif consistency weight to zero")
    parser.add_argument("--heads", type=int, default=4,
                        help="number of attention heads H")
    parser.add_argument("--motifs", type=int, default=8,
                        help="number of motif prototypes M per head")
    parser.add_argument("--no_motif_attention", action="store_true",
                        help="disable motif-attention (use diff only)")
    parser.add_argument("--no_projection", action="store_true",
                        help="skip the final projection layer")
    parser.add_argument("--no_rel_attn", action="store_true",
                        help="drop relation embedding from diff/query")
    parser.add_argument("--fix_prototypes", action="store_true",
                        help="freeze motif prototype parameters")

    args = parser.parse_args()

    # Set random seeds for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Save configuration.
    save_config(args, args.output_dir)

    # 1. Load and process data.
    processor = KGDataProcessor(args.data_path, seed=args.seed)
    train_triplets, valid_triplets, test_triplets = processor.split_data(train_ratio=0.8, valid_ratio=0.1)
    num_entities = len(processor.entities)
    num_relations = len(processor.relations)
    logger.info(f"Total entities: {num_entities}, Total relations: {num_relations}")

    # 2. Create Datasets and DataLoaders.
    train_dataset = TripletDataset(train_triplets)
    valid_dataset = TripletDataset(valid_triplets)
    test_dataset = TripletDataset(test_triplets)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True if device.type == "cuda" else False)

    # 3. Initialize the model.
    if args.model_name == "GIE":
        model = GIE(num_entities, num_relations, args.embedding_dim, gamma=args.margin)
    
    elif args.model_name == "UrbanRidehailKG":
            model = UrbanRidehailKG(
        num_entities, num_relations,
        embedding_dim=args.embedding_dim,
        num_heads=args.heads,
        num_motifs=args.motifs,
        gamma=args.gamma,
        use_density=not args.no_density,
        use_motif_attention=not args.no_motif_attention,
        use_projection=not args.no_projection,
        use_relation_attention=not args.no_rel_attn,  
        fix_prototypes=args.fix_prototypes
    )

    elif args.model_name == "MultiHeadDensityMotifAttentionKG":
        model = MultiHeadDensityMotifAttentionKG(num_entities, num_relations, args.embedding_dim)
    else:
        model = KGModels(args.model_name, num_entities, num_relations, args.embedding_dim)
    model = model.to(device)

    # 4. Setup optimizer.
    optimizer = get_optimizer(model.parameters(), optimizer_name="Adam", lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    best_valid_mrr = -1
    best_epoch = 0
    epochs_no_improve = 0

    # 5. Training loop with early stopping.
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            loss = train_step(
                model, batch, optimizer, args.margin, num_entities, device,
                motif_lambda=(0.0 if args.no_motif_loss else args.motif_lambda),
                reg_type=args.reg_type,
                reg_weight=args.reg_weight,
                neg_sampling=args.neg_sampling,
                temperature=args.temperature
            )

            num_batches += 1
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch} | Avg Training Loss: {avg_loss:.4f}")

        # 6. Validation.
        valid_metrics = valid_step(model, valid_dataset, device, args.margin)
        density_metrics = evaluate_density_metrics(model, valid_dataset, device)
        valid_metrics.update(density_metrics)
        scheduler.step(valid_metrics['MRR'])
        logger.info(f"Density Corr: {density_metrics.get('Density_Correlation',0):.4f}")

        logger.info(f"Validation: MRR={valid_metrics['MRR']:.4f}, Hits@1={valid_metrics['Hits@1']:.4f}, "
              f"Hits@3={valid_metrics['Hits@3']:.4f}, Hits@10={valid_metrics['Hits@10']:.4f}")

        # Check for improvement.
        if valid_metrics['MRR'] > best_valid_mrr:
            best_valid_mrr = valid_metrics['MRR']
            best_epoch = epoch
            epochs_no_improve = 0
            model_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            logger.info(f"*** Improved validation MRR. Model saved at epoch {epoch} ***")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} consecutive epoch(s).")
        
        # Early stopping condition.
        if epochs_no_improve >= args.patience:
            logger.info(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
            break

    logger.info(f"Training finished. Best validation MRR: {best_valid_mrr:.4f} at epoch {best_epoch}")

    # 7. Test set evaluation.
    test_metrics = valid_step(model, test_dataset, device, args.margin)
    logger.info("Test Metrics:")
    logger.info(f"MRR: {test_metrics['MRR']:.4f}")
    logger.info(f"Hits@1: {test_metrics['Hits@1']:.4f}")
    logger.info(f"Hits@3: {test_metrics['Hits@3']:.4f}")
    logger.info(f"Hits@10: {test_metrics['Hits@10']:.4f}")
    
    # After computing test_metrics (which might contain numpy.float32 values)
    test_metrics_py = {k: float(v) for k, v in test_metrics.items()}
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(test_metrics_py, f, indent=4)
    logger.info(f"Test metrics saved to {os.path.join(args.output_dir, 'metrics.json')}")


    # 8. Save final entity embeddings if available.
    if hasattr(model, "entity_embeddings"):
        embeddings = model.entity_embeddings.weight.detach().cpu()
        emb_file = os.path.join(args.output_dir, "entity_embeddings.csv")
        save_embeddings(emb_file, embeddings, processor.entity2id)

if __name__ == "__main__":
    main()
