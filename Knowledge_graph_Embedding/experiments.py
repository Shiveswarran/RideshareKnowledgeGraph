# experiments.py

import argparse
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from kg_data_processor import KGDataProcessor
from helpers import get_optimizer, apply_regularizer, evaluate_ranking
from models import KGModels, UrbanRidehailKG

# reuse your train_step and valid_step from main.py
from main import train_step, valid_step, TripletDataset  

def run_one(model, train_ds, valid_ds, test_ds, processor, device, args):
    """
    Train `model` on train_ds, early-stop on valid_ds, then return test metrics.
    """
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    optimizer = get_optimizer(model.parameters(), lr=args.lr, weight_decay=args.reg_weight)

    best_mrr = -np.inf
    wait = 0

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        for batch in train_loader:
            _ = train_step(
                model, batch, optimizer, args.margin, 
                len(processor.entities), device,
                motif_lambda=args.motif_lambda,
                reg_type=args.reg_type,
                reg_weight=args.reg_weight,
                neg_sampling=args.neg_sampling,
                temperature=args.temperature
            )

        # Validation
        vm = valid_step(model, valid_ds, device, args.margin)
        if vm['MRR'] > best_mrr:
            best_mrr = vm['MRR']
            wait = 0
            torch.save(model.state_dict(), "tmp_best.pt")
        else:
            wait += 1
            if wait >= args.patience:
                break

    # Load best model and evaluate on TEST split
    model.load_state_dict(torch.load("tmp_best.pt", map_location=device))
    test_metrics = evaluate_ranking(model, test_ds, device)
    return test_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_csv", default="results.csv")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--motif_lambda", type=float, default=0.1)
    parser.add_argument("--reg_type", choices=[None,"N2","N3"], default=None)
    parser.add_argument("--reg_weight", type=float, default=0.0)
    parser.add_argument("--neg_sampling", choices=["uniform","density"], default="uniform")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    # Load and split data
    proc = KGDataProcessor(args.data_path)
    train_t, valid_t, test_t = proc.split_data(train_ratio=0.8, valid_ratio=0.1)
    train_ds = TripletDataset(train_t)
    valid_ds = TripletDataset(valid_t)
    test_ds  = TripletDataset(test_t)

    models = [ "MuRP", "MuRE", "ConvE"] #"UrbanRidehailKG",
    dims   = [32, 64, 128, 200, 400]

    results = []
    for name in models:
        for D in dims:
            # Instantiate model
            if name == "UrbanRidehailKG":
                m = UrbanRidehailKG(
                    len(proc.entities), len(proc.relations),
                    embedding_dim=D
                ).to(device)
            else:
                m = KGModels(
                    model_name=name,
                    num_entities=len(proc.entities),
                    num_relations=len(proc.relations),
                    embedding_dim=D
                ).to(device)

            # Train & evaluate
            tm = run_one(m, train_ds, valid_ds, test_ds, proc, device, args)

            # Collect test metrics
            results.append({
                "model": name,
                "dim": D,
                "Test_MRR": tm["MRR"],
                "Test_Hits@1": tm["Hits@1"],
                "Test_Hits@3": tm["Hits@3"],
                "Test_Hits@10": tm["Hits@10"]
            })
            print(f"{name:20s} dim={D:3d}  Test MRR={tm['MRR']:.4f}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved results to {args.output_csv}")

    # Plot TEST MRR
    plt.figure(figsize=(8,5))
    sns.lineplot(
        data=df, x="dim", y="Test_MRR", hue="model",
        marker="o", linewidth=2
    )
    plt.title("Test MRR vs Embedding Dimension")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Test MRR")
    plt.xticks(dims)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig("test_mrr_comparison.png", dpi=300)
    print("Saved plot as test_mrr_comparison.png")

if __name__ == "__main__":
    main()

