#!/bin/bash
# run_all.sh
# This script runs all the specified KG embedding models with preset margin parameters,
# runs training for 200 epochs with a patience of 20 epochs for early stopping,
# and collects the test metrics into a CSV file.
# It also saves each model's best model and entity embeddings in a separate folder.

# Set common parameters.
DATA_PATH="/scratch/umni5/a/shives/Research_projects/Knowledge_graphs/RideshareKnowledgeGraph/Data/knowledge_graph_triplets.csv"   
EPOCHS=200
PATIENCE=20
BATCH_SIZE=1024
EMBEDDING_DIM=64
LR=0.001

# Define the models and their corresponding margin hyperparameters.
declare -A model_margin
model_margin["TransE"]=1.0
model_margin["RotatE"]=2.0
model_margin["ReflexE"]=1.0
model_margin["MuRE"]=1.0
model_margin["ComplexE"]=1.0
model_margin["MuRP"]=1.0
model_margin["RotH"]=2.0
model_margin["RefH"]=2.0
model_margin["ConvE"]=1.0
#model_margin["TuckER"]=1.0
model_margin["GIE"]=12.0

# CSV file to accumulate results.
RESULTS_CSV="results.csv"
echo "Model,MRR,Hits@1,Hits@3,Hits@10" > $RESULTS_CSV

# Loop over each model.
for model in "${!model_margin[@]}"; do
    margin=${model_margin[$model]}
    OUTPUT_DIR="output/${model}"
    mkdir -p $OUTPUT_DIR
    echo "------------------------------------------------"
    echo "Running model: $model with margin = $margin"
    echo "Output directory: $OUTPUT_DIR"

    # Run main.py with the appropriate parameters.
    python main.py \
      --data_path $DATA_PATH \
      --model_name $model \
      --embedding_dim $EMBEDDING_DIM \
      --lr $LR \
      --margin $margin \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --patience $PATIENCE \
      --output_dir $OUTPUT_DIR \
      --use_gpu

    # After training, assume main.py saved test metrics in metrics.json.
    if [ -f "$OUTPUT_DIR/metrics.json" ]; then
        # Parse metrics.json to obtain MRR and Hits values.
        metrics=$(python -c "import json; f=json.load(open('$OUTPUT_DIR/metrics.json')); print(f\"{f['MRR']},{f['Hits@1']},{f['Hits@3']},{f['Hits@10']}\")")
        echo "$model,$metrics" >> $RESULTS_CSV
    else
        echo "Warning: No metrics.json found in $OUTPUT_DIR; skipping $model in results."
    fi
done

echo "------------------------------------------------"
echo "All runs complete. Results saved to $RESULTS_CSV."
