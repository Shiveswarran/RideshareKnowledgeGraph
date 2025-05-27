#!/usr/bin/env bash
#
# run_ablation.sh
# Runs a suite of ablations for the Multi-Head Density Motif Attention KGE.

# Path to your triplets CSV
DATA_PATH=/scratch/umni5/a/shives/Research_projects/Knowledge_graphs/RideshareKnowledgeGraph/Data/knowledge_graph_triplets.csv

# Common hyperparameters
EMB_DIM=200
LR=0.001
MARGIN=1.0
EPOCHS=200
BATCH=128
PATIENCE=20
REG_TYPE=N2
REG_WEIGHT=1e-5
NEG_SAMPLING=density
TEMP=0.5
BASE_OUT=./ablation_results

mkdir -p $BASE_OUT

# 1) Full model (baseline)
python main.py \
  --data_path $DATA_PATH \
  --model_name UrbanRidehailKG \
  --embedding_dim $EMB_DIM \
  --lr $LR --margin $MARGIN \
  --epochs $EPOCHS --batch_size $BATCH --patience $PATIENCE \
  --reg_type $REG_TYPE --reg_weight $REG_WEIGHT \
  --neg_sampling $NEG_SAMPLING --temperature $TEMP \
  --output_dir $BASE_OUT/baseline

# 2) No density modulation
python main.py \
  --data_path $DATA_PATH \
  --model_name UrbanRidehailKG \
  --embedding_dim $EMB_DIM \
  --lr $LR --margin $MARGIN \
  --epochs $EPOCHS --batch_size $BATCH --patience $PATIENCE \
  --no_density \
  --reg_type $REG_TYPE --reg_weight $REG_WEIGHT \
  --neg_sampling $NEG_SAMPLING --temperature $TEMP \
  --output_dir $BASE_OUT/no_density

# 3) No motif consistency loss
python main.py \
  --data_path $DATA_PATH \
  --model_name UrbanRidehailKG \
  --embedding_dim $EMB_DIM \
  --lr $LR --margin $MARGIN \
  --epochs $EPOCHS --batch_size $BATCH --patience $PATIENCE \
  --motif_lambda 0 \
  --reg_type $REG_TYPE --reg_weight $REG_WEIGHT \
  --neg_sampling $NEG_SAMPLING --temperature $TEMP \
  --output_dir $BASE_OUT/no_motif_loss

# 4) Single head only
python main.py \
  --data_path $DATA_PATH \
  --model_name UrbanRidehailKG \
  --embedding_dim $EMB_DIM \
  --heads 1 \
  --lr $LR --margin $MARGIN \
  --epochs $EPOCHS --batch_size $BATCH --patience $PATIENCE \
  --reg_type $REG_TYPE --reg_weight $REG_WEIGHT \
  --neg_sampling $NEG_SAMPLING --temperature $TEMP \
  --output_dir $BASE_OUT/single_head

# 5) No motif attention (vanilla TransE diff only)
python main.py \
  --data_path $DATA_PATH \
  --model_name UrbanRidehailKG \
  --embedding_dim $EMB_DIM \
  --no_motif_attention \
  --lr $LR --margin $MARGIN \
  --epochs $EPOCHS --batch_size $BATCH --patience $PATIENCE \
  --reg_type $REG_TYPE --reg_weight $REG_WEIGHT \
  --neg_sampling $NEG_SAMPLING --temperature $TEMP \
  --output_dir $BASE_OUT/no_attention

# 6) No self-adversarial sampling
python main.py \
  --data_path $DATA_PATH \
  --model_name UrbanRidehailKG \
  --embedding_dim $EMB_DIM \
  --neg_sampling uniform \
  --lr $LR --margin $MARGIN \
  --epochs $EPOCHS --batch_size $BATCH --patience $PATIENCE \
  --reg_type $REG_TYPE --reg_weight $REG_WEIGHT \
  --output_dir $BASE_OUT/no_adv_sampling

# 7) Vary number of motifs (M = 4,8,16)
for M in 4 8 16; do
  python main.py \
    --data_path $DATA_PATH \
    --model_name UrbanRidehailKG \
    --embedding_dim $EMB_DIM \
    --motifs $M \
    --lr $LR --margin $MARGIN \
    --epochs $EPOCHS --batch_size $BATCH --patience $PATIENCE \
    --reg_type $REG_TYPE --reg_weight $REG_WEIGHT \
    --neg_sampling $NEG_SAMPLING --temperature $TEMP \
    --output_dir $BASE_OUT/motifs_${M}
done

# 8) Vary number of heads (H = 1,2,4,8)
for H in 1 2 4 8; do
  python main.py \
    --data_path $DATA_PATH \
    --model_name UrbanRidehailKG \
    --embedding_dim $EMB_DIM \
    --heads $H \
    --lr $LR --margin $MARGIN \
    --epochs $EPOCHS --batch_size $BATCH --patience $PATIENCE \
    --reg_type $REG_TYPE --reg_weight $REG_WEIGHT \
    --neg_sampling $NEG_SAMPLING --temperature $TEMP \
    --output_dir $BASE_OUT/heads_${H}
done

# 9) No projection layer
python main.py \
  --data_path $DATA_PATH \
  --model_name UrbanRidehailKG \
  --embedding_dim $EMB_DIM \
  --no_projection \
  --lr $LR --margin $MARGIN \
  --epochs $EPOCHS --batch_size $BATCH --patience $PATIENCE \
  --reg_type $REG_TYPE --reg_weight $REG_WEIGHT \
  --neg_sampling $NEG_SAMPLING --temperature $TEMP \
  --output_dir $BASE_OUT/no_projection

# 10) No relation in attention
python main.py \
  --data_path $DATA_PATH \
  --model_name UrbanRidehailKG \
  --embedding_dim $EMB_DIM \
  --no_rel_attn \
  --lr $LR --margin $MARGIN \
  --epochs $EPOCHS --batch_size $BATCH --patience $PATIENCE \
  --reg_type $REG_TYPE --reg_weight $REG_WEIGHT \
  --neg_sampling $NEG_SAMPLING --temperature $TEMP \
  --output_dir $BASE_OUT/no_rel_attn

# 11) Fixed (untrainable) prototypes
python main.py \
  --data_path $DATA_PATH \
  --model_name UrbanRidehailKG \
  --embedding_dim $EMB_DIM \
  --fix_prototypes \
  --lr $LR --margin $MARGIN \
  --epochs $EPOCHS --batch_size $BATCH --patience $PATIENCE \
  --reg_type $REG_TYPE --reg_weight $REG_WEIGHT \
  --neg_sampling $NEG_SAMPLING --temperature $TEMP \
  --output_dir $BASE_OUT/fix_prototypes
