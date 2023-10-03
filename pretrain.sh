#!/bin/sh

PROJECT_DIR=wspalign
EXPERIMENT_DIR=/data/local/qiyuw/WSPAlign/experiments
OUTPUT_DIR=$EXPERIMENT_DIR/pretraining

DATA_DIR=/data/10/WSPAlign
TRAIN_FILE=$DATA_DIR/train-6langs.json
DEV_FILE=$DATA_DIR/kftt_dev.json

MODEL_TYPE=xlm-roberta # or mbert
MODEL_NAME=xlm-roberta-base
# MODEL_NAME=bert-base-multilingual-cased

date
hostname
echo $EXPERIMENT_DIR

echo ""
echo "### pretraining ###"
mkdir -p $OUTPUT_DIR
python $PROJECT_DIR/run_spanpred.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --do_train \
    --do_eval \
    --eval_all_checkpoints \
    --train_file $TRAIN_FILE \
    --predict_file $DEV_FILE \
    --learning_rate 1e-6 \
    --per_gpu_train_batch_size 64 \
    --num_train_epochs 10 \
    --max_seq_length 384 \
    --max_query_length 158 \
    --max_answer_length 158 \
    --doc_stride 64 \
    --n_best_size 10 \
    --data_dir $OUTPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --save_steps 100000 \
    --thread 4 2>&1 \
    --evaluate_during_training \
    --per_gpu_eval_batch_size 96 \
    --logging_steps 2000 \
    --version_2_with_negative \
    --warmup_step 2000 \
| tee $EXPERIMENT_DIR/pretraining.log
