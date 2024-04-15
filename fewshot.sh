#!/bin/sh

DATE=$(date '+%Y-%m-%d')
PROJECT_DIR=wspalign
EXPERIMENT_DIR=/data/local/qiyuw/WSPAlign/experiments-fewshot-$DATE
OUTPUT_DIR=$EXPERIMENT_DIR/fewshot

DATA_DIR=/data/10/WSPAlign # path to your test files
TRAIN_FILE=$DATA_DIR/few_ft_data
TEST_FILE=$DATA_DIR/test_data

MODEL_TYPE=xlm-roberta # or mbert
MODEL_NAME=qiyuw/WSPAlign-xlm-base # path to your model

date
hostname
echo $EXPERIMENT_DIR
rm -rf $EXPERIMENT_DIR/*

echo ""
echo "### few-shot finetuning ###"
mkdir -p $OUTPUT_DIR

simple_hypersearch \
"
mkdir -p $OUTPUT_DIR/{lang}-{lr}-{bz}
python $PROJECT_DIR/run_spanpred.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --do_train \
    --do_eval \
    --eval_all_checkpoints \
    --train_file $TRAIN_FILE/{lang}_few.json \
    --predict_file $TEST_FILE/{lang}_test.json \
    --learning_rate {lr} \
    --per_gpu_train_batch_size {bz} \
    --num_train_epochs 250 \
    --max_seq_length 384 \
    --max_query_length 158 \
    --max_answer_length 158 \
    --doc_stride 64 \
    --n_best_size 10 \
    --data_dir $OUTPUT_DIR/{lang}-{lr}-{bz} \
    --output_dir $OUTPUT_DIR/{lang}-{lr}-{bz} \
    --overwrite_output_dir \
    --save_steps 10000000 \
    --thread 4 2>&1 \
    --per_gpu_eval_batch_size 12 \
    --logging_steps 500 \
    --version_2_with_negative \
| tee $EXPERIMENT_DIR/fewshot-{lang}-{lr}-{bz}.log
rm $OUTPUT_DIR/{lang}-{lr}-{bz}/checkpoint*
rm $OUTPUT_DIR/{lang}-{lr}-{bz}/cached*
" \
-p lang kftt deen enfr roen -p lr 1e-6 3e-6 1e-5 3e-5 -p bz 5 8 12 | simple_gpu_scheduler --gpus 0,1,2,3,4,5,6,7