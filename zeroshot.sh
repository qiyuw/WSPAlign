#!/bin/sh

DATE=$(date '+%Y-%m-%d')
PROJECT_DIR=wspalign
EXPERIMENT_DIR=/data/local/qiyuw/WSPAlign/experiments-zeroshot-$DATE
OUTPUT_DIR=$EXPERIMENT_DIR/zeroshot

DATA_DIR=/data/10/WSPAlign # path to your test files
TEST_FILE=$DATA_DIR/test_data

MODEL_TYPE=xlm-roberta # or mbert
MODEL_NAME=qiyuw/WSPAlign-xlm-base # path to your model

date
hostname
echo $EXPERIMENT_DIR
rm -rf $EXPERIMENT_DIR/*

echo ""
echo "### zeroshot ###"
mkdir -p $OUTPUT_DIR

simple_hypersearch \
"
mkdir -p $OUTPUT_DIR/{lang}
python $PROJECT_DIR/run_spanpred.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --do_eval \
    --eval_all_checkpoints \
    --predict_file $TEST_FILE/{lang}_test.json \
    --max_seq_length 384 \
    --max_query_length 158 \
    --max_answer_length 158 \
    --doc_stride 64 \
    --n_best_size 10 \
    --data_dir $OUTPUT_DIR/{lang} \
    --output_dir $OUTPUT_DIR/{lang} \
    --overwrite_output_dir \
    --thread 4 2>&1 \
    --per_gpu_eval_batch_size 48 \
    --version_2_with_negative \
| tee $EXPERIMENT_DIR/zeroshot-{lang}-zeroshot.log" \
-p lang kftt deen enfr roen | simple_gpu_scheduler --gpus 0,1,2,3