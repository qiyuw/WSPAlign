
NPREDFILE=$1
LANG=$2
TOKENIZER=$3 # [BERT, ROBERTA]

TESTFILE_DIR=/data/10/WSPAlign/wspalign_acl2023_eval # path to your test files

case "$LANG" in
   "kftt")  TESTFILE=/data/10/qiyuw/sentence_pairs/eval/kftt_test.json
            TEXTFILE=$TESTFILE_DIR/kftt/kftt_devtest.txt
            MOSESFILE=$TESTFILE_DIR/kftt/kftt_devtest.moses
            OPTION="-l"
   ;;
   "deen")  TESTFILE=/data/10/qiyuw/sentence_pairs/eval/deen_test.json
            TEXTFILE=$TESTFILE_DIR/deen/deen_test.text
            MOSESFILE=$TESTFILE_DIR/deen/deen_test.moses
            OPTION=""
   ;;
   "enfr")  TESTFILE=/data/10/qiyuw/sentence_pairs/eval/enfr_test.json
            TEXTFILE=$TESTFILE_DIR/enfr/enfr_test.text
            MOSESFILE=$TESTFILE_DIR/enfr/enfr_test.moses
            OPTION=""
   ;;
   "roen")  TESTFILE=/data/10/qiyuw/sentence_pairs/eval/roen_test.json
            TEXTFILE=$TESTFILE_DIR/roen/roen_test.text
            MOSESFILE=$TESTFILE_DIR/roen/roen_test.moses
            OPTION=""
   ;;
esac

echo $TESTFILE
echo $NPREDFILE
echo $TEXTFILE
echo $MOSESFILE

RANDOM_STR=$RANDOM

python convert_start_end.py -q $TESTFILE -n $NPREDFILE -m 160 -t $TOKENIZER > $RANDOM_STR-charindex_nbest_predictions.json

python get_alignment.py $OPTION -a $TEXTFILE -n $RANDOM_STR-charindex_nbest_predictions.json -m 160 

python get_alignment.py -b $OPTION -a $TEXTFILE -n $RANDOM_STR-charindex_nbest_predictions.json -m 160 > $RANDOM_STR-"$LANG"_test.moses.bidi_th

python aer.py $MOSESFILE $RANDOM_STR-"$LANG"_test.moses.bidi_th

rm $RANDOM_STR*