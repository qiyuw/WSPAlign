DATA_PATH=data # path to your data
mkdir $DATA_PATH

# pretrain
mkdir $DATA_PATH/pt_data
wget https://huggingface.co/datasets/qiyuw/wspalign_pt_data/blob/main/train-6langs.json -O "$DATA_PATH"/ft_data/train-6langs.json
wget https://huggingface.co/datasets/qiyuw/wspalign_pt_data/blob/main/train-6langs.json -O "$DATA_PATH"/ft_data/kftt_dev.json

# finetune
mkdir $DATA_PATH/ft_data
for LANG in kftt deen enfr roen
do
    wget https://huggingface.co/datasets/qiyuw/wspalign_ft_data/blob/main/"$LANG"_ft.json -O "$DATA_PATH"/ft_data/"$LANG"_ft.json
done

# few shot
mkdir $DATA_PATH/few_ft_data
for LANG in kftt deen enfr roen
do
    wget https://huggingface.co/datasets/qiyuw/wspalign_few_ft_data/blob/main/"$LANG"_few.json -O "$DATA_PATH"/few_ft_data/"$LANG"_few.json
done

# test and eval dataset
mkdir $DATA_PATH/test_data
for LANG in kftt deen enfr roen
do
    wget https://huggingface.co/datasets/qiyuw/wspalign_test_data/blob/main/"$LANG"_test.json -O "$DATA_PATH"/few_ft_data/"$LANG"_test.json
done