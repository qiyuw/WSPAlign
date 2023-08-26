# WSPAlign
WSPAlign: Word Alignment Pre-training via Large-Scale Weakly Supervised Span Prediction, published at ACL 2023 main conference.

This repository includes the source codes of paper [WSPAlign: Word Alignment Pre-training via Large-Scale Weakly Supervised Span Prediction](https://aclanthology.org/2023.acl-long.621/).
Part of the implementation is from [word_align](https://github.com/nttcslab-nlp/word_align).

## Model list
| Model List| Description|
|-------|-------|
|[qiyuw/WSPAlign-xlm-base](https://huggingface.co/qiyuw/WSPAlign-xlm-base) | Pretrained on xlm-roberta |
|[qiyuw/WSPAlign-mbert-base](https://huggingface.co/qiyuw/WSPAlign-mbert-base) | Pretrained on mBERT|
|[qiyuw/WSPAlign-ft-kftt](https://huggingface.co/qiyuw/WSPAlign-ft-kftt)| Finetuned with English-Japanese KFTT dataset|
|[qiyuw/WSPAlign-ft-deen](https://huggingface.co/qiyuw/WSPAlign-ft-deen)| Finetuned with German-English dataset|
[qiyuw/WSPAlign-ft-enfr](https://huggingface.co/qiyuw/WSPAlign-ft-enfr)| Finetuned with English-French dataset|
[qiyuw/WSPAlign-ft-roen](https://huggingface.co/qiyuw/WSPAlign-ft-roen)| Finetuned with Romanian-English dataset|

Use our model checkpoints with [huggingface](https://huggingface.co/)

## Data preparation
| Dataset list| Description|
|-------|-------|
|[qiyuw/wspalign_pt_data](https://huggingface.co/datasets/qiyuw/wspalign_pt_data) | Pre-training dataset|
|[qiyuw/wspalign_ft_data](https://huggingface.co/datasets/qiyuw/wspalign_ft_data) | Finetuning dataset|
|[qiyuw/wspalign_few_ft_data](https://huggingface.co/datasets/qiyuw/wspalign_few_ft_data) | Few-shot fintuning dataset|
|[qiyuw/wspalign_test_data](https://huggingface.co/datasets/qiyuw/wspalign_test_data)| Test dataset for evaluation|
|[qiyuw/qiyuw/wspalign_acl2023_eval](https://huggingface.co/datasets/qiyuw/wspalign_acl2023_eval)|Evaluation data used in the paper|

Construction of `Finetuning`, `Test` and `Evaluation` dataset can be found at [word_align](https://github.com/nttcslab-nlp/word_align).

Run `download_dataset.sh` to download all the above datasets.

## Pre-train and finetune
You can do pre-train, finetune and evaluate by running the following scripts.
### Pre-train
See [pretrain.sh](pretrain.sh) for details.
 
You can also use pre-traned model to directly do word alignment (zero-shot), see [zero-shot.sh](zero-shot.sh) for details.

### Finetune
See [finetune.sh](finetune.sh), [fewshot.sh](fewshot.sh) for details.

### Evaluate
Refer to [WSPAlign Inference](https://github.com/qiyuw/WSPAligner-inference) for details.

## License

This software is released under the `CC-BY-NC-SA-4.0 License`, see [LICENSE.txt](LICENSE.txt).
