import os
import torch
import argparse
import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    pipeline
)
from langdetect import detect
from spacy.lang.en import English
from spacy.lang.ja import Japanese
from spacy.lang.fr import French
from spacy.lang.de import German
from spacy.lang.zh import Chinese
from spacy.lang.ro import Romanian

from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

context_sep =' \u00b6 ' # use ' ¶ ' (pilcrow sign) for context separator
language_tokenizer_map = {
        'en': English,
        'ja': Japanese,
        'fr': French,
        'de': German,
        'zh': Chinese,
        'ro': Romanian,
    } # map of language to tokenizer


def word_alignments_data_collate_fn(batch):
    src_questions, src_contexts, tgt_questions, tgt_contexts = zip(*batch)
    # flatten for question-answering pipeline
    src_example_lengths = [len(qs) for qs in src_questions]
    tgt_example_lengths = [len(qs) for qs in tgt_questions]
    src_questions = [q for qs in src_questions for q in qs]
    src_contexts = [c for cs in src_contexts for c in cs]
    tgt_questions = [q for qs in tgt_questions for q in qs]
    tgt_contexts = [c for cs in tgt_contexts for c in cs]
    return src_questions, tgt_questions, src_contexts, tgt_contexts, src_example_lengths, tgt_example_lengths

class WordAlignDataset(Dataset):
    def __init__(self, src_examples, tgt_examples, src_texts, tgt_texts):
        assert len(src_examples) == len(tgt_examples) == len(src_texts) == len(tgt_texts),\
            "src_examples, tgt_examples, src_texts, tgt_texts must have the same length"
        self.src_examples = src_examples
        self.tgt_examples = tgt_examples
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts

    def __len__(self):
        return len(self.src_examples)

    def __getitem__(self, idx):
        src_example = self.src_examples[idx]
        tgt_example = self.tgt_examples[idx]
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # flatten
        src_questions = []
        src_contexts = []
        tgt_questions = []
        tgt_contexts = []
        for example in src_example:
            src_questions.append(example)
            src_contexts.append(tgt_text)
        for example in tgt_example:
            tgt_questions.append(example)
            tgt_contexts.append(src_text)
        return src_questions, src_contexts, tgt_questions, tgt_contexts
    
def post_unflatten(batch_outputs):
    predictions, example_lengths = batch_outputs
    unflattened_predictions = []
    s = 0
    for l in example_lengths:
        unflattened_predictions.append(predictions[s:s+l])
        s += l
    return unflattened_predictions

def get_tokenizer(src_lang, tgt_lang):
    if src_lang not in language_tokenizer_map:
        raise ValueError("Language {} not supported by SpaCy tokenizer.".format(args.src_tokenizer_lang))
    if tgt_lang not in language_tokenizer_map:
        raise ValueError("Language {} not supported by SpaCy tokenizer.".format(args.tgt_tokenizer_lang))
    # special case for Chinese
    if src_lang in ['zh-cn', 'zh-tw']:
        src_lang = 'zh'
    if tgt_lang in ['zh-cn', 'zh-tw']:
        tgt_lang = 'zh'
    if src_lang not in language_tokenizer_map:
        src_lang = 'en'
    if tgt_lang not in language_tokenizer_map:
        tgt_lang = 'en'
    src_tokenizer_to_load = language_tokenizer_map[src_lang]()
    tgt_tokenizer_to_load = language_tokenizer_map[tgt_lang]()
    src_tokenizer = src_tokenizer_to_load.tokenizer
    tgt_tokenizer = tgt_tokenizer_to_load.tokenizer
    return src_tokenizer, tgt_tokenizer

def make_word_alignments_data(sent, tokenizer, context_sep=' \u00b6 '):
    words = tokenizer(sent)
    examples = []
    for word in words:
        example = sent[:word.idx] + context_sep + sent[word.idx:word.idx+len(word.text)] + context_sep + sent[word.idx+len(word.text):]
        examples.append(example)
    return examples, words

def align(pipe, src_examples, tgt_texts, batch_size):
    # flatten
    questions = []
    contexts = []
    for src_example, tgt_text in zip(src_examples, tgt_texts):
        questions.extend(src_example)
        contexts.extend([tgt_text] * len(src_example))
    preds = pipe(question=questions, context=contexts, batch_size=batch_size)

    # unflatten
    unflattened_preds = []
    s = 0
    for i in range(len(src_examples)):
        l = len(src_examples[i])
        unflattened_preds.append(preds[s:s+l])
        s += l
    return unflattened_preds

def find_matching_word(src_pred, tgt_words):
    start = src_pred['start']
    end = src_pred['end']
    # filter out tgt_words that are not in the predicted span
    predicted_word_idx = []
    for i, tgt_word in enumerate(tgt_words):
        if tgt_word.idx >= start and tgt_word.idx+len(tgt_word.text) <= end:
            predicted_word_idx.append(i)
    return predicted_word_idx

def bidirectional_align(inputs, args):
    src_predictions, tgt_predictions, src_words, tgt_words = inputs
    src_to_tgt = {}
    for i, src_pred in enumerate(src_predictions):
        predicted_word_idx = find_matching_word(src_pred, tgt_words)
        for ith_tgt_word in predicted_word_idx:
            word_pair = "{}-{}".format(i, ith_tgt_word)
            src_to_tgt[word_pair] = [src_pred['score'] / len(predicted_word_idx), 1]
            # divide by the number of predicted word idx to prevent too long spans from dominating,
            # one for counting if we have two predictions for the same word-pair.
    for i, tgt_pred in enumerate(tgt_predictions):
        predicted_word_idx = find_matching_word(tgt_pred, src_words)
        for ith_src_word in predicted_word_idx:
            word_pair = "{}-{}".format(ith_src_word, i)
            if word_pair in src_to_tgt:
                src_to_tgt[word_pair][0] += tgt_pred['score'] / len(predicted_word_idx)
                src_to_tgt[word_pair][1] += 1
            else:
                continue
    # filter out word-pairs with low scores and unidirectional predictions
    src_to_tgt = {k: v for k, v in src_to_tgt.items() if v[0] > args.threshold}
    if args.force_bidirectional:
        src_to_tgt = {k: v for k, v in src_to_tgt.items() if v[1] > 1}
    return src_to_tgt

def batch_align(pipe, src_text, tgt_text, src_tokenizer, tgt_tokenizer, args):
    assert type(tgt_text) == list and len(src_text) == len(tgt_text), "tgt_text must be a list of strings with the same length as src_text"
    batch_size = args.batch_size

    # parallelize pre-tokenization
    p = Pool(args.n_cpu)
    make_word_alignments_data_with_src_tokenizer = partial(make_word_alignments_data, tokenizer=src_tokenizer)
    make_word_alignments_data_with_tgt_tokenizer = partial(make_word_alignments_data, tokenizer=tgt_tokenizer)
    imap_res_src = p.imap(make_word_alignments_data_with_src_tokenizer, src_text, len(src_text)//args.n_cpu)
    imap_res_tgt = p.imap(make_word_alignments_data_with_tgt_tokenizer, tgt_text, len(tgt_text)//args.n_cpu)
    src_examples = []
    src_words = []
    tgt_examples = []
    tgt_words = []
    for r in tqdm(zip(imap_res_src, imap_res_tgt), total=len(src_text), desc="Tokenizing source and target text"):
        res_src, res_tgt = r
        src_examples.append(res_src[0])
        src_words.append(res_src[1])
        tgt_examples.append(res_tgt[0])
        tgt_words.append(res_tgt[1])

    dataset = WordAlignDataset(src_examples, tgt_examples, src_text, tgt_text)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=word_alignments_data_collate_fn)

    # use question-answering pipeline for prediction
    src_predictions = []
    src_example_lengths = []
    tgt_predictions = []
    tgt_example_lengths = []
    pipe_partial = partial(pipe, batch_size=batch_size, doc_stride=158, question_first=True, max_question_len=512, max_answer_len=32, max_seq_len=512) # for panicException, see https://github.com/huggingface/tokenizers/issues/944
    for i, (src_questions, tgt_questions, src_contexts, tgt_contexts, src_example_l, tgt_example_l)\
        in tqdm(enumerate(dataloader), total=len(dataset)//batch_size, desc="Question-answering pipeline forwarding"):
            src_predictions.append(pipe_partial(question=src_questions, context=src_contexts)) # sentence-pair-wise -> example-wise
            tgt_predictions.append(pipe_partial(question=tgt_questions, context=tgt_contexts))
            src_example_lengths.append(src_example_l)
            tgt_example_lengths.append(tgt_example_l)

    # unflatten
    p = Pool(args.n_cpu)
    imap_res_src = p.imap(post_unflatten, zip(src_predictions, src_example_lengths), len(src_text)//args.n_cpu)
    imap_res_tgt = p.imap(post_unflatten, zip(tgt_predictions, tgt_example_lengths), len(src_text)//args.n_cpu)
    src_predictions = []
    tgt_predictions = []
    for r in tqdm(zip(imap_res_src, imap_res_tgt), total=len(src_predictions), desc="Unflattening predictions"):
        res_src, res_tgt = r
        src_predictions.extend(res_src)
        tgt_predictions.extend(res_tgt)

    # parallelize postprocessing
    assert len(src_predictions) == len(tgt_predictions) == len(src_words) == len(tgt_words), "src_predictions, tgt_predictions, src_words, tgt_words must have the same length"
    partial_bidirectional_align = partial(bidirectional_align, args=args)
    i_res_align = p.imap(partial_bidirectional_align, zip(src_predictions, tgt_predictions, src_words, tgt_words), len(src_text)//args.n_cpu)
    src_to_tgt = []
    for r in tqdm(i_res_align, total=len(src_text), desc="Bidirectional alignment"):
        src_to_tgt.append(r)
    p.close()

    res = []
    for src_t, src_w, tgt_t, tgt_w, s2t in zip(src_text, src_words, tgt_text, tgt_words, src_to_tgt):
        res.append(AlignedPair(src_t, src_w, tgt_t, tgt_w, s2t))
    return res

def load_data(src_file, tgt_file):
    with open(src_file, 'r') as f:
        src_text = f.readlines()
    with open(tgt_file, 'r') as f:
        tgt_text = f.readlines()
    return src_text, tgt_text

class AlignedPair:
    def __init__(self, src_text, src_words, tgt_text, tgt_words, src_to_tgt):
        self.src_words = src_words
        self.tgt_words = tgt_words
        self.src_text = src_text.strip()
        self.tgt_text = tgt_text.strip()
        self.src_to_tgt = src_to_tgt
        self.src_to_tgt_score, self.tgt_to_src_score = self.align_score()
        self.aligned_score = (self.src_to_tgt_score + self.tgt_to_src_score) / 2

    def __str__(self):
        res = "Src: {}\nTgt: {}\n".format(self.src_text, self.tgt_text)
        for src_tgt in self.src_to_tgt:
            src_idx, tgt_idx = src_tgt.split('-')
            src_idx = int(src_idx)
            tgt_idx = int(tgt_idx)
            src_word = self.src_words[src_idx]
            tgt_word = self.tgt_words[tgt_idx]
            res += "{} ---- {} | Score: {}\n".format(src_word, tgt_word, self.src_to_tgt[src_tgt][0])
        return res
    
    def align_score(self):
        src_l = len(self.src_words)
        tgt_l = len(self.tgt_words)
        src_aligned_words = set()
        tgt_aligned_words = set()
        for src_tgt in self.src_to_tgt:
            src_idx, tgt_idx = src_tgt.split('-')
            src_idx = int(src_idx)
            tgt_idx = int(tgt_idx)
            src_aligned_words.add(src_idx)
            tgt_aligned_words.add(tgt_idx)
        return len(src_aligned_words) / src_l, len(tgt_aligned_words) / tgt_l

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--src_file", type=str, default=None, help="Source text to align.")
    parser.add_argument("--tgt_file", type=str, default=None, help="Target text to align.")
    parser.add_argument("--save_to_dir", type=str, default=None, required=True, help="Directory to save alignments.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of CPUs to use for tokenization.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for alignment score.")
    parser.add_argument("--force_bidirectional", action="store_true", help="Force bidirectional alignment.")
    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    src_texts, tgt_texts = load_data(args.src_file, args.tgt_file)

    # SpaCy tokenizer
    src_lang = args.src_file.split('/')[-1].split('.')[-1]
    tgt_lang = args.tgt_file.split('/')[-1].split('.')[-1]
    src_tokenizer, tgt_tokenizer = get_tokenizer(src_lang, tgt_lang)
    print("Source tokenizer: {}".format(src_tokenizer))
    print("Target tokenizer: {}".format(tgt_tokenizer))

    # use question-answering pipeline for prediction
    pipe = pipeline("question-answering", model=args.model_name_or_path, device=args.device)
    aligned_pairs = batch_align(pipe, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, args)

    # save alignments to file
    if not os.path.exists(args.save_to_dir):
        os.makedirs(args.save_to_dir)
    src_tag = args.src_file.split('/')[-3]
    tgt_tag = args.tgt_file.split('/')[-3]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_to_file = "{}_{}_{}_{}_{}.txt".format(src_tag, src_lang, tgt_tag, tgt_lang, timestamp)
    args.save_to_file = os.path.join(args.save_to_dir, save_to_file)
    if os.path.exists(args.save_to_file):
        raise ValueError("File {} already exists.".format(args.save_to_file))
    with open(args.save_to_file, 'w') as f:
        for aligned_pair in aligned_pairs:
            f.write("{}\t{}\t{}\t{}\t{}".format(aligned_pair.src_text, aligned_pair.tgt_text, aligned_pair.src_to_tgt_score, aligned_pair.tgt_to_src_score, aligned_pair.aligned_score))
            f.write('\n')
    print("Done.")