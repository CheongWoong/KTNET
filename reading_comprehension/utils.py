from collections import defaultdict
import collections
import pickle as pkl
from itertools import product

import string
import re
import os
import logging

import numpy as np
import torch

from scipy.stats import entropy
from scipy.spatial import distance

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag

lemmatizer = WordNetLemmatizer()
stopword = stopwords.words('english')

logger = logging.getLogger(__name__)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = "/".join(dir_path.rstrip("/").split("/")[:-1])

################################################################################
# POS tagger/lemmatizer
def get_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize(word):
    return lemmatizer.lemmatize(word, get_pos(word))

################################################################################

################################################################################
class Synonym_Helper():
    def __init__(self, knowledge_bases=['conceptnet']):
        self.synonym_dict = defaultdict(set)
        for knowledge_base in knowledge_bases:
            filepath = os.path.join('knowledge_bases', knowledge_base, 'synonym-dict.pkl')
            temp = pkl.load(open(filepath, 'rb'))
            for key in temp:
                self.synonym_dict[key] = self.synonym_dict[key].union(temp[key])

    def extract_synonyms(self, word):
        synonyms = set()

        if word in self.synonym_dict:
            synonyms = synonyms.union(self.synonym_dict[word])

        word_lemma = lemmatize(word)
        if word_lemma in self.synonym_dict:
            synonyms = synonyms.union(self.synonym_dict[word_lemma])

        return synonyms

################################################################################


################################################################################
# score functions
def normalize_answer(s):
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

################################################################################


################################################################################
# preprocess/ postprocess for MRC
def preprocess(tokenizer, question, context):
    tokenized = tokenizer(question,
                          context,
                          truncation='only_second',
                          max_length=384,
                          return_overflowing_tokens=True,
                          return_offsets_mapping=True,
                          padding='max_length')

    # skip overflow
    if len(tokenized['input_ids']) > 1:
        return None, None

    for i in range(len(tokenized["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized.sequence_ids(i)

        pad_on_right = tokenizer.padding_side == "right"
        context_index = 1 if pad_on_right else 0

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]

    encodings = {k: torch.tensor(v).to(device) for k, v in tokenized.items() if
                 k in ['input_ids', 'token_type_ids', 'attention_mask']}
    info = {k: v for k, v in tokenized.items() if k not in ['input_ids', 'token_type_ids', 'attention_mask']}

    return encodings, info


def postprocess(output, info, context, return_index=False):
    n_best_size = 20
    max_answer_length = 30
    null_score_diff_threshold = 0

    start_logits = output['start_logits'][0].detach().cpu().numpy()
    end_logits = output['end_logits'][0].detach().cpu().numpy()
    offset_mapping = info['offset_mapping'][0]

    prelim_predictions = []

    null_score = start_logits[0] + end_logits[0]
    null_prediction = {
        'offsets': (0, 0),
        'score': null_score,
        'start_logit': start_logits[0],
        'end_logit': end_logits[0]
    }

    start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
    end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
    for start_index in start_indexes:
        for end_index in end_indexes:
            if (start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
            ):
                continue
            if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                continue
            prelim_predictions.append(
                {
                    'offsets': (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                    'score': start_logits[start_index] + end_logits[end_index],
                    'start_logit': start_logits[start_index],
                    'end_logit': end_logits[end_index],
                }
            )
    prelim_predictions.append(null_prediction)

    predictions = sorted(prelim_predictions, key=lambda x: x['score'], reverse=True)[:n_best_size]

    if not any(p['offsets'] == (0, 0) for p in predictions):
        predictions.append(null_prediction)

    for pred in predictions:
        offsets = pred.pop('offsets')
        pred['text'] = context[offsets[0]: offsets[1]]

    if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]['text'] == ''):
        predictions.insert(0, {'text': 'empty', 'start_logit': 0.0, 'end_logit': 0.0, 'score': 0.0})

    predictions.append({'text': 'empty', 'start_logit': 0.0, 'end_logit': 0.0, 'score': 0.0})

    i = 0
    while predictions[i]['text'] == '':
        i += 1
    best_non_null_pred = predictions[i]

    score_diff = float(null_score - best_non_null_pred['start_logit'] - best_non_null_pred['end_logit'])
    if score_diff > null_score_diff_threshold:
        prediction = ''
    else:
        prediction = best_non_null_pred['text']

    if return_index == True:
        return start_logits, end_logits, start_index, end_index, output['attentions'], prediction, score_diff

    else:
        return start_logits, end_logits, output['attentions'], prediction, score_diff

################################################################################


################################################################################
# find target index of original word and synonym word at the input_ids
def find_word_index_in_input_ids(word_pairs, encodings, tokenizer):
	target_position_mask = torch.zeros((384, 384), dtype=torch.float32).to(device)

	for word_pair in word_pairs:
		word, synonym = word_pair
		word_lemma, synonym_lemma = lemmatize(word), lemmatize(synonym)
		words = [word, word_lemma]
		synonyms = [synonym, synonym_lemma]

		original_word_index = torch.zeros_like(encodings['input_ids'], dtype=torch.float32)
		synonym_word_index = torch.zeros_like(encodings['input_ids'], dtype=torch.float32)

		is_question = True
		for index, encoding in enumerate(encodings['input_ids'][0]):
			if encoding == tokenizer.pad_token_id:
				break
			if encoding == tokenizer.sep_token_id:
				is_question = False

			input_word = tokenizer.decode(encoding).strip()
			input_word_lemma = lemmatize(input_word)

			if not is_question:
				for w in words:
					if w == input_word or w == input_word_lemma:
						original_word_index[0][index] = 1
			else:
				for w in synonyms:
					if w == input_word or w == input_word_lemma:
						synonym_word_index[0][index] = 1

		original_synonym_index = torch.matmul(original_word_index.T, synonym_word_index)
		synonym_original_index = torch.matmul(synonym_word_index.T, original_word_index)
		target_position_mask += (original_synonym_index + synonym_original_index)

	target_position_mask = target_position_mask > 0

	if target_position_mask.sum() < 1:
		return None
	else:
		return target_position_mask

def find_synonym_pairs(question, context, synonym_helper, tokenizer):
	synonym_pairs = set()

	context = word_tokenize(context)
	question = word_tokenize(question)

	context_lemma = [lemmatize(c_word) for c_word in context]

	for q_idx, q_word in enumerate(question):
		q_word_lemma = lemmatize(q_word)
		if q_word in stopword or q_word_lemma in stopword:
			continue
		q_token = tokenizer.tokenize(q_word)
		if len(q_token) > 1:
			continue

		synonyms = synonym_helper.extract_synonyms(q_word)

		for synonym in synonyms:
			synonym_lemma = lemmatize(synonym)
			if synonym in stopword or synonym_lemma in stopword:
				continue
			synonym_token = tokenizer.tokenize(synonym)
			if len(synonym_token) > 1:
				continue
			if synonym in context + context_lemma or synonym_lemma in context + context_lemma:
				synonym_pairs.add((synonym, q_word))

	return synonym_pairs

# get attention map for AttentionOverride
def get_attention_map(model, output, encodings, qid, tokenizer):
	attention_override = output[2]  # attentions weight of bert model after the attention softmax

	batch_size = 1
	num_layers = len(output[2])  # num of layers
	num_heads = model.config.num_attention_heads  # num of heads
	seq_len = len(encodings['input_ids'][0])  # sequence length

	assert len(attention_override) == num_layers
	assert attention_override[0].shape == (batch_size, num_heads, seq_len, seq_len)

	layer_attn_data = []
	head_attn_data = []
	word_attn_data = []

	for layer in range(num_layers):
		layer_attention_weight = attention_override[layer]
		layer_attention_mask = torch.ones_like(layer_attention_weight, dtype=torch.bool)
		attn_per_layer_data = {
			'layer': layer,
			'attention': layer_attention_weight,
			'attention_mask': layer_attention_mask
		}
		layer_attn_data.append(attn_per_layer_data)

		for head in range(num_heads):
			head_attention_weight = attention_override[layer][0][head]
			head_attention_mask = torch.zeros_like(layer_attention_weight, dtype=torch.bool)
			head_attention_mask[0][head] = 1  # Set mask to 1 for single head only

			normalized_attention_weight = torch.sum(head_attention_weight, dim=0) / len(head_attention_weight)

			attn_per_head_data = {
				'layer': layer,
				'head': head,
				'attention': head_attention_weight,
				'attention_mask': head_attention_mask
			}
			head_attn_data.append(attn_per_head_data)

			original_word_index_list, synonym_word_index_list = find_word_index_in_input_ids(qid, encodings, tokenizer)

			word_attention_mask = torch.zeros_like(layer_attention_weight, dtype=torch.bool)
			word_attention_weight = attention_override[layer][0][head]

			items = [original_word_index_list, synonym_word_index_list]
			for i in list(product(*items)):
				word_attention_mask[0][head][i] = 1

			items_inverse = [synonym_word_index_list, original_word_index_list]
			for i in list(product(*items_inverse)):
				word_attention_mask[0][head][i] = 1

			attn_per_word_data = {
				'layer': layer,
				'head': head,
				'word': qid.split('_')[2],
				'synonym': qid.split('_')[3],
				'attention': word_attention_weight,
				'attention_mask': word_attention_mask
			}
			word_attn_data.append(attn_per_word_data)

	assert len(layer_attn_data) == 12
	assert len(head_attn_data) == 12 * 12
	assert len(word_attn_data) == 12 * 12

	return layer_attn_data, head_attn_data, word_attn_data

################################################################################


################################################################################
# compute effect
def dist(p, q, option='jsd'):
    p = torch.nn.Softmax(dim=0)(torch.from_numpy(p)).detach().cpu() + 1e-6
    q = torch.nn.Softmax(dim=0)(torch.from_numpy(q)).detach().cpu() + 1e-6

    # kl divergence
    if option == 'kld':
        return entropy(p, q)
    # jenson-shannon divergence
    elif option == 'jsd':
        return distance.jensenshannon(p, q)
    # total variation norm
    elif option == 'var':
        return float((torch.sum(torch.abs(p - q)) / 2))

def entropy(p):
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)

def print_2d_tensor(tensor):
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))
