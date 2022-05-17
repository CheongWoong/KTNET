import argparse
import random
import copy
import os
import json
from tqdm import tqdm

from collections import defaultdict
import numpy as np

from nltk.corpus import stopwords
from nltk import word_tokenize

from utils import Synonym_Helper, lemmatize, compute_f1

stopword = stopwords.words('english')

def parse_args():
	parser = argparse.ArgumentParser(description='get arguments.')
	parser.add_argument(
		'--model_name'
	)
	parser.add_argument(
		'--knowledge_bases',
		nargs='+',
		default=['conceptnet']
	)
	parser.add_argument(
		'--threshold', type=float, default=0.1
	)
	parser.add_argument(
		'--k', type=int, default=5
	)
	
	args = parser.parse_args()
	return args

def random_replacement_validation(random_score, qid, q_idx, threshold):
	key = qid + '_' + str(q_idx)
	if key in random_score and random_score[key] < threshold:
		return True
	return False

def synonym_replacement(sample, synonym_helper, random_score, threshold, k):
	new_samples = []

	qid = sample['id']
	question = word_tokenize(sample['question'])
	context = word_tokenize(sample['context'])

	context_lemma = [lemmatize(c_word) for c_word in context]

	for q_idx, q_word in enumerate(question):
		q_word_lemma = lemmatize(q_word)
		if q_word in stopword or q_word_lemma in stopword:
			continue
		if q_word not in context + context_lemma and q_word_lemma not in context + context_lemma:
			continue
		if not random_replacement_validation(random_score, qid, q_idx, threshold):
			continue

		synonyms = synonym_helper.extract_synonyms(q_word)

		for synonym in synonyms:
			synonym_lemma = lemmatize(synonym)
			if synonym in stopword or synonym_lemma in stopword:
				continue
			if synonym in context + context_lemma or synonym_lemma in context + context_lemma:
				continue
			new_question = ' '.join(question[:q_idx] + [synonym] + question[q_idx+1:])
			new_sample = copy.deepcopy(sample)
			new_sample['question'] = new_question
			new_sample['id'] = '_'.join([qid, str(q_idx), q_word, synonym])
			new_samples.append(new_sample)
	
	return new_samples

def main():
	dir_path = os.path.dirname(os.path.realpath(__file__))
	dir_path = "/".join(dir_path.rstrip("/").split("/")[:-3])

	data_path = os.path.join('..', 'data', 'SQuAD')
	model_path = os.path.join('output_'+args.model_name)

	synonym_helper = Synonym_Helper(args.knowledge_bases)

	random.seed(0)

	f = json.load(open(os.path.join(data_path, 'validation.json'), 'r'))
	new_f = {}

	version = f['version']
	data = f['data']
	new_f['version'] = version
	new_f['data'] = []

	# process the random-replacement scores	
	original_pred = json.load(open(os.path.join(model_path, 'predictions.json'), 'r'))
	random_pred = json.load(open(os.path.join(model_path, 'random-replacement', 'predictions.json'), 'r'))
	random_score = defaultdict(list)
	for key in random_pred:
		qid, q_idx = key.split('_')[:2]
		score = compute_f1(random_pred[key], original_pred[qid])
		random_score[qid + '_' + q_idx].append(score)
	for key in random_score:
		random_score[key] = np.mean(random_score[key])
	#######################################
	
	for sample in tqdm(data, desc='Processing synonym replacement (validation)'):
		new_samples = synonym_replacement(sample, synonym_helper, random_score, args.threshold, args.k)
		for new_sample in new_samples:
			new_f['data'].append(new_sample)

	print(len(new_f['data']), 'samples generated from', len(f['data']), 'original samples (threshold = '+str(args.threshold)+' / k = '+str(args.k)+')')

	out_dir = os.path.join(data_path, 'synonym-replacement', args.model_name)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	with open(os.path.join(out_dir, 'validation.json'), 'w') as fout:
		json.dump(new_f, fout)


if __name__ == '__main__':
	args = parse_args()
	main()
	print('Done')
	
