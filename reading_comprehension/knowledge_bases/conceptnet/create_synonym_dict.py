from collections import defaultdict
import pickle as pkl
import os
from tqdm import tqdm

def main():
	dir_path = os.path.dirname(os.path.realpath(__file__))
	dir_path = '/'.join(dir_path.rstrip('/').split('/'))

	fin = open(os.path.join(dir_path, 'synonym-antonym-single-word.txt'), 'r').readlines()
	fout = open(os.path.join(dir_path, 'synonym-dict.pkl'), 'wb')

	synonym_dict = defaultdict(set)

	for line in tqdm(fin, desc='Creating synonym dictionary'):
		subj, rel, obj = line.strip().split('\t')

		if subj == obj:
			continue
		elif rel != 'Synonym':
			continue

		synonym_dict[subj].add(obj)
		synonym_dict[obj].add(subj)

	pkl.dump(synonym_dict, fout)

	fout.close()


if __name__ == '__main__':
	main()
