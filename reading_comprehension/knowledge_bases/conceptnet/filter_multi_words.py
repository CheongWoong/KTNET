import os
from tqdm import tqdm

def main():
	dir_path = os.path.dirname(os.path.realpath(__file__))
	dir_path = '/'.join(dir_path.rstrip('/').split('/'))

	fin = open(os.path.join(dir_path, 'synonym-antonym.txt'), 'r').readlines()
	fout = open(os.path.join(dir_path, 'synonym-antonym-single-word.txt'), 'w')

	for line in tqdm(fin, desc='Filtering out multi-words'):
		subj, rel, obj = line.strip().split('\t')
		if '_' in subj or '_' in obj:
			continue
		else:
			fout.write(line.strip() + '\n')

	fout.close()


if __name__ == '__main__':
	main()
