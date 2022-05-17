import csv
import os
from tqdm import tqdm

def main():
	dir_path = os.path.dirname(os.path.realpath(__file__))
	dir_path = '/'.join(dir_path.rstrip('/').split('/'))

	with open(os.path.join(dir_path, 'conceptnet-assertions-5.7.0.csv'), newline='') as csvfile:
		fout = open(os.path.join(dir_path, 'synonym-antonym.txt'), 'w')

		spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
		for raw in tqdm(spamreader, desc='Extracting synonyms/antonyms'):
			rel, subj, obj = raw[1:4]
        
	        # Filter out out-of-interest relations
			if 'Antonym' in rel:
				rel = 'Antonym'
			elif 'Synonym' in rel:
				rel = 'Synonym'
			else:
				continue
            
	        # Filter out non-English
			if subj.split('/')[2] != 'en' or obj.split('/')[2] != 'en':
				continue
            
			subj = subj.split('/')[3]
			obj = obj.split('/')[3]
        
	        #print(subj, rel, obj)
			fout.write('\t'.join([subj, rel, obj]) + '\n')

		fout.close()


if __name__ == '__main__':
	main()

