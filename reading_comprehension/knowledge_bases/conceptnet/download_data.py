import os

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = '/'.join(dir_path.rstrip('/').split('/'))
    os.system('wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz -P ' + dir_path)
    os.system('gzip -d ' + dir_path + '/conceptnet-assertions-5.7.0.csv.gz')

if __name__ == '__main__':
	main()
