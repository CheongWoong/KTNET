import os

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = '/'.join(dir_path.rstrip('/').split('/'))

    os.system('python3 ' + dir_path + '/extract_en_synonym_antonym.py')
    os.system('python3 ' + dir_path + '/filter_multi_words.py')
    os.system('python3 ' + dir_path + '/create_synonym_dict.py')


if __name__ == '__main__':
    main()
    print('Done')
