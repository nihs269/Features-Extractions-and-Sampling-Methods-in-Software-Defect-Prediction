import csv
import numpy as np
import gensim

from preprocessing import *
from dataset import DATASET
from sklearn.feature_extraction.text import TfidfVectorizer
from codeBert import codebert_features

Techniques = {0: 'Original',  #
              1: 'SMOTE',  # 1
              2: 'BorderlineSMOTE',  #
              3: 'SVMSMOTE',  #
              4: 'ADASYN',  #
              5: 'RandomOverSampler',  # 1
              6: 'RandomUnderSampler',  # 1
              7: 'CondensedNearestNeighbour',  #
              8: 'NearMiss',  # 1
              9: 'TomekLinks',  # 1
              10: 'EditedNearestNeighbours',  #
              11: 'OneSidedSelection',  #
              12: 'NeighbourhoodCleaningRule'}  #

model = './word2vec/GoogleNews-vectors-negative300.bin'
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(model, binary=True)


def get_tf_idf_vector():
    tfidf_file_names = []
    tfidf_dict = {}
    parser = Parser(DATASET)
    src_prep = SrcPreprocessing(parser.src_parser())
    src_prep.preprocess()
    source_files = src_prep.src_files

    src_strings = [' '.join(src.file_name['stemmed'] + src.class_names['stemmed']
                            + src.method_names['stemmed']
                            + src.pos_tagged_comments['stemmed']
                            + src.attributes['stemmed'])
                   for src in source_files.values()]

    tfidf = TfidfVectorizer(sublinear_tf=True, smooth_idf=False)
    src_tfidf = tfidf.fit_transform(src_strings)
    tfidf_array = src_tfidf.toarray()

    for i, key in enumerate(source_files):
        dictKey = key[:-5].replace('.', '/') + '.java'
        tfidf_file_names.append(dictKey)
        tfidf_dict.update({dictKey: tfidf_array[i]})

    return tfidf_file_names, tfidf_dict


def get_embedded():
    embedded_file_names = []
    embedded_dict = {}
    parser = Parser(DATASET)
    src_prep = SrcPreprocessing(parser.src_parser())
    src_prep.preprocess()
    source_files = src_prep.src_files

    for key in source_files:
        dictKey = key[:-5].replace('.', '/') + '.java'
        embedded_file_names.append(dictKey)

        src_text = source_files.get(key).file_name['unstemmed'] + source_files.get(key).class_names['unstemmed'] + source_files.get(key).attributes['unstemmed'] + source_files.get(key).comments['unstemmed'] + source_files.get(key).method_names['unstemmed']
        src_vec = np.zeros((300,))
        for src_token in src_text:
            try:
                src_vec += word_vectors.get_vector(src_token)
            except:
                pass
        src_vec = src_vec / len(src_text)
        embedded_dict.update({dictKey: src_vec})
    return embedded_file_names, embedded_dict


def get_labels(mapped_file):
    file_name = ''
    with open(mapped_file, 'r') as f:
        label_file = csv.reader(f)
        label_file_names = []
        label_dict = {}
        for row in label_file:

            if DATASET.name in ['ant', 'camel', 'lucene', 'log4j', 'xalan', 'velocity', 'synapse', 'poi']:
                index = row[0].find('org/')
                if index != -1:
                    file_name = row[0][index:].replace('.restrictedcontent.embed', '')
                else:
                    file_name = row[0].replace('.restrictedcontent.embed', '')

            if DATASET.name == 'jedit':
                index = max(row[0].find('org/'), row[0].find('gnu/'), row[0].find('bsh/'))
                if index != -1:
                    file_name = row[0][index:].replace('.restrictedcontent.embed', '')
                else:
                    file_name = row[0].replace('.restrictedcontent.embed', '')

            if DATASET.name == 'ivy':
                index = max(row[0].find('org/'), row[0].find('fr/'))
                if index != -1:
                    file_name = row[0][index:].replace('.restrictedcontent.embed', '')
                else:
                    file_name = row[0].replace('.restrictedcontent.embed', '')

            if DATASET.name == 'xerces':
                index = max(row[0].find('org/'), row[0].find('javax/'))
                if index != -1:
                    file_name = row[0][index:].replace('.restrictedcontent.embed', '')
                else:
                    file_name = row[0].replace('.restrictedcontent.embed', '')

            if DATASET.name == 'pbeans':
                index = row[0].find('net/')
                if index != -1:
                    file_name = row[0][index:].replace('.restrictedcontent.embed', '')
                else:
                    file_name = row[0].replace('.restrictedcontent.embed', '')

            label_file_names.append(file_name)

            if row[1] == '0':
                label_dict.update({file_name: 0})
            else:
                label_dict.update({file_name: 1})

    return label_file_names, label_dict


def get_metric(labeled_file):
    with open(labeled_file, 'r') as f:
        metric_file = csv.reader(f)
        metric_file_names = []
        metric_dict = {}
        for row in metric_file:
            if row[0] == 'name':
                continue
            file_name = row[2].replace('.', '/') + '.java'
            if file_name not in metric_file_names:
                metric_file_names.append(file_name)
            else:
                print(file_name)

            metric = []
            for i in range(3, 23):
                metric.append(round(float(row[i]), 2))

            metric_dict.update({file_name: metric})

    return metric_file_names, metric_dict


def extract_features():
    metric_file_names, metric_dict = get_metric(DATASET.labeled)
    label_file_names, label_dict = get_labels(DATASET.mapped)
    embedded_file_names, embedded_dict = get_embedded()
    tfidf_file_names, tfidf_dict = get_tf_idf_vector()
    codebert_file_names, codebert_dict = codebert_features()

    X = []
    X_concat = []
    Y = []

    """W2V"""
    # for file_name in metric_file_names:
    #     if (file_name in label_file_names) and (file_name in embedded_file_names):
    #         X.append(embedded_dict.get(file_name))
    #         X_concat.append(metric_dict.get(file_name))
    #         Y.append(label_dict.get(file_name))

    """Tf-idf"""
    # for file_name in metric_file_names:
    #     if (file_name in label_file_names) and (file_name in tfidf_file_names):
    #         X.append(tfidf_dict.get(file_name))
    #         X_concat.append(metric_dict.get(file_name))
    #         Y.append(label_dict.get(file_name))

    """CodeBert"""
    for file_name in metric_file_names:
        if (file_name in label_file_names) and (file_name in codebert_file_names):
            X.append(codebert_dict.get(file_name))
            X_concat.append(metric_dict.get(file_name))
            Y.append(label_dict.get(file_name))

    return X, X_concat, Y


def save_data(i, X_train, X_concat_train, Y_train, X_test, X_concat_test, Y_test):
    np.savetxt('./codebert/' + DATASET.name + '_' + DATASET.version + '/' + Techniques[i] + '_X_train.txt',
               X_train)
    np.savetxt(
        './codebert/' + DATASET.name + '_' + DATASET.version + '/' + Techniques[i] + '_X_concat_train.txt',
        X_concat_train)
    np.savetxt('./codebert/' + DATASET.name + '_' + DATASET.version + '/' + Techniques[i] + '_Y_train.txt',
               Y_train)
    np.savetxt('./codebert/' + DATASET.name + '_' + DATASET.version + '/' + Techniques[i] + '_X_test.txt',
               X_test)
    np.savetxt('./codebert/' + DATASET.name + '_' + DATASET.version + '/' + Techniques[i] + '_X_concat_test.txt',
               X_concat_test)
    np.savetxt('./codebert/' + DATASET.name + '_' + DATASET.version + '/' + Techniques[i] + '_Y_test.txt',
               Y_test)


def load_data(i):
    X_train = np.loadtxt(
        './codebert/' + DATASET.name + '_' + DATASET.version + '/' + Techniques[i] + '_X_train.txt')
    X_concat_train = np.loadtxt(
        './codebert/' + DATASET.name + '_' + DATASET.version + '/' + Techniques[i] + '_X_concat_train.txt')
    Y_train = np.loadtxt(
        './codebert/' + DATASET.name + '_' + DATASET.version + '/' + Techniques[i] + '_Y_train.txt')
    X_test = np.loadtxt(
        './codebert/' + DATASET.name + '_' + DATASET.version + '/' + Techniques[i] + '_X_test.txt')
    X_concat_test = np.loadtxt(
        './codebert/' + DATASET.name + '_' + DATASET.version + '/' + Techniques[i] + '_X_concat_test.txt')
    Y_test = np.loadtxt(
        './codebert/' + DATASET.name + '_' + DATASET.version + '/' + Techniques[i] + '_Y_test.txt')

    return X_train, X_concat_train, Y_train, X_test, X_concat_test, Y_test


if __name__ == '__main__':
    X, X_concat, Y = extract_features()
