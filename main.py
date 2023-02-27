import random

from feature_extraction import *
from util import *
from sampling import *
from NewCNN import cnn_model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def processing():
    X, X_concat, Y = extract_features()
    X_merge = merge(X, X_concat)
    X_merge_train, X_merge_test, Y_train, Y_test, kNeighbors = train_test_split(X_merge, Y)

    # Original
    X_train, X_concat_train = split(X, X_merge_train)
    X_test, X_concat_test = split(X, X_merge_test)
    save_data(0, X_train, X_concat_train, Y_train, X_test, X_concat_test, Y_test)

    # SMOTE
    X_merge_train_new, Y_train_new = smote(X_merge_train, Y_train, kNeighbors)
    X_train, X_concat_train = split(X, X_merge_train_new)
    X_test, X_concat_test = split(X, X_merge_test)
    save_data(1, X_train, X_concat_train, Y_train_new, X_test, X_concat_test, Y_test)

    # BorderlineSMOTE
    X_merge_train_new, Y_train_new = borderline_smote(X_merge_train, Y_train, kNeighbors)
    X_train, X_concat_train = split(X, X_merge_train_new)
    X_test, X_concat_test = split(X, X_merge_test)
    save_data(2, X_train, X_concat_train, Y_train_new, X_test, X_concat_test, Y_test)

    # SVMSMOTE
    X_merge_train_new, Y_train_new = svm_smote(X_merge_train, Y_train, kNeighbors)
    X_train, X_concat_train = split(X, X_merge_train_new)
    X_test, X_concat_test = split(X, X_merge_test)
    save_data(3, X_train, X_concat_train, Y_train_new, X_test, X_concat_test, Y_test)

    # ADASYN
    X_merge_train_new, Y_train_new = adasyn(X_merge_train, Y_train, kNeighbors)
    X_train, X_concat_train = split(X, X_merge_train_new)
    X_test, X_concat_test = split(X, X_merge_test)
    save_data(4, X_train, X_concat_train, Y_train_new, X_test, X_concat_test, Y_test)

    # RandomOverSampler
    X_merge_train_new, Y_train_new = random_over_sampler(X_merge_train, Y_train)
    X_train, X_concat_train = split(X, X_merge_train_new)
    X_test, X_concat_test = split(X, X_merge_test)
    save_data(5, X_train, X_concat_train, Y_train_new, X_test, X_concat_test, Y_test)

    # RandomUnderSampler
    X_merge_train_new, Y_train_new = random_under_sampler(X_merge_train, Y_train)
    X_train, X_concat_train = split(X, X_merge_train_new)
    X_test, X_concat_test = split(X, X_merge_test)
    save_data(6, X_train, X_concat_train, Y_train_new, X_test, X_concat_test, Y_test)

    # CondensedNearestNeighbour
    X_merge_train_new, Y_train_new = condensed_nearest_neighbour(X_merge_train, Y_train)
    X_train, X_concat_train = split(X, X_merge_train_new)
    X_test, X_concat_test = split(X, X_merge_test)
    save_data(7, X_train, X_concat_train, Y_train_new, X_test, X_concat_test, Y_test)

    # NearMiss
    X_merge_train_new, Y_train_new = near_miss(X_merge_train, Y_train)
    X_train, X_concat_train = split(X, X_merge_train_new)
    X_test, X_concat_test = split(X, X_merge_test)
    save_data(8, X_train, X_concat_train, Y_train_new, X_test, X_concat_test, Y_test)

    # TomekLinks
    X_merge_train_new, Y_train_new = tomek_links(X_merge_train, Y_train)
    X_train, X_concat_train = split(X, X_merge_train_new)
    X_test, X_concat_test = split(X, X_merge_test)
    save_data(9, X_train, X_concat_train, Y_train_new, X_test, X_concat_test, Y_test)

    # EditedNearestNeighbours
    X_merge_train_new, Y_train_new = edited_nearest_neighbours(X_merge_train, Y_train)
    X_train, X_concat_train = split(X, X_merge_train_new)
    X_test, X_concat_test = split(X, X_merge_test)
    save_data(10, X_train, X_concat_train, Y_train_new, X_test, X_concat_test, Y_test)

    # OneSidedSelection
    X_merge_train_new, Y_train_new = one_sided_selection(X_merge_train, Y_train)
    X_train, X_concat_train = split(X, X_merge_train_new)
    X_test, X_concat_test = split(X, X_merge_test)
    save_data(11, X_train, X_concat_train, Y_train_new, X_test, X_concat_test, Y_test)

    # NeighbourhoodCleaningRule
    X_merge_train_new, Y_train_new = neighbourhood_cleaning_rule(X_merge_train, Y_train)
    X_train, X_concat_train = split(X, X_merge_train_new)
    X_test, X_concat_test = split(X, X_merge_test)
    save_data(12, X_train, X_concat_train, Y_train_new, X_test, X_concat_test, Y_test)


def pred(prediction):
    Y_pred = []
    for y in prediction:
        if y < 0.5:
            Y_pred.append(0)
        if y >= 0.5:
            Y_pred.append(1)
    return Y_pred


def main():
    data_path = glob.glob('./codebert/' + DATASET.name + '_' + DATASET.version + '/*.txt')
    if len(data_path) != len(Techniques) * 6:
        print('Running processing...')
        processing()

    for i in range(len(Techniques)):
        total_accuracy = []
        total_f1 = []
        total_auc = []

        X_train, X_concat_train, Y_train, X_test, X_concat_test, Y_test = load_data(i)
        src_vec = np.array(random.sample(range(X_train.shape[0]), X_train.shape[0]))

        for time in range(20):
            if time % 10 == 9:
                print('Training ' + str(time + 1) + ' times...')
            my_model = cnn_model(X_train, src_vec)
            my_model.fit([X_train, X_concat_train], Y_train, batch_size=64, epochs=30, verbose=0)
            prediction = my_model.predict([X_test, X_concat_test], verbose=0)
            Y_pred = pred(prediction)

            total_accuracy.append(accuracy_score(Y_test, Y_pred))
            total_f1.append(f1_score(Y_test, Y_pred))
            total_auc.append(roc_auc_score(Y_test, Y_pred))

        print('Evaluating ' + DATASET.name + '_' + DATASET.version + '_' + Techniques[i] + '...')
        print('Accuracy', round(sum(total_accuracy) / len(total_accuracy) * 100, 1))
        print('F1', round(sum(total_f1) / len(total_f1) * 100, 1))
        print('AUC', round(sum(total_auc) / len(total_auc) * 100, 1))


if __name__ == '__main__':
    main()
