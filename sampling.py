from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour, NearMiss, TomekLinks, \
    EditedNearestNeighbours, OneSidedSelection, NeighbourhoodCleaningRule


def smote(X_train, Y_train, kNeighbors):
    sampler = SMOTE(k_neighbors=kNeighbors)
    X_train_new, Y_train_new = sampler.fit_resample(X_train, Y_train)
    return X_train_new, Y_train_new


def borderline_smote(X_train, Y_train, kNeighbors):
    sampler = BorderlineSMOTE(k_neighbors=kNeighbors)
    X_train_new, Y_train_new = sampler.fit_resample(X_train, Y_train)
    return X_train_new, Y_train_new


def svm_smote(X_train, Y_train, kNeighbors):
    sampler = SVMSMOTE(k_neighbors=kNeighbors)
    X_train_new, Y_train_new = sampler.fit_resample(X_train, Y_train)
    return X_train_new, Y_train_new


def adasyn(X_train, Y_train, kNeighbors):
    sampler = ADASYN(n_neighbors=kNeighbors)
    X_train_new, Y_train_new = sampler.fit_resample(X_train, Y_train)
    return X_train_new, Y_train_new


def random_over_sampler(X_train, Y_train):
    sampler = RandomOverSampler()
    X_train_new, Y_train_new = sampler.fit_resample(X_train, Y_train)
    return X_train_new, Y_train_new


def random_under_sampler(X_train, Y_train):
    sampler = RandomUnderSampler()
    X_train_new, Y_train_new = sampler.fit_resample(X_train, Y_train)
    return X_train_new, Y_train_new


def condensed_nearest_neighbour(X_train, Y_train):
    sampler = CondensedNearestNeighbour()
    X_train_new, Y_train_new = sampler.fit_resample(X_train, Y_train)
    return X_train_new, Y_train_new


def near_miss(X_train, Y_train):
    sampler = NearMiss()
    X_train_new, Y_train_new = sampler.fit_resample(X_train, Y_train)
    return X_train_new, Y_train_new


def tomek_links(X_train, Y_train):
    sampler = TomekLinks()
    X_train_new, Y_train_new = sampler.fit_resample(X_train, Y_train)
    return X_train_new, Y_train_new


def edited_nearest_neighbours(X_train, Y_train):
    sampler = EditedNearestNeighbours()
    X_train_new, Y_train_new = sampler.fit_resample(X_train, Y_train)
    return X_train_new, Y_train_new


def one_sided_selection(X_train, Y_train):
    sampler = OneSidedSelection()
    X_train_new, Y_train_new = sampler.fit_resample(X_train, Y_train)
    return X_train_new, Y_train_new


def neighbourhood_cleaning_rule(X_train, Y_train):
    sampler = NeighbourhoodCleaningRule()
    X_train_new, Y_train_new = sampler.fit_resample(X_train, Y_train)
    return X_train_new, Y_train_new


if __name__ == '__main__':
    print()
