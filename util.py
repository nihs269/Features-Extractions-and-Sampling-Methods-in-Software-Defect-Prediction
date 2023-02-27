TEST_SIZE = 0.2


def merge(X, X_concat):
    X_merge = []
    for index in range(len(X)):
        X_merge.append(list(X[index]) + list(X_concat[index]))
    return X_merge


def split(X, X_merge):
    X_train_new = []
    X_concat_train_new = []
    for sample in X_merge:
        X_train_new.append(sample[0: len(X[0])])
        X_concat_train_new.append(sample[len(X[0]): len(sample)])
    return X_train_new, X_concat_train_new


def calculate_imb_ratio(y):
    num_of_label_0 = 0
    num_of_label_1 = 0
    label_0 = []
    label_1 = []
    for i in range(len(y)):
        if y[i] == 0:
            num_of_label_0 += 1
            label_0.append(i)
        if y[i] == 1:
            num_of_label_1 += 1
            label_1.append(i)
    imb_ratio = num_of_label_0 / num_of_label_1

    return imb_ratio, label_1, label_0


def train_test_split(x, y):
    ratio, label_1, label_0 = calculate_imb_ratio(y)
    maj_class = []
    min_class = []
    if ratio > 1:
        maj_class = label_0
        min_class = label_1
    if ratio < 1:
        maj_class = label_1
        min_class = label_0
        ratio = 1 / ratio

    x_train = []
    y_train = []

    kNeighbors = 0
    for i in range(int(len(x) * (1 - TEST_SIZE) / (round(ratio) + 1))):
        for j in range(round(ratio)):
            x_train.append(x[maj_class[i + j]])
            y_train.append(y[maj_class[i + j]])
        x_train.append(x[min_class[i]])
        y_train.append(y[min_class[i]])
        kNeighbors += 1

    x_test = x[len(x_train):len(x)]
    y_test = y[len(y_train):len(y)]

    return x_train, x_test, y_train, y_test, kNeighbors - 1
