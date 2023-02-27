from tensorflow.keras import Input, Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, concatenate


def cnn_model(X_train, src_vec, filter_size=5):
    A1 = Input(shape=(X_train.shape[1]))
    AA1 = Input(shape=20)

    A2 = Embedding(input_dim=src_vec.shape[0], output_dim=32, input_length=X_train.shape[1],
                   embeddings_regularizer=regularizers.l2(0.0001), activity_regularizer=regularizers.l2(0.0001),
                   embeddings_initializer='he_normal')(A1)

    A3 = Conv1D(filters=32, strides=2, kernel_size=filter_size, activation='relu', kernel_initializer='he_normal')(A2)
    A4 = MaxPooling1D()(A3)

    A5 = Conv1D(filters=32, strides=2, kernel_size=filter_size, activation='relu', kernel_initializer='he_normal')(A4)
    A6 = MaxPooling1D()(A5)

    A7 = Conv1D(filters=64, strides=2, kernel_size=filter_size, activation='relu', kernel_initializer='he_normal')(A6)
    A8 = MaxPooling1D()(A7)

    A9 = Flatten()(A8)

    A10 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001),
                activity_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.0001),
                kernel_initializer='he_normal', bias_initializer='he_normal')(A9)
    A11 = Dropout(0.5)(A10)
    A12 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001),
                activity_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.0001),
                kernel_initializer='he_normal', bias_initializer='he_normal')(A11)
    A13 = Dropout(0.5)(A12)
    A14 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001),
                activity_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.0001),
                kernel_initializer='he_normal', bias_initializer='he_normal')(A13)
    A15 = Dropout(0.5)(A14)
    A16 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001),
                activity_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.0001),
                kernel_initializer='he_normal', bias_initializer='he_normal')(A15)

    A17 = concatenate([A16, AA1])

    A18 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001),
                activity_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.0001),
                kernel_initializer='he_normal', bias_initializer='he_normal')(A17)
    A19 = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001),
                activity_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.0001),
                kernel_initializer='glorot_normal', bias_initializer='glorot_normal')(A18)

    model = Model(inputs=[A1, AA1], outputs=A19)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
