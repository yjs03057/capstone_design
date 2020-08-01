from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras import regularizers
from random import shuffle

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

def shuffle_data(xFeed, yFeed):
    xFeed_shuf = []
    yFeed_shuf = []
    index_shuf = list(range(len(xFeed)))
    shuffle(index_shuf)
    shuffle(index_shuf)
    for i in index_shuf:
        xFeed_shuf.append(xFeed[i])
        yFeed_shuf.append(yFeed[i])
    return xFeed_shuf, yFeed_shuf


def direction_model(input_shape, optimizer):
    input = Input(shape=input_shape)
    l2 = regularizers.l2(0.01)
    lrelu = LeakyReLU(alpha=0.1)
    x = Conv2D(16, kernel_size=5, activation=lrelu)(input)

    for i in range(3):
        s1 = Conv2D(32, kernel_size=3, kernel_regularizer='l2')(x)
        s1 = BatchNormalization()(s1)
        s1 = Activation('relu')(s1)
        s1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(s1)

        s2 = Conv2D(32, kernel_size=1, kernel_regularizer='l2')(x)
        s2 = BatchNormalization()(s2)
        s2 = Activation('relu')(s2)
        s2 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(s2)

        s3 = Conv2D(32, kernel_size=5, kernel_regularizer='l2')(x)
        s3 = BatchNormalization()(s3)
        s3 = Activation('relu')(s3)
        s3 = MaxPooling2D(pool_size=(1, 1), strides=(2, 2))(s3)
        x = concatenate([s1, s2, s3], 1)
        x = lrelu(x)

    x = Conv2D(64, kernel_size=1, activation=lrelu, kernel_regularizer=l2)(x)
    x = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(x)
    x = Conv2D(128, kernel_size=5, activation=lrelu, kernel_regularizer=l2)(x)
    x = MaxPooling2D(pool_size=(1, 1), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(50)(x)
    x = lrelu(x)
    x = Dropout(0.5)(x)
    x = Dense(6)(x)
    y = Softmax()(x)
    model = Model(inputs=input, outputs=y)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

def gender_model():
    l2 = regularizers.l2(0.01)
    model = Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer='l2',
               input_shape=(40, 150, 1)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer='l2'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer='l2'),
        BatchNormalization(),
        Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(1024, activation='relu', kernel_regularizer='l2'),

        Dense(512, activation='relu', kernel_regularizer='l2'),

        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

def prepare_train(feature, label, divide):
    FEATURE_DIR = '/content/drive/My Drive/capstone/'

    features = np.load(FEATURE_DIR + feature + '.npy')
    labels_np = np.load(FEATURE_DIR + label + '.npy')

    enc = OneHotEncoder()
    enc.fit(labels_np.reshape(-1, 1))
    labels = enc.transform(labels_np.reshape(-1, 1)).toarray()

    features, labels = shuffle_data(features, labels)

    train_data = features[:divide]
    train_label = labels[:divide]
    test_data = features[divide:]
    test_label = labels[divide:]

    train_data = np.array(train_data)
    train_data = train_data[:, :, :, np.newaxis, ]
    test_data = np.array(test_data)
    test_data = test_data[:, :, :, np.newaxis, ]

    train_label = np.array(train_label)
    test_label = np.array(test_label)

    return train_data, test_data, train_label, test_label

def direction_train():
    train_data, test_data, train_label, test_label = prepare_train('/feature2', '/label2', 1440)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    mc = ModelCheckpoint('direction_best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    input_shape = (120, 150, 1)
    model = direction_model(input_shape, Nadam())

    history = model.fit(train_data,
                          train_label,
                          batch_size=128,
                          epochs=300,
                          verbose=1,
                          validation_split=0.2,
                          callbacks=[es, mc])
    model.save('final_direction_model.h5')

    score = model.evaluate(test_data, test_label, verbose=1)
    print("최종 정확도 : " + str(score[1] * 100) + " %")

def gender_train():
    train_data, test_data, train_label, test_label = prepare_train('/feature3', '/label3', 2880)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    mc = ModelCheckpoint('gender_best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    model = gender_model()

    history = model.fit(train_data,
                          train_label,
                          batch_size=128,
                          epochs=150,
                          verbose=1,
                          validation_split=0.2,
                          callbacks=[es, mc])

    model.save('final_gender_model.h5')

    score = model.evaluate(test_data, test_label, verbose=1)
    print("최종 정확도 : " + str(score[1] * 100) + " %")

direction_train()
gender_train()


