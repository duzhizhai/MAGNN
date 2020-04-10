# -*- coding: UTF-8 -*-

import gzip
import pickle
import hickle
import numpy as np
import joblib
from haversine import haversine


def dump_obj(obj, filename, protocol=-1, serializer=pickle, usejoblib=False):
    if usejoblib:
        joblib.dump(obj, filename)
    elif serializer == hickle:
        serializer.dump(obj, filename, mode='w', compression='gzip')
    else:
        with gzip.open(filename, 'wb') as fout:
            serializer.dump(obj, fout, protocol)


def load_obj(filename, serializer=pickle, usejoblib=False):
    if usejoblib:
        obj = joblib.load(filename)
    elif serializer == hickle:
        obj = serializer.load(filename)
    else:
        with gzip.open(filename, 'rb') as fin:
            obj = serializer.load(fin)
    return obj


def get_one_hot(y_label, class_num):
    one_hot_index = np.arange(len(y_label)) * class_num + y_label
    one_hot = np.zeros((len(y_label), class_num))
    one_hot.flat[one_hot_index] = 1
    return one_hot


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def process_data(data, class_num=129):
    adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, \
        classLatMedian, classLonMedian, userLocation, tf_idf_sum = data

    '''use bert features for all user.'''
    features = np.vstack((X_train, X_dev, X_test))  # only for dec2vec_feature
    print("feature shape:{}".format(features.shape))

    '''get labels'''
    labels = np.hstack((Y_train, Y_dev, Y_test))
    labels = get_one_hot(labels, class_num=class_num)

    '''get index of train val and test'''
    len_train, len_val, len_test = int(X_train.shape[0]), int(X_dev.shape[0]), int(X_test.shape[0])
    idx_train, idx_val, idx_test = range(len_train), range(len_train, len_train+len_val), \
                                   range(len_train+len_val, len_train+len_val+len_test)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train, y_val, y_test = np.zeros(labels.shape), np.zeros(labels.shape), np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    data = (adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,
            idx_train, idx_val, idx_test, U_train, U_dev, U_test,
            classLatMedian, classLonMedian, userLocation)
    return data


###############################
# Morton wang add for geolocation prediction.
###############################

def geo_accAT161(logit_val, U_test, classLatMedian, classLonMedian, userLocation):

    y_pred = np.argmax(logit_val, axis=1)  # 1代表行

    assert len(y_pred) == len(U_test), "#preds: %d, #users: %d" % (len(y_pred), len(U_test))

    distances = []
    for i in range(0, len(y_pred)):
        user = U_test[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        prediction = str(y_pred[i])
        lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    return acc_at_161

