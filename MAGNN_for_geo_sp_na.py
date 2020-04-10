import numpy as np
import tensorflow as tf

from models import GAT, SpGAT
from utils import process
from geo_dataProcess import load_obj, process_data, geo_accAT161

# training params
batch_size = 1
nb_epochs = 100000
patience = 30
lr = 0.001              # learning rate
l2_coef = 0.0005        # weight decay
hid_units = [4]        # n umbers of hidden units per each attention head in each layer
n_heads = [4, 1]       # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
# model = GAT
model = SpGAT

# for cmu:
# dataset = './geo_data/cmu/dump_doc_dim_512.pkl'
# dataset = './geo_data/cmu/dump_doc_dim_1024.pkl'
# trained_model_path = 'geo_data/model/cmu_model_head{}'.format(n_heads[0])

# for na:
dataset = './geo_data/na/dump_doc_dim_512.pkl'
trained_model_path = 'geo_data/model/na_model_head{}'.format(n_heads[0])

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

sparse = True

'''Morton wang added for geolocation prediction.'''
# load data.
data = load_obj(filename=dataset)
data = process_data(data, class_num=256)
(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, idx_val, idx_test,
 U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation) = data

# features = process.preprocess_features_geodata(features)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

features = features[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

if sparse:
    biases = process.preprocess_adj_bias(adj)
    bbias = biases
else:
    adj = adj.todense()
    adj = adj[np.newaxis]
    biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)
    bbias = biases[0:1]
'''Morton wang added for geolocation prediction.'''

''' original code for classification'''
# dataset = 'cora'
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
# features, spars = process.preprocess_features(features)
#
# nb_nodes = features.shape[0]
# ft_size = features.shape[1]
# nb_classes = y_train.shape[1]
#
# features = features[np.newaxis]
# y_train = y_train[np.newaxis]
# y_val = y_val[np.newaxis]
# y_test = y_test[np.newaxis]
# train_mask = train_mask[np.newaxis]
# val_mask = val_mask[np.newaxis]
# test_mask = test_mask[np.newaxis]
#
# if sparse:
#     biases = process.preprocess_adj_bias(adj)
#     bbias = biases
# else:
#     adj = adj.todense()
#     adj = adj[np.newaxis]
#     biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)
#     bbias = biases[0:1]
''' original code for classification'''

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        if sparse:
            bias_in = tf.sparse_placeholder(dtype=tf.float32)
        else:
            bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train, attn_drop, ffd_drop, hid_units=hid_units,
                             n_heads=n_heads, bias_mat=bias_in, residual=residual, activation=nonlinearity)

    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])

    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, mask=msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, mask=msk_resh)

    train_op = model.training(loss, lr, l2_coef)
    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vacc_mx, curr_step = -1, 0
    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(nb_epochs):
            '''train process'''
            train_feed_dict = {ftr_in: features[0:1],  bias_in: bbias, lbl_in: y_train[0:1],
                               msk_in: train_mask[0:1], is_train: True, attn_drop: 0.0, ffd_drop: 0.0}
            _, loss_value_tr, logit_value, acc_classify_tr = sess.run(
                [train_op, loss, logits, accuracy], feed_dict=train_feed_dict)
            # acc161_tr = geo_accAT161(logit_value[0][idx_train], U_train, classLatMedian, classLonMedian, userLocation)

            '''validate process'''
            val_feed_dict = {ftr_in: features[0:1],  bias_in: bbias, lbl_in: y_val[0:1],
                             msk_in: val_mask[0:1], is_train: False, attn_drop: 0.0, ffd_drop: 0.0}
            loss_value_vl, logit_value, acc_classify_vl = sess.run(
                [loss, logits, accuracy], feed_dict=val_feed_dict)
            acc161_vl = geo_accAT161(logit_value[0][idx_val], U_dev, classLatMedian, classLonMedian, userLocation)

            '''show information'''
            print('epoch %d\t Training: loss = %.5f, acc_classify = %.5f, acc@161 = %.5f | '
                  'Val: loss = %.5f, acc_classify = %.5f, acc@161 = %.5f' %
                  (epoch, loss_value_tr, acc_classify_tr, 0, loss_value_vl, acc_classify_vl, acc161_vl))

            '''early stop'''
            if acc161_vl >= vacc_mx:
                vacc_mx = acc161_vl
                curr_step = 0
                saver.save(sess=sess, save_path=trained_model_path)
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Max acc@161: ', vacc_mx)
                    break

        '''test process'''
        saver.restore(sess, save_path=trained_model_path)
        test_feed_dict = {ftr_in: features[0:1],  bias_in: bbias, lbl_in: y_test[0:1],
                          msk_in: test_mask[0:1], is_train: False, attn_drop: 0.0, ffd_drop: 0.0}
        loss_value_ts, logit_value, acc_classify_ts = sess.run(
            [loss, logits, accuracy], feed_dict=test_feed_dict)
        acc161_ts = geo_accAT161(logit_value[0][idx_test], U_test, classLatMedian, classLonMedian, userLocation)
        print('Test loss:', loss_value_ts, '; Test acc_classify:', acc_classify_ts, '; Test acc@161:', acc161_ts)

        sess.close()
