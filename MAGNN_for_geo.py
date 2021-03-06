import time
import numpy as np
import tensorflow as tf

from models import GAT
from utils import process
from geo_dataProcess import load_obj, process_data, geo_accAT161

from sklearn.metrics import recall_score, precision_score, f1_score

checkpt_file = 'geo_data/model/mod_cmu.ckpt'

dataset = './geo_data/cmu/dump_doc_dim_512.pkl'

# training params
batch_size = 1
nb_epochs = 100000
patience = 30
lr = 0.05          # learning rate
l2_coef = 0.0005    # weight decay
hid_units = [4]     # numbers of hidden units per each attention head in each layer
n_heads = [4, 1]    # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT

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

# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data('cora')

# load data.
data = load_obj(filename=dataset)
data = process_data(data)
(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, idx_val, idx_test,
 U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation) = data

# GAT initial code.
features = process.preprocess_features_geodata(features)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

adj = adj.todense()

features = features[np.newaxis]
adj = adj[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                hid_units = hid_units, n_heads = n_heads,bias_mat = bias_in,
                                residual = residual, activation = nonlinearity)

    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])

    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, mask=msk_resh)
    # accuracy = model.masked_accuracy(log_resh, lab_resh, mask=msk_resh)

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = features.shape[0]

            while tr_step * batch_size < tr_size:
                _, loss_value_tr, logit_value = sess.run([train_op, loss, log_resh],
                    feed_dict={
                        ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_in: biases[tr_step*batch_size:(tr_step+1)*batch_size],
                        lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: True,
                        attn_drop: 0.6, ffd_drop: 0.6})
                acc_tr = geo_accAT161(logit_value[idx_train], U_train, classLatMedian, classLonMedian, userLocation)
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = features.shape[0]

            while vl_step * batch_size < vl_size:
                loss_value_vl, logit_value, lab_val = sess.run([loss, log_resh, lab_resh],
                    feed_dict={
                        ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_in: biases[vl_step*batch_size:(vl_step+1)*batch_size],
                        lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                acc_vl = geo_accAT161(logit_value[idx_val], U_dev, classLatMedian, classLonMedian, userLocation)
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1
                # calculate recall, precision and f1-score
                lab_val = np.argmax(lab_val, axis=1)[idx_val]
                logits_value = np.argmax(logit_value, axis=1)[idx_val]
                recall = recall_score(lab_val, logits_value, average='macro')
                precision = precision_score(lab_val, logits_value, average='macro')
                f_value = f1_score(lab_val, logits_value, average='macro')

            print('epoch %d\t Training: loss = %.5f, acc@161 = %.5f | Val: loss = %.5f, acc@161 = %.5f' %
                    (epoch, train_loss_avg/tr_step, train_acc_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step))
            print(f"recall:{recall}, precision:{precision}, f1-score:{f_value}")

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max acc@161: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', acc@161: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)

        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0
        ts_size = features.shape[0]

        while ts_step * batch_size < ts_size:
            loss_value_ts, logit_value, lab_test = sess.run([loss, log_resh, lab_resh],
                feed_dict={
                    ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            acc_ts = geo_accAT161(logit_value[idx_test], U_test, classLatMedian, classLonMedian, userLocation)
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1
            # calculate recall, precision and f1-score
            lab_val = np.argmax(lab_val, axis=1)[idx_test]
            logits_value = np.argmax(logit_value, axis=1)[idx_test]
            recall = recall_score(lab_val, logits_value, average='macro')
            precision = precision_score(lab_val, logits_value, average='macro')
            f_value = f1_score(lab_val, logits_value, average='macro')

        print('Test loss:', ts_loss/ts_step, '; Test acc@161:', ts_acc/ts_step)
        print(f"recall:{recall}, precision:{precision}, f1-score:{f_value}")

        sess.close()
