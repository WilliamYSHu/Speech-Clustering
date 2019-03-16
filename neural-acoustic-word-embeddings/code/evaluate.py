from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# --
from os import path, makedirs
import numpy as np
import tensorflow as tf
from model import Model
from data import Dataset
from average_precision import average_precision

class Config(object):
    """Set up model for debugging."""

    trainfile = "../kaldi/data/len6-50frames-count2/train/mfcc.scp"
    devfile = "../kaldi/data/len6-50frames-count2/dev/mfcc.scp"
    batch_size = 32
    current_epoch = 0
    num_epochs = 100
    feature_dim = 39
    feature_mean = -0.0054857
    num_layers = 3
    hidden_size = 256
    bidirectional = True
    keep_prob = 0.7
    margin = 0.5
    max_same = 1
    max_diff = 5
    lr = 0.001
    mom = 0.9
    logdir = "../logs/hystest6"
    ckptdir = "../ckpts/hystest6"
    log_interval = 100
    ckpt = None
    debugmode = True

    makedirs(logdir, exist_ok=True)
    makedirs(ckptdir, exist_ok=True)


def main():
    config = Config()

    full_dataset = np.load('swbd_test2.npy').item()

    train_data = Dataset(full_dataset, partition="train", config=config)
    dev_data = Dataset(full_dataset, partition="dev", config=config)
    test_data = Dataset(full_dataset, partition="test", config=config)

    train_model = Model(is_train=True, config=config, reuse=None)
    dev_model = Model(is_train=False, config=config, reuse=True)

    batch_size = config.batch_size

    saver = tf.train.Saver()

    proto = tf.ConfigProto(intra_op_parallelism_threads=2)
    with tf.Session(config=proto) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(config.ckptdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restored from %s" % ckpt.model_checkpoint_path)

        embeddings = {}
        for partition, data in [("train", train_data), ("dev", dev_data), ("test", test_data)]:
            embeddings[partition] = {"embs": [], "labels": [], "ids": []}
            for x, ts, ids, labels in data.batch_for_evaluation(batch_size):
                embeddings[partition]["embs"].append(dev_model.get_embeddings(sess, x, ts))
                embeddings[partition]["labels"].append(labels)
                embeddings[partition]["ids"].append(ids)
            embeddings[partition]["embs"] = np.concatenate(embeddings[partition]["embs"])
            embeddings[partition]["labels"] = np.concatenate(embeddings[partition]["labels"])
            embeddings[partition]["ids"] = np.concatenate(embeddings[partition]["ids"])
            print("%s ap: %.4f" % (partition, average_precision(embeddings[partition]["embs"], embeddings[partition]["ids"])))
        np.save("swbd_rest_embeddings.npy", embeddings)

if __name__ == "__main__":
    main()
