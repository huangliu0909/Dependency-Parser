import time
from my_model import *


def test():
    w, p, d, t = get_input("feature")

    w_tensor = tf.placeholder(tf.float32, shape=[None, w_input_node], name="w_tensor")
    p_tensor = tf.placeholder(tf.float32, shape=[None, p_input_node], name="p_tensor")
    d_tensor = tf.placeholder(tf.float32, shape=[None, d_input_node], name="d_tensor")
    y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="y_")
    # regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    y = inference(w_tensor, p_tensor, d_tensor)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    variable_averages = tf.train.ExponentialMovingAverage(0.99)
    variables_to_store = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_store)
    F = open("./Model/dev_result_1.txt", "w+")
    while True:
        with tf.Session() as sess:

            ckpt = tf.train.get_checkpoint_state("./Model/")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict={w_tensor: w,
                                                               p_tensor: p,
                                                               d_tensor: d,
                                                               y_: t})
                print("after %s training steps, validation accuracy = %g" % (global_step, accuracy_score))
                F.write(str([global_step, accuracy_score]) + "\n")
                F.close()
                F = open("./Model/dev_result_1.txt", "a")
            else:
                print("model not found!")

        time.sleep(5)


if __name__ == "__main__":
    test()