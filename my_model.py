import tensorflow as tf
import math
n_classes = 3
dropout = 0.5
batch_size = 1024
lr = 0.0001
n_epochs = 10
learning_rate_decay = 0.99
w_input_node = 18
p_input_node = 18
d_input_node = 12
l1_hidden_size = 200
l2_hidden_size = 200
regularization_rate = 0.001
training_steps = 300000
beta_regul=10e-7
embedded = 50


def get_input(s):
    w = []
    with open("data/word_" + s + ".txt", "r") as f:
        for line in f:
            w.append([float(v) for v in line.replace("\n", "").replace("[", "").replace("]", "").split(",")])
    p = []
    with open("data/pos_" + s + ".txt", "r") as f:
        for line in f:
            p.append([float(v) for v in line.replace("\n", "").replace("[", "").replace("]", "").split(",")])
    d = []
    with open("data/dep_" + s + ".txt", "r") as f:
        for line in f:
            d.append([float(v) for v in line.replace("\n", "").replace("[", "").replace("]", "").split(",")])
    t = []
    with open("data/trans_" + s + ".txt", "r") as f:
        for line in f:
            t.append([float(v) for v in line.replace("\n", "").replace("[", "").replace("]", "").split(",")])
    return w, p, d, t


def get_weight_variable(shape, name):
    '''
    weights = tf.get_variable(
        name, shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    '''
    val = math.sqrt(6. / sum(shape))
    weights = tf.get_variable(shape=shape, dtype=tf.float32,
                          initializer=tf.random_uniform_initializer(minval=-val, maxval=val, dtype=tf.float32), name=name)
    '''
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    '''
    return weights


def inference(w_tensor, p_tensor, d_tensor):
    w_tensor = tf.nn.l2_normalize(w_tensor, dim=0)
    p_tensor = tf.nn.l2_normalize(p_tensor, dim=0)
    d_tensor = tf.nn.l2_normalize(d_tensor, dim=0)
    with tf.variable_scope('layer1'):
        w11 = get_weight_variable([w_input_node, l1_hidden_size], "w11")
        w12 = get_weight_variable([p_input_node, l1_hidden_size], "w12")
        w13 = get_weight_variable([d_input_node, l1_hidden_size], "w13")
        b1 = tf.Variable(tf.random_uniform([l1_hidden_size, ]))
        preactivations = tf.pow(tf.add_n([tf.matmul(w_tensor, w11),
                                          tf.matmul(p_tensor, w12),
                                          tf.matmul(d_tensor, w13)]) + b1, 3, name="preactivations")
        layer1 = tf.nn.relu(preactivations)
    '''
    with tf.variable_scope("layer_2"):
        w2 = get_weight_variable([l1_hidden_size, l2_hidden_size], "w2", regularizer)
        b2 = tf.Variable(tf.random_uniform([l2_hidden_size, ]))
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, w2), b2), name="activations")
    '''
    with tf.variable_scope("layer_3"):
        w3 = get_weight_variable([l1_hidden_size, n_classes], "w3")
        b3 = tf.Variable(tf.random_uniform([n_classes, ]))
        predictions = tf.nn.relu(tf.add(tf.matmul(layer1, w3), b3), name="activations")
        # predictions = tf.add(tf.matmul(layer1, w3), b3)

    return predictions


# def create_embedding():


def train():
    # input
    w_tensor = tf.placeholder(tf.float32, shape=[None, w_input_node], name="w_tensor")
    p_tensor = tf.placeholder(tf.float32, shape=[None, p_input_node], name="p_tensor")
    d_tensor = tf.placeholder(tf.float32, shape=[None, d_input_node], name="d_tensor")
    y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="y_")
    # regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    y = inference(w_tensor, p_tensor, d_tensor)
    global_step = tf.Variable(0, trainable=False)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # loss = tf.reduce_mean(cross_entropy + beta_regul * tf.add_n(tf.get_collection('losses')))
    loss = tf.reduce_mean(cross_entropy)
    variable_averages = tf.train.ExponentialMovingAverage(0.99, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    w, p, d, t = get_input("feature")
    dataset_size = len(t)

    learning_rate = tf.train.exponential_decay(
        lr,
        global_step,
        dataset_size/batch_size,
        learning_rate_decay

    )

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    # train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver()
    min_loss = 100
    F = open("./Model/train_result.txt", "w+")
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        for i in range(training_steps):
            start = (i * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={w_tensor: w[start:end],
                                                      p_tensor: p[start:end],
                                                      d_tensor: d[start:end],
                                                      y_: t[start:end]})
            if i % 1000 == 0:
                print("\nafter %d training steps, loss on training batch is %g." % (i, loss_value))
                F.write(str([i, loss_value]) + "\n")
                F.close()
                F = open("./Model/train_result.txt", "a")
                f = open("./Model/model.ckpt", "w+")
                f.close()
                saver.save(
                    # sess, os.path.join("./Model/", "model.ckpt"),
                    sess, "./Model/model.ckpt",
                    global_step=i
                )
    F.close()


if __name__ == '__main__':
    # 18, 18, 12, 3
    train()