import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset_utils import GenerateRedRects
from net import SimpleSegmentationNetwork

import yaml

t_start = datetime.datetime.now()

with open("params/params.yaml", 'r') as stream:
    try:
        cfg = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

img_size_x = 16
img_size_y = 16
batch_size = 10
max_iter = 10000
dg = GenerateRedRects(size=[img_size_x,img_size_y,3], batch_size=batch_size)
layer_info=[{'name': 'c1', 'height': img_size_x, 'width': img_size_y, 'input': 3, 'output': 64},
            {'name': 'c2', 'height': img_size_x, 'width': img_size_y, 'input': 64, 'output': 128}]
ssn = SimpleSegmentationNetwork(layer_info=layer_info)

# x, y, y_seg = dg.next()

# for i in range(0, batch_size):
#     plt.imshow(x[i])
#     plt.imshow(y_seg[i])
#     plt.show()

init_op = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init_op)

# convenience
train_step = ssn.train_step
accuracy = ssn.accuracy
x = ssn.x
y_ = ssn.y_
output = ssn.output
y_ = ssn.y_
loss = ssn.loss
netout = ssn.raw_scores_per_pxl

accuracy, loss, netout

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
nn = tf.split(netout, 2, axis=3)
tf.summary.image('netout', nn[0])
tf.summary.image('netout', nn[1])

tf.summary.image('input', x)
# tf.summary.image('gt', y_)
graph = tf.get_default_graph()
init_op = tf.global_variables_initializer()
sess.run(init_op)

summary_writer = tf.summary.FileWriter("./summary/" + str(datetime.datetime.now()) + "/", sess.graph)
run_metadata = tf.RunMetadata()

merged = tf.summary.merge_all()

saver = tf.train.Saver()

for i in range(0, max_iter):
    xx, yy, yy_seg, yy_seg_classes = dg.next()
    _, summary, accuracy_, loss_, netout_ = sess.run([train_step, merged, accuracy, loss, netout], feed_dict={x: xx, y_: yy_seg_classes})
    print i, "accuracy", accuracy_, "loss", loss_
    summary_writer.add_summary(summary, global_step=i)

    if i % cfg['SAVE_INTERVAL'] == 0:
        t_start_aux = datetime.datetime.now()
        save_path = saver.save(sess, cfg['CHECKPOINT_DIR'] + cfg['MODEL_NAME'], global_step=i)
        print "Model saved in file:", save_path, "done in", datetime.datetime.now() - t_start_aux
    # if i % 5 == 0:
    #     NUM_FIGS_X = 2
    #     NUM_FIGS_Y = 2
    #
    #     fig = plt.figure()
    #     a = fig.add_subplot(NUM_FIGS_X, NUM_FIGS_Y, 1)
    #     plt.imshow(xx[0][:, :, 0])
    #     a = fig.add_subplot(NUM_FIGS_X, NUM_FIGS_Y, 2)
    #     plt.imshow(xx[0][:, :, 1])
    #     a = fig.add_subplot(NUM_FIGS_X, NUM_FIGS_Y, 3)
    #     plt.imshow(netout_[0][:, :, 0])
    #     a = fig.add_subplot(NUM_FIGS_X, NUM_FIGS_Y, 4)
    #     plt.imshow(netout_[0][:, :, 1])
    #
    #     plt.show()