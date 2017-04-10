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
batch_size = 1000
max_iter = 10000
dg = GenerateRedRects(size=[img_size_x,img_size_y, 3], batch_size=batch_size)
layer_info = [{'name': 'c1', 'height': img_size_x, 'width': img_size_y, 'input': 3, 'output': 64},
            {'name': 'c2', 'height': img_size_x, 'width': img_size_y, 'input': 64, 'output': 128}]

print "cfg['CHECKPOINT_DIR']", cfg['CHECKPOINT_DIR']

sess = tf.Session()
ssn = SimpleSegmentationNetwork(sess, checkpoint_dir=cfg['CHECKPOINT_DIR'], load_from_checkpoint=cfg['LOAD_FROM_CHECKPOINT'], num_classes=2, layer_info=layer_info)

# convenience
train_step = ssn.train_step
accuracy = ssn.accuracy
x = ssn.x
y_ = ssn.y_
output = ssn.output
y_ = ssn.y_
loss = ssn.cross_entropy
netout = ssn.raw_scores_per_pxl
summary_op = ssn.summary_op
summary_writer = tf.summary.FileWriter("./summary/" + str(datetime.datetime.now()) + "/", sess.graph)

global_step = ssn.global_step

saver = tf.train.Saver()
global_step_ = sess.run(global_step)
for i in range(global_step_, max_iter):
    xx, yy, yy_seg, yy_seg_classes = dg.next()
    _, summary, accuracy_, loss_, netout_, global_step_ = sess.run([train_step, summary_op, accuracy, loss, netout, global_step], feed_dict={x: xx, y_: yy_seg_classes})
    print i, "global_step", global_step_, "accuracy", accuracy_, "loss", loss_
    summary_writer.add_summary(summary, global_step=global_step_)

    if i % cfg['SAVE_INTERVAL'] == 0 and cfg['SAVE_INTERVAL'] > 0:
        t_start_aux = datetime.datetime.now()
        save_path = saver.save(sess, cfg['CHECKPOINT_DIR'] + cfg['MODEL_NAME'], global_step=i)
        print "Model saved in file:", save_path, "done in", datetime.datetime.now() - t_start_aux
    if i % cfg['PLOT_RESULTS'] == 0 and cfg['PLOT_RESULTS'] > 0:
        NUM_FIGS_X = 2
        NUM_FIGS_Y = 2

        fig = plt.figure()
        a = fig.add_subplot(NUM_FIGS_X, NUM_FIGS_Y, 1)
        plt.imshow(xx[0][:, :, 0])
        a = fig.add_subplot(NUM_FIGS_X, NUM_FIGS_Y, 2)
        plt.imshow(xx[0][:, :, 1])
        a = fig.add_subplot(NUM_FIGS_X, NUM_FIGS_Y, 3)
        plt.imshow(netout_[0][:, :, 0])
        a = fig.add_subplot(NUM_FIGS_X, NUM_FIGS_Y, 4)
        plt.imshow(netout_[0][:, :, 1])

        plt.show()