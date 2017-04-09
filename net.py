import tensorflow as tf
from net_utils import convolutional, weight_variable


class SimpleSegmentationNetwork():
    def __init__(self, layer_info=[{'name': 'c1', 'height': 32, 'width': 32, 'input': 3, 'output': 64}], num_classes=2):
        self.x = tf.placeholder(tf.float32)
        self.y_ = tf.placeholder(tf.float32)

        self.layer_info = layer_info
        self.layer = [self.x]
        for i in range(0, len(layer_info)):
            print i, 'len(self.layer)', len(self.layer), 'output', layer_info[i], 'input', self.layer[i - 1]
            self.layer.append(self.create_hidden(layer_info[i], self.layer[i]))

        # self.output = self.layer[-1]
        self.create_output(num_classes)

    def create_hidden(self, layer_info, layer_input):
        return convolutional(layer_input, self.layerinfo2filter(layer_info), name=layer_info['name'])

    def layerinfo2filter(self, layer_info):
        return [layer_info['height'], layer_info['width'], layer_info['input'], layer_info['output']]

    def create_output(self, num_classes):
        out_conv = self.layer[-1]
        raw_scores_per_pxl = convolutional(out_conv, [self.layer_info[-1]['height'], self.layer_info[-1]['width'],
                                                      self.layer_info[-1]['output'], num_classes], name='raw_scores')
        self.raw_scores_per_pxl = raw_scores_per_pxl
        raw_scores_per_pxl_flat = tf.reshape(raw_scores_per_pxl, [-1, num_classes])
        y_flat = tf.reshape(self.y_, [-1, num_classes])
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_flat, logits=raw_scores_per_pxl_flat), name='cross_entropy')
        self.loss = cross_entropy


        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(y_flat, raw_scores_per_pxl_flat), tf.float32))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.output = tf.argmax(tf.nn.softmax(raw_scores_per_pxl), axis=-1)

    def create_output_ws(self, num_classes):
        out_conv = self.layer[-1]
        # reduce dimensions to num_classes
        raw_scores_per_pxl = convolutional(out_conv, [self.layer_info[-1]['height'], self.layer_info[-1]['width'], self.layer_info[-1]['output'], num_classes], name='raw_scores')

        raw_scores_per_image = tf.nn.max_pool(raw_scores_per_pxl,
                                              ksize=[1, self.layer_info[-1]['height'], self.layer_info[-1]['width'], 1],
                                              strides=[1, self.layer_info[-1]['height'], self.layer_info[-1]['width'], 1],
                                              padding='VALID',
                                              name='global_maxpool')
        self.raw_scores_per_pxl = raw_scores_per_pxl
        # reduce to one array for cross entropy and softmax
        # raw_scores_flat = tf.reshape(out_conv, [-1, num_classes])

        # y_flat = tf.reshape(self.y_, [-1, num_classes])
        # comput cross entropy from raw values
        self.mse = tf.reduce_mean(tf.squared_difference(raw_scores_per_image, self.y_), name='ms_error')
        self.loss = self.mse

        self.output = tf.round(raw_scores_per_image)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.output, self.y_), tf.float32), name='accuracy')
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        # output_flat_segment = tf.nn.softmax(raw_scores_flat)

        # self.output = tf.reshape(output_flat_segment, [-1, self.layer_info[-1]['height'], self.layer_info[-1]['width'], num_classes])

        # self.accuracy_segment = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_flat_segment, -1), tf.argmax(output_flat_segment, -1)), tf.float32))
        # self.loss_segment = cross_entropy

        # # reduce to one array for cross entropy and softmax
        # raw_scores_flat = tf.reshape(out_conv, [-1, num_classes])
        #
        # y_flat = tf.reshape(self.y_, [-1, num_classes])
        # # comput cross entropy from raw values
        # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_flat, logits=raw_scores_flat), name='cross_entropy')
        #
        # output_flat_segment = tf.nn.softmax(raw_scores_flat)
        #
        # self.output = tf.reshape(output_flat_segment, [-1, self.layer_info[-1]['height'], self.layer_info[-1]['width'], num_classes])
        #
        # self.accuracy_segment = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_flat_segment, -1), tf.argmax(output_flat_segment, -1)), tf.float32))
        # self.loss_segment = cross_entropy
        # self.train_step_segment = tf.train.AdamOptimizer(1e-4).minimize(self.loss_segment)
