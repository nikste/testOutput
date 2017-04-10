import tensorflow as tf
from net_utils import convolutional, weight_variable


class SimpleSegmentationNetwork:

    def load_from_checkpoint(self, sess, checkpoint_dir):
        file_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
        new_saver = tf.train.import_meta_graph(file_path + '.meta')
        new_saver.restore(sess, file_path)

        sess.run(tf.all_variables())
        x = tf.get_collection('x')[0]
        y_ = tf.get_collection('y_')[0]
        raw_scores_per_pxl = tf.get_collection('raw_scores_per_pxl')[0]
        output = tf.get_collection('output')[0]
        accuracy = tf.get_collection('accuracy')[0]
        global_step = tf.get_collection('global_step')[0]
        train_step = tf.get_collection('train_step')[0]
        cross_entropy = tf.get_collection('cross_entropy')[0]
        summary_op = tf.get_collection('summary_op')[0]

        return x, y_, raw_scores_per_pxl, output, accuracy, global_step, train_step, cross_entropy, summary_op

    def __init__(self, sess, checkpoint_dir=None, load_from_checkpoint=False, num_classes=2,
                 layer_info=[{'name': 'c1', 'height': 32, 'width': 32, 'input': 3, 'output': 64}]):

        if load_from_checkpoint:
            self.x, self.y_, self.raw_scores_per_pxl, self.output, \
                self.accuracy, self.global_step, self.train_step, self.cross_entropy,\
                self.summary_op = self.load_from_checkpoint(sess, checkpoint_dir=checkpoint_dir)
        else:
            self.x = tf.placeholder(tf.float32)
            self.y_ = tf.placeholder(tf.float32)

            self.layer_info = layer_info
            self.layer = [self.x]
            for i in range(0, len(layer_info)):
                print i, 'len(self.layer)', len(self.layer), 'output', layer_info[i], 'input', self.layer[i - 1]
                self.layer.append(self.create_hidden(layer_info[i], self.layer[i]))

            # self.output = self.layer[-1]
            self.raw_scores_per_pxl, self.output, \
                self.accuracy, self.global_step, self.train_step, self.cross_entropy = self.create_output(num_classes)

            self.summary_op = self.create_summaries()
            self.create_collections()

            init_op = tf.initialize_all_variables()
            sess.run(init_op)

    def create_hidden(self, layer_info, layer_input):
        return convolutional(layer_input, self.layerinfo2filter(layer_info), name=layer_info['name'])

    def layerinfo2filter(self, layer_info):
        return [layer_info['height'], layer_info['width'], layer_info['input'], layer_info['output']]

    def create_output(self, num_classes):
        out_conv = self.layer[-1]
        raw_scores_per_pxl = convolutional(out_conv, [self.layer_info[-1]['height'], self.layer_info[-1]['width'],
                                                      self.layer_info[-1]['output'], num_classes], name='raw_scores')
        raw_scores_per_pxl = raw_scores_per_pxl
        raw_scores_per_pxl_flat = tf.reshape(raw_scores_per_pxl, [-1, num_classes])
        y_flat = tf.reshape(self.y_, [-1, num_classes])

        # performance measures
        output = tf.argmax(tf.nn.softmax(raw_scores_per_pxl), axis=-1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(input=y_flat, axis=-1), tf.argmax(input=raw_scores_per_pxl_flat, axis=-1)), tf.float32))

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_flat, logits=raw_scores_per_pxl_flat),
                                       name='cross_entropy')

        # Train op
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=global_step)

        return raw_scores_per_pxl, output, accuracy, global_step, train_step, cross_entropy

    def create_summaries(self):
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        tf.summary.scalar('accuracy', self.accuracy)
        raw_scores_per_pxl_split = tf.split(self.raw_scores_per_pxl, 2, axis=3)
        tf.summary.image('raw_scores_per_pxl0', raw_scores_per_pxl_split[0])
        tf.summary.image('raw_scores_per_pxl1', raw_scores_per_pxl_split[1])
        tf.summary.image('output', tf.expand_dims(tf.cast(self.output * 255, tf.uint8), axis=-1))
        tf.summary.image('input', self.x)
        run_metadata = tf.RunMetadata()
        summary_op = tf.summary.merge_all()
        return summary_op

    def create_collections(self):
        tf.add_to_collection('x', self.x)
        tf.add_to_collection('y_', self.y_)
        tf.add_to_collection('raw_scores_per_pxl', self.raw_scores_per_pxl)
        tf.add_to_collection('output', self.output)
        tf.add_to_collection('accuracy', self.accuracy)
        tf.add_to_collection('global_step', self.global_step)
        tf.add_to_collection('train_step', self.train_step)
        tf.add_to_collection('cross_entropy', self.cross_entropy)
        tf.add_to_collection('summary_op', self.summary_op)

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
