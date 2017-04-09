import numpy as np

class GenerateRedRects():
    def __init__(self, size=[32,32,3], batch_size=10):
        self.size = size
        self.batch_size = batch_size

    def next(self):
        return self.__next__()

    def __next__(self):
        all_x = []
        all_y = []
        all_y_seg = []
        all_y_seg_one_hot = []

        for i in range(0, self.batch_size):
            x, y, y_seg, y_seg_one_hot = self.generate()
            all_x.append(x)
            all_y.append(y)
            all_y_seg.append(y_seg)
            all_y_seg_one_hot.append(y_seg_one_hot)

        return all_x, all_y, all_y_seg, all_y_seg_one_hot

    def generate(self):

        put_in_red_square = np.random.randint(0, 2)
        # print "putinsquare:", put_in_red_square
        # x = np.random.random_sample(self.size)
        x = np.zeros(self.size)

        y_seg = np.zeros([self.size[0], self.size[1]])
        y_seg_classes_0 = np.zeros([self.size[0], self.size[1], 1])
        y_seg_classes_1 = np.ones([self.size[0], self.size[1], 1])
        y_seg_classes = np.concatenate([y_seg_classes_0, y_seg_classes_1], axis=2)

        y = [0, 1]
        if put_in_red_square > 0:
            l0 = np.random.randint(1, self.size[1])
            # l1 = np.random.randint(0, self.size[0])
            offset0 = np.random.randint(0, - l0 + self.size[0])
            offset1 = np.random.randint(0, - l0 + self.size[1])
            r = [1.0, 0, 0]
            red_square = np.array([[r] * l0] * l0)
            # red_square = np.zeros([l0, l0, 3])
            red_square_seg = np.zeros([l0, l0]).fill(1)
            x[offset0:offset0+l0, offset1: offset1 + l0, :] = red_square
            y_seg[offset0:offset0+l0, offset1: offset1 + l0] = red_square_seg
            y = [1, 0]

            red_square0 = np.zeros([l0, l0, 1])
            red_square1 = np.ones([l0, l0, 1])
            red_square = np.concatenate([red_square1, red_square0], axis=2)
            y_seg_classes[offset0:offset0 + l0, offset1: offset1 + l0] = red_square

        return x, y, y_seg, y_seg_classes
