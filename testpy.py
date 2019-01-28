import tensorflow as tf
from mobilenetv1.base_func import *


if __name__ == "__main__":
    g = tf.get_default_graph()
    sess = tf.Session()
    load_model('pretrained/mobilenetv1_1.0.pb')
    g.get_operations()