import json
import numpy as np
import tensorflow as tf

def get_bbox(str):
    obj = json.loads(str.decode('utf-8'))
    bbox = obj['bounding_box']
    return np.array([bbox['x'], bbox['y'], bbox['height'], bbox['width']], dtype='f')

def get_multiple_bboxes(str):
    return [[get_bbox(x) for x in str]]

raw = tf.placeholder(tf.string, [None])
[parsed] = tf.py_func(get_multiple_bboxes, [raw], [tf.float32])