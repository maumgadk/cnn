#-*- encoding: utf-8 -*-
import math
import tensorflow as tf
import numpy as np
from PIL import Image
import h5py

def model():

    x = tf.placeholder('float', name='inputX')#, 1, "inputX")
    y = tf.placeholder('float', name='inputY')#, 1, "inputY")

    bias = tf.constant(1.0, shape=[1])
    b = tf.Variable(bias)

    c = tf.multiply(x,y, "outC")
    z = tf.add(c,b, 'outZ')

    return x, y, z

def genSimpleTF():

    x, y, z = model()

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('z', z)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        res= sess.run(z, feed_dict={x:10., y: 20.})

        save_path = saver.save(sess,'./simpleModel.ckpt')

    return res


def restoreSimpleTF():

    x, y, z = model()

    with tf.Session() as sess:

       sess.run(tf.global_variables_initializer())

       #new_saver = tf.train.import_meta_graph('simpleModel.ckpt.meta')
       #new_saver.restore(sess, tf.train.latest_checkpoint('./'))

       loader = tf.train.Saver()
       loader.restore(sess, './simpleModel.ckpt')

       result = sess.run(z, feed_dict={x:10, y:10})

       return result

def restoreSimpleTF2():

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    new_saver = tf.train.import_meta_graph('simpleModel.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))

    x= tf.get_collection('x')[0]
    y= tf.get_collection('y')[0]
    z= tf.get_collection('z')[0]

    return sess.run(z, feed_dict={x:20, y:30})[0]


if __name__ == '__main__':
    res = genSimpleTF()
    print('x*y + c = 201, %g'%res)

    #res2 = restoreSimpleTF()
    res2 = restoreSimpleTF2()
    print'x*y + c = 601, ', res2






