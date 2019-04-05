import tensorflow as tf
import numpy as np

a=np.random.rand(10)
print(a)

# Tensor flow version check
print('tf version=', tf.__version__)

# Tensor test
a = tf.constant('hello')
sess = tf.Session()
val = sess.run(a)
print('val=', val)
sess.close()

# Tensor test a+b
a = tf.constant(3, tf.float32)
b = tf.constant(4, tf.float32)
c = tf.add(a, b)
sess = tf.Session()
val = sess.run(c)
print(sess.run([a,b]))
print('a+b=', val)
sess.close()

