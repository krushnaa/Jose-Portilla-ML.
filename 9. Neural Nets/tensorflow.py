# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 22:13:17 2017

@author: Stefan Draghici
"""

import tensorflow as tf

hello_tf=tf.constant('hello TF')

session=tf.Session()
session.run(hello_tf)

x=tf.constant(5)
y=tf.constant(7)

with tf.Session() as s:
    print("Operation with constants:")
    print(s.run(x+y))
    print(s.run(x-y))
    print(s.run(x/y))
    print(s.run(x*y))
    print(s.run(x**y))