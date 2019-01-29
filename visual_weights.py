#! /usr/bin/env python2

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import time
import argparse

DIR = "./"
FILE = "model-200"

plt.ion
plt.figure()

def display_weights( W ):
    ( n, x, y, z ) = W.shape
    h = int( np.sqrt( n ) )

    W = W.transpose( 0, 3, 1, 2 )
    X = np.ones( ( n, z, x + 1, y + 1 ) )

    for i in xrange( n ):
        for j in xrange( 3 ):
            X[ i ][ j ][ :x, :y ] = W[ i ][ j ]

    X = X.transpose( 0, 2, 3, 1 )
        
    image = np.vstack( [ np.hstack( [ ( X[ i * h + j ] ) * 255 
                                        for j in xrange( h ) ] ) 
                                            for i in xrange( h ) ] )
    plt.imshow( image )


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description="Display Weights" )
    parser.add_argument( 'weight', type=str )
    args = parser.parse_args()
    print args

    tf.reset_default_graph()

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph( os.path.join( DIR, "model_ckpt.meta" ) )

        for _ in xrange( 1000 ):
            saver.restore( sess, tf.train.latest_checkpoint( DIR ) )
            w = sess.run( "W1_1:0" )
            w = w.transpose( 3, 0, 1, 2 )
            display_weights( w )
            plt.pause( 10 )


