#-*- encoding: utf-8 -*-
import math
import tensorflow as tf
import numpy as np
from PIL import Image
import h5py
import datetime
import sys

class cnn:
    """ CNN layer parameter """

    def __init__(self,num_w_fltr, num_b_fltr, weight, bias):
        self.num_w_fltr = num_w_fltr # (channel, filter)
        self.num_b_fltr = num_b_fltr # (filter, 1)
        self.weight = weight
        self.bias = bias


class nn_img:
    """
    Images for training or testing
    images: imasges for neural network input
    labels: images for reference
    """

    def __init__(self, images, labels):
        """ """
        self._size = len(images)

        #If you have not enough memory, you should not use the following 2 lines  
        #self.images = np.array(images)
        #self.labels = np.array(labels)
        #If you have not enough memory, you should enable the following 2 commented lines  
        self.images = images
        self.labels = labels
        self.batch_idx = 0


    def reset_batch_index(self, rd_idx=0):
        """  """
        self.batch_idx = rd_idx


    def next_batch(self, batch_size):
        """
        function for mini batch training
        """
        if self.batch_idx == self._size:
            self.batch_idx = 0
        prev_idx = self.batch_idx
        #self.batch_idx = min(self._size, self.batch_idx + batch_size)
        self.batch_idx += batch_size 
        if self.batch_idx > self._size:
            prev_idx =0
            self.batch_idx = batch_size

        return (self.images[prev_idx : self.batch_idx ], 
                self.labels[prev_idx : self.batch_idx ])


class imgData:
    """
    Image Data set : Generated by SRCNN, please refer to generate_test.m
    """

    def __init__(self, test_h5, train_h5):
        self.test = nn_img(test_h5['data'], test_h5['label'])
        self.train = nn_img(train_h5['data'], train_h5['label'])


def setCNNParameter():
    """ Set channel number and filter number of each cnn layer for training"""

    #Number of neural network layer
    num_layer = 20 

    NNlayer = []
    for i in range(num_layer):
        if i ==0:
            layer = cnn((1,64),(64,1), None, None)
        elif i== (num_layer -1):
            #(c,f,b,1)
            layer = cnn((64,1),(1,1), None, None)
        else:
            layer = cnn((64,64),(64,1), None, None)
        NNlayer.append(layer)

    return NNlayer


def genCNNParameter(dat):
    """ read CNN parameter from model.txt"""

    NNlayer = []
    for i in range(len(dat)):
        dat2 = dat[i].split(', ')
        
        #remove line feed ('\n') and convert data to numpy array
        npdat2 = np.array(dat2[:-1], np.float16) 

        #fn1: number of channel, #fn2: number of filter, #fw, fh: filter width, height
        head = 4 
        fw, fh, fn1, fn2 = npdat2[:head].astype("int32")
        
        # Last line that contains number of bias filter of last layer
        if i == (len(dat) - 1):
            head = 3 
            fw, fh, fn1 = npdat2[:head].astype("int32")
            fn2 = 1

        dsize = fw * fh * fn1 * fn2
        weight = npdat2[head:dsize + head]
        bias1, bias2 = dat2[head + dsize:dsize + head + 2]
        bias = npdat2[dsize + 2 + head:]

        weight =np.array(weight).reshape(fn1,fn2,fh,fw)
        weight = np.swapaxes(weight, 0,3)
        weight = np.swapaxes(weight, 1,2)

        #Set up network parameter for specific cnn layer
        layer = cnn((fn1, fn2), (bias1, bias2), weight, bias)
        NNlayer.append(layer)

    return NNlayer


def getYCbCr(file_name):
    """ read image file (file_name) and return numpy array of Y, Cb, Cr """

    im = Image.open(file_name)
    im = im.convert('YCbCr')
    imYCbCr= np.array(im)
    imY = imYCbCr[:, :, 0]
    imCb= imYCbCr[:, :, 1] 
    imCr= imYCbCr[:, :, 2] 

    return imY, imCb, imCr


def getBlur(img):
    """ 
    img: 1 channel(color) image whose type is numpy array 
    return blured image using bicubic operation
    """

    half_dim = img.shape[0]//2, img.shape[1]//2
    im = Image.fromarray(img, mode='L')
    im = im.resize(half_dim, Image.BICUBIC)
    im = im.resize(img.shape, Image.BICUBIC)

    return np.array(im)


def blurAndNormalize(file_name):
    """ blur and then normalize Y(0~255) pixel value 0~1 """

    Y, Cb, Cr = getYCbCr(file_name)
    blurImg = getBlur(Y)

    return blurImg/255.


def toColorImage(Y, Cb, Cr):
    """ Y, Cb, Cr: numpy arrary 
        colorImg: 8 bits color numpy array """

    colorImg = np.ndarray((Y.shape[0], Y.shape[1], 3),dtype="uint8")
    colorImg[:,:,0]= Y
    colorImg[:,:,1]= Cb
    colorImg[:,:,2]= Cr

    return colorImg


def testModel(NNlayer, imY):
    """ 
    function to test VDSR nn using pre-trained model parameter in model.txt 
    which is obtained from VDSR github
    """

    nLayer = len(NNlayer)

    features = tf.placeholder('float', [None, imY.shape[0], imY.shape[1]], name='Input')
    input_layer = tf.reshape(features, [-1, imY.shape[0], imY.shape[1] , 1])
    rawInput = tf.reshape(features, [-1, imY.shape[0], imY.shape[1], 1])

    w = dict()
    b = dict()

    for i, layer in enumerate(NNlayer):
        nFilter = int(layer.num_w_fltr[1])
        nChannel = int(layer.num_w_fltr[0])
        nBias = int(layer.num_b_fltr[0])

        w[i] = tf.placeholder('float', [3, 3, nChannel, nFilter])
        b[i] = tf.placeholder('float', [nBias])

        x = tf.nn.conv2d(input_layer, filter=w[i], strides=[1, 1, 1, 1], padding='SAME', name="Conv" + str(i))
        conv = tf.nn.bias_add(x, b[i])

        if i < (nLayer - 1):
            conv = tf.nn.relu(conv)
        input_layer = conv

    result = conv + rawInput

    weight ={}
    bias = {}
    for i, layer in enumerate(NNlayer):
        weight[i] = layer.weight.reshape(3,3,int(layer.num_w_fltr[0]), int(layer.num_w_fltr[1]))
        bias[i] = layer.bias.reshape(int(layer.num_b_fltr[0]))

    sess=tf.Session()
    tf.global_variables_initializer()

    fd = {}
    fd[features] = [imY]
    for i in range(nLayer):
        fd[w[i]] = weight[i]
        fd[b[i]] = bias[i]

    res = sess.run(result,feed_dict=fd)

    return res


def Xavier_init(n_inputs, n_outputs, uniform=True):

    if uniform:
        # 6 was used in the paper.
        init_range = tf.sqrt(4.0*6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = tf.sqrt(4.0*3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def He_init(n_inputs, n_outputs, uniform=True):
    """ xavier with 2x stddev"""

    if uniform:
        init_range = tf.sqrt(4.0*6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(4.0*3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

def xaver_init(shape, uniform=True):
    n_inputs = 1
    for i in shape:
        n_inputs *= i

    n_outputs = shape[-1]

    return Xavier_init(n_inputs, n_outputs, uniform)


def he_init(shape, uniform=True):
    n_inputs = 1
    for i in shape:
        n_inputs *= i

    n_outputs = shape[-1]

    return He_init(n_inputs, n_outputs, uniform)


def weight_variable(w_idx, shape):
    """ Initialize neural network weights(Tensorflow variable) """
    #initial = tf.truncated_normal(shape, stddev=0.1)
    #return tf.Variable(initial)
    return tf.get_variable('w'+str(w_idx), shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    #return tf.get_variable('w'+str(w_idx), shape=shape, initializer=he_init(shape))

def bias_variable(b_idx, shape):
    """ Initialize neural network bias (Tensorflow variable) """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    #return tf.get_variable('b'+ str(b_idx), shape=shape, initializer=he_init(shape))

def restoreModel(session):
    """ restore Graph and model parameters"""
    """
    try:
        new_saver = tf.train.import_meta_graph('model.ckpt.meta')
    except IOError:
        new_saver = None

    if new_saver is not None:
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    """
    saver = tf.train.Saver()

    try:
        saver.restore(session, './model.ckpt')

    except:
        pass
    
    return saver 


def initTraining(img_data):
    """Initialize training parameters """

    nEpoch = 1
    batch_size = 100 
    #training_data_size = len(img_data.train.images)
    training_data_size = 10000
    ##Number of step per Epoch = 24930, (batch_size = 50, training_data_size = len(img_data.train.images)

    try:
        iterFile = open('iter.txt', 'r')
        rd_idxs = iterFile.readlines()
        iterFile .close()

        if len(rd_idxs) == 0: rd_idx = 0
        else: rd_idx = int(rd_idxs[-1])

        if rd_idx >= training_data_size: rd_idx = 0

    except IOError:
        rd_idx = 0

    print ('Number of epoch: %d'%nEpoch)
    print ('Batch size: %d'%batch_size)
    print ('Training data size: %d'%training_data_size)
    print ('rd_idx: %d'%rd_idx)

    logFile = open('psnr.log', 'a')
    iterFile  = open('iter.txt', 'w')

    return nEpoch, batch_size, training_data_size, rd_idx, logFile, iterFile 


def genW_B(NNlayer, img_shape ):
    """ Initialize cnn weight and bias tensor Variables.
        These variables are shared among the GPUs. 
    """
    features = tf.placeholder('float', [None, 1, img_shape[0], img_shape[1]], name='TrInput')
    ref_img = tf.placeholder('float', [None, 1, img_shape[0], img_shape[1]], name='result')

    w = dict()
    b = dict()

    nLayer = len(NNlayer) #number of layer

    for i, layer in enumerate(NNlayer[:nLayer]):
        nChannel = int(layer.num_w_fltr[0])
        nFilter = int(layer.num_w_fltr[1])
        nBias = int(layer.num_b_fltr[0])
        
        w[i] = weight_variable(i, [3, 3, nChannel, nFilter])
        b[i] = bias_variable(i, [nBias])
        
    return features, ref_img, w, b


def Model(NNlayer, img_shape, features, ref_img, w, b ):
    """ VDSR CNN model """
    
    nLayer = len(NNlayer) #number of layer

    input_layer = tf.reshape(features, [ -1, img_shape[0], img_shape[1] , 1])
    rawInput = tf.reshape(features, [ -1, img_shape[0], img_shape[1], 1])
    y_ = tf.reshape(ref_img, [ -1, img_shape[0], img_shape[1] , 1])

    for i, layer in enumerate(NNlayer):

        x = tf.nn.conv2d(input_layer, filter=w[i], 
             strides=[1, 1, 1, 1], padding='SAME', name="Conv" + str(i))
        conv = tf.nn.bias_add(x, b[i])
         
        if i < (nLayer - 1):
            conv = tf.nn.relu(conv)
        input_layer = conv
     
    result = conv + rawInput

    SSE_res = tf.reduce_mean(tf.square(tf.subtract( y_,  result)))

    return result, SSE_res 


def report_psnr(mse, logFile, iterFile, niter, rd_idx, isEndEpoch=False ):
    """ Print and log psnr per each training step or epoch """

    PSNR = psnr(mse) 

    logLn = None
    if isEndEpoch:
        logLn = "Epoch %d, Test PSNR: %g" % (niter, PSNR)
    else:
        logLn = "step %d, training PSNR: %g" % (niter, PSNR) 

    logFile.write(logLn+'\n'); logFile.flush() 
    print(logLn) 
    
    iterFile.write(str(rd_idx)+'\n') 
    iterFile.flush() 

    return PSNR


def average_gradients(tower_grads):
    """Averaging gradient """

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads   


def trainModel(NNlayer, img_data, img_shape, gpu_list, isTest=False):
    """
    :param NNlayer: data structure for convolutional neural network
    :param img_data: image structure for training and test
    :param img_shape: dimension of image (height, width)
    :param isTest: if True: train, else reconstruct img_data
    :return: if isTest: reconstructed image, else PSNR of training result
    """

    with tf.device('/cpu:0'): 

        num_gpu = len(gpu_list) 

        features, ref_img, w, b = genW_B(NNlayer, img_shape)

        global_step = tf.Variable(0, trainable=False)
        #starter_learning_rate = 0.0001
        #learning_rate = tf.train.exponential_decay(starter_learning_rate, 
        #                    global_step, 1000, 0.96, staircase=True)
        #learning_rate = 0.001
        #opt = tf.train.GradientDescentOptimizer(learning_rate)

        opt = tf.train.AdamOptimizer(0.01) # Learning rate = 0.01

        features_lst= tf.split(features, num_gpu)
        ref_img_lst= tf.split(ref_img, num_gpu)

        SSE_ress= []
        grads = []
        for i, gpu in enumerate(gpu_list):
            with tf.device(gpu):
                result, SSE= Model(NNlayer, img_shape, features_lst[i], ref_img_lst[i], w, b)
                SSE_ress.append(SSE)
                grad = opt.compute_gradients(SSE)
                grads.append(grad)

        SSE_res = tf.reduce_mean(SSE_ress)

        av_grad=average_gradients(grads) 
        train_step= opt.apply_gradients(av_grad, global_step=global_step )


        #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(SSE_res, global_step=global_step) #1e-3

        
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())

        #Load variables saved
        saver = restoreModel(sess)

        if isTest:
            return sess.run(result, feed_dict={features: [img_data.test.images[0:1]], 
                                       ref_img: [img_data.test.images[0:1]] })

        nEpoch, batch_size, training_data_size, rd_idx, logFile, iterFile = initTraining(img_data)
        PSNR = 0.

        start_time = datetime.datetime.now()
        print('Start Time: %s'%str(start_time))

        for niter in range(nEpoch):
            if niter >0: rd_idx = 0

            img_data.train.reset_batch_index(rd_idx)

            i=0
            while( img_data.train.batch_idx <= training_data_size):
                batch = img_data.train.next_batch(batch_size)

                if i % 50 == 0:
                    mse = sess.run(SSE_res, feed_dict={features: batch[0], ref_img: batch[1]})
                    rd_idx = img_data.train.batch_idx 
                    report_psnr(mse, logFile, iterFile, i, rd_idx)

                save_path = saver.save(sess, './model.ckpt')
                sess.run(train_step, feed_dict={features: batch[0], ref_img: batch[1] })
                i += 1

            mse = sess.run(SSE_res, feed_dict={features: img_data.test.images[:100], 
                            ref_img: img_data.test.labels[:100]})
            PSNR = report_psnr(mse, logFile, iterFile, niter, rd_idx, isEndEpoch=True)
            save_path = saver.save(sess, './model.ckpt')
        
        rd_idx =0
        iterFile.write(str(rd_idx))
        iterFile.close()
        logFile.close()

        end_time =  datetime.datetime.now()
        spent_time = end_time - start_time
        print('Processing time: %s'%str(spent_time))

    return PSNR


def showResult(resImg, fn=None):
    """ Display YCbCr image whose type is numpy array """
    img = resImg.astype('uint8')
    img = Image.fromarray(img, mode='YCbCr')
    img.show("Result")
    if fn is not None:
        img = img.convert('RGB')
        img.save(fn)


def psnr(mse):
    """ input: mean square error return: PSNR 
        Peak Signal value is 1.
    """
    
    #if mse == 0.0 return np.nan
    return 10.*math.log10(1./mse)


def calcPSNR(img1, img2):
    """ calc psnr of (img1-img2)"""
    sse= np.sum((img1.astype('float') - img2.astype('float'))**2)
    mse = sse/float(img1.shape[0] * img1.shape[1])

    return 10.*math.log10(255.*255./mse)


def test_main():
    """ Show the result of VDSR paper """

    fn = 'model.txt'
    f = open(fn, 'r')
    dat = f.readlines()

    NN= genCNNParameter(dat)
    numLayer = len(NN) 

    img_fn = 'baby_GT.bmp'
    im = blurAndNormalize(img_fn)

    result = testModel(NN, im)

    resImg = result[0, :, :, 0]*255
    resImg = np.clip(resImg,0,255)

    Y, Cb, Cr = getYCbCr(img_fn)
    colorImg = np.ndarray((Y.shape[0], Y.shape[1], 3),dtype="uint8")
    colorImg[:,:,0]=resImg
    colorImg[:,:,1]=Cb
    colorImg[:,:,2]=Cr

    print( 'PSNR of NN: ',    calcPSNR(Y, resImg), 'dB')
    print( 'PSNR of Bicubic', calcPSNR(Y, im*255), 'dB')

    showResult(colorImg)


def train_main(gpu_list, im=None ):

    nnLayer = setCNNParameter()

    if im is not None:
        pseudoH5 = dict() 
        pseudoH5['data'] = [im]
        pseudoH5['label'] = [im]

        iData = imgData(pseudoH5, pseudoH5)
        imgShape = im.shape
        
        return trainModel(nnLayer, iData, imgShape, gpu_list, isTest=True)

    train_h5 = h5py.File("train.h5", "r")
    test_h5  = h5py.File("test.h5", "r")

    #Data preparation
    iData = imgData(test_h5, train_h5)
    imgShape = (41,41)
        
    psnr = trainModel(nnLayer, iData, imgShape, gpu_list, isTest=False)


def inference(gpu_list):
    """ Prediction """

    img_fn = sys.argv[2]
    Y, Cb, Cr = getYCbCr(img_fn)
    blurImg = getBlur(Y)

    result = train_main(gpu_list, blurImg/255.)

    resImg = result[0, :, :, 0]*255
    resImg = np.clip(resImg,0,255)


    print( 'PSNR of NN: ',      calcPSNR(Y, resImg), 'dB')
    print( 'PSNR of Bicubic: ', calcPSNR(Y, blurImg), 'dB')

    rec_colorImg = toColorImage(resImg, Cb, Cr)
    showResult(rec_colorImg, 'result.bmp')

    bicubic_colorImg = toColorImage(blurImg, Cb, Cr)
    showResult(bicubic_colorImg, 'bicubic.bmp')
    

if __name__ == '__main__':

    if len(sys.argv) <2:
        print('Wrong parameter')
        print('Ex) For test, Type python test.py test')
        print('Ex) For test, Type python test.py t2 image.bmp')
        print('Ex) For train, Type python test.py train [gpu num_gpu]\n, default is cpu')
    
    if sys.argv[1] == 'test':
        test_main()
        
    elif sys.argv[1] == 'train':
        if len(sys.argv) == 4:
            num_gpu = int(sys.argv[3])
            gpu_list = [('/gpu:'+str(i)) for i in range(num_gpu)]
        else:
            gpu_list = ['/cpu:0']

        train_main(gpu_list)

    elif sys.argv[1] == 't2':
        if len(sys.argv) <3:
            print('Wrong parameter')
            print('Ex) For test, Type python test.py t2 image_file_name')
            sys.exit()
        
        inference(['cpu:0'])

    else:
        print('Wrong parameter')
        print('Ex) For test,\n Type python test.py test') 
        print('Ex) For train,\n Type python test.py test') 
        

