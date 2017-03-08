import numpy as np
from PIL import Image


class cnn:
    """
    CNN layer parameter 
    
    """
    def __init__(self,num_w_fltr, num_b_fltr, weight, bias):
        self.num_w_fltr = num_w_fltr
        self.num_b_fltr = num_b_fltr
        self.weight = weight
        self.bias = bias


class nn_img:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.batch_idx = 0

    def next_batch(self, batch_size):
        self.batch_idx = self.batch_idx + batch_size
        return (self.images[self.batch_idx - batch_size : self.batch_idx ], self.labels[self.batch_idx - batch_size : self.batch_idx ])

class imgData:
    def __init__(self, test_h5, train_h5):
        self.test = nn_img(test_h5['data'], test_h5['label'])
        self.train = nn_img(train_h5['data'], train_h5['label'])

def setCNNParameter():
    num_layer = 20
    NNlayer = []
    for i in range(num_layer):
        if i ==0:
            layer = cnn((1,64),(64,1), None, None)
        elif i==20:
            #(c,f,b,1)
            layer = cnn((64,1),(1,1), None, None)
        else:
            layer = cnn((64,64),(64,1), None, None)
        NNlayer.append(layer)

    return NNlayer


def genCNNParameter(dat):
    NNlayer = []
    for i in range(len(dat)):
        dat2 = dat[i].split(', ')
        npdat2 = np.array(dat2[:-1], np.float16)

        #fn1: number of channel
        #fn2: number of filter
        #
        head = 4
        fw, fh, fn1, fn2 = npdat2[:head].astype("int32")

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

        #print i, 'weigh: ', weight.shape
        #weight = np.reshape(weight, dsize)

        layer = cnn((fn1, fn2), (bias1, bias2), weight, bias)

        NNlayer.append(layer)

    return NNlayer


def getYCbCr(fn):
    im = Image.open(fn)
    im = im.convert('YCbCr')
    imYCbCr= np.array(im)
    imY= imYCbCr[:, :, 0]  # to make a separate copy as array is immutable
    imCb= imYCbCr[:, :, 1]  # to make a separate copy as array is immutable
    imCr= imYCbCr[:, :, 2]  # to make a separate copy as array is immutable


    return imY, imCb, imCr


def getBlur(img):
    dim = img.shape
    half_dim = dim[0]//2, dim[1]//2
    im = Image.fromarray(img, mode='L')
    im = im.resize(half_dim, Image.BICUBIC)
    im = im.resize(dim, Image.BICUBIC)

    return np.array(im)


def normalize(fn):
    Y, Cb, Cr = getYCbCr(fn)
    blurImg = getBlur(Y)

    return blurImg/255.


def testModel(NNlayer, imY, nLayer=20):
    dim=imY.shape
    features = tf.placeholder('half', [dim[0], dim[1]], name='Input')
    input_layer = tf.reshape(features, [1, dim[0], dim[1] , 1])
    rawInput = tf.reshape(features, [1, dim[0], dim[1], 1])

    w = dict()
    b = dict()

    for i, layer in enumerate(NNlayer[:nLayer]):
        nFilter = int(layer.num_w_fltr[1])
        nChannel = int(layer.num_w_fltr[0])
        nBias = int(layer.num_b_fltr[0])

        w[i] = tf.placeholder('half', [3, 3, nChannel, nFilter])
        b[i] = tf.placeholder('half', [nBias])

        x = tf.nn.conv2d(input_layer, filter=w[i], strides=[1, 1, 1, 1], padding='SAME', name="Conv" + str(i))
        conv = tf.nn.bias_add(x, b[i])

        if i < (len(NNlayer) - 1):
            conv = tf.nn.relu(conv)
        input_layer = conv

    result = conv + rawInput

    weight ={}
    bias = {}
    for i, layer in enumerate(NNlayer[:nLayer]):
        weight[i] = layer.weight.reshape(3,3,int(layer.num_w_fltr[0]), int(layer.num_w_fltr[1]))
        bias[i] = layer.bias.reshape(int(layer.num_b_fltr[0]))

    sess=tf.Session()
    tf.global_variables_initializer()

    fd = {}
    fd[features] = imY
    for i in range(len(NNlayer[:nLayer])):
        fd[w[i]] = weight[i]
        fd[b[i]] = bias[i]
    res = sess.run(result,feed_dict=fd)
    return res


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def trainModel(NNlayer, img_data, img_shape):
    """
    :param NNlayer: data structure for convolutional neural network
    :param img_data: image structure for training and test
    :param img_shape: dimension of image (height, width)
    :return: accuracy of training result
    """
    nLayer = len(NNlayer) #number of layer

    # training image의 해상도가 모두 다른데 어떻게?
    # 3가지의 scale에 대해 training하는 것 같은데 2,3,4
    dim=img_shape
    features = tf.placeholder('half', [dim[0], dim[1]], name='Input')
    input_layer = tf.reshape(features, [-1, dim[0], dim[1] , 1])
    rawInput = tf.reshape(features, [-1, dim[0], dim[1], 1])

    ref_img = tf.placeholder('half', [dim[0], dim[1]], name='result')
    y_ = tf.reshape(ref_img, [-1, dim[0], dim[1] , 1])

    w = dict()
    b = dict()

    for i, layer in enumerate(NNlayer[:nLayer]):
        nChannel = int(layer.num_w_fltr[0])
        nFilter = int(layer.num_w_fltr[1])
        nBias = int(layer.num_b_fltr[0])

        w[i] = weight_variable([3, 3, nChannel, nFilter])
        b[i] = bias_variable([nBias])

        x = tf.nn.conv2d(input_layer, filter=w[i], strides=[1, 1, 1, 1], padding='SAME', name="Conv" + str(i))
        conv = tf.nn.bias_add(x, b[i])

        if i < (len(NNlayer) - 1):
            conv = tf.nn.relu(conv)
        input_layer = conv

    result = conv + rawInput

    mse = -tf.reduce_sum(tf.square(tf.sub(y_- result)))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(mse)
    #correct_prediction = tf.equal(tf.argmax(result, 1), tf.argmax(y_, 1))
    accuracy= 10.*math.log10(255*255/mse)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        batch = img_data.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={features: batch[0], ref_img: batch[1]})
            print("step %d, training PSNR%g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], ref_img: batch[1] })

    res = sess.run(accuracy, feed_dict={x: img_data.test.images, ref_img: img_data.test.labels})


    return res


def showResult(resImg):
    img = resImg.astype('uint8')
    img = Image.fromarray(img, mode='YCbCr')
    img.show("Result")


def calcPSNR(img1, img2):

    mse= np.sum((img1.astype('float') - img2.astype('float'))**2)
    mse /= float(img1.shape[0] * img1.shape[1])

    PSNR = 10.*math.log10(255*255/mse)

    return PSNR

numLayer = 20

def test_main():

    fn = 'model.txt'
    f = open(fn, 'r')
    dat = f.readlines()

    NN= genCNNParameter(dat)

    img_fn = 'baby_GT.bmp'
    im = normalize(img_fn)

    result = testModel(NN, im, numLayer)

    resImg = result[0, :, :, 0]*255
    resImg = np.clip(resImg,0,255)

    Y, Cb, Cr = getYCbCr(img_fn)
    colorImg = np.ndarray((Y.shape[0], Y.shape[1], 3),dtype="uint8")
    colorImg[:,:,0]=resImg
    colorImg[:,:,1]=Cb
    colorImg[:,:,2]=Cr

    print( 'PSNR of NN: ', calcPSNR(Y, resImg), 'dB')
    print( 'PSNR of Bicubic', calcPSNR(Y, im*255), 'dB')

    showResult(colorImg)


def train_main():
    import h5py
    train_h5 = h5py.File("train.h5", "r")
    test_h5  = h5py.File("test.h5", "r")

    #Data preparation
    iData = imgData(test_h5, train_h5)
    imgShape = (41,41)
    nnLayer = setCNNParameter()

    result = trainModel(iData, nnLayer, imgShape)

    print("test accuracy in PSRN %g"%result)

if __name__ == '__main__':
    test_main();