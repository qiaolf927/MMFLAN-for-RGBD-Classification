import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import urllib
import os
import tarfile
import skimage
import skimage.io
import skimage.transform
from generateRGBDData import *
from generateJHUIT50Data import *
from generateocidData import *
from utils import *
from math import ceil
import random
from imgaug import imgaug as ia
from imgaug import augmenters as iaa

def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1],str(y[i]),
                 color=plt.cm.Set1(d[i] / 6.),
                 fontdict={'weight': 'bold', 'size': 9})
#str(y[i])
    plt.axis('off')
    plt.xticks([]), plt.yticks([])
    # if title is not None:
    #     plt.title(title)
    tsne_path='./'+title+'.png'
    plt.savefig(tsne_path, dpi=600)
    #plt.show()

def data_aug(batch, batch_depth):
    num_img = batch.shape[0]
    list = []
    list_depth = []
    for i in range(num_img):
        val_fliplr = random.randrange(0, 2, 1)  # in questo modo il due non e compreso e restituisce i valori 0 o 1
        list.extend([iaa.Fliplr(val_fliplr)])
        list_depth.extend([iaa.Fliplr(val_fliplr)])

        val_fliplr = random.randrange(0, 2, 1)  # in questo modo il due non e compreso e restituisce i valori 0 o 1
        list.extend([iaa.Flipud(val_fliplr)])
        list_depth.extend([iaa.Flipud(val_fliplr)])

        val_scala = random.randrange(5, 11, 1)
        val = float(val_scala / 10.0)
        list.extend([iaa.Affine(val, mode='edge')])
        list.extend([iaa.Affine(10.0 / val_scala, mode='edge')])
        list_depth.extend([iaa.Affine(val, mode='edge')])
        list_depth.extend([iaa.Affine(10.0 / val_scala, mode='edge')])

        val_rotation = random.randrange(-180, 181, 90)
        list.extend([iaa.Affine(rotate=val_rotation, mode='edge')])
        list_depth.extend([iaa.Affine(rotate=val_rotation, mode='edge')])

        augseq = iaa.Sequential(list)
        batch[i] = augseq.augment_image(batch[i])
        augseq_depth = iaa.Sequential(list)
        batch_depth[i] = augseq_depth.augment_image(batch_depth[i])

        list = []
        list_depth = []
    return batch, batch_depth


# PARAMETERS
batch_size = 100
training_epochs = 500
every_epoch = 1
#datasetpath = "./rgbd-dataset_eval"
#classnum = 51
datasetpath = "./JHUIT50"
classnum = 49
#datasetpath = "./ocid"
#classnum = 59
#Loading Data
print ("LOADING RGB-D DATASET")
# 10 settings according to article
#testinstance_ids = [2,5,15,21,24,29,31,37,39,44,50,54,61,67,73,76,88,106,111,118,122,132,135,139,145,151,159,164,169,173,175,185,189,195,201,204,210,213,217,223,227,235,240,243,248,257,270,280,283,290,292]
#testinstance_ids = [2,5,15,18,24,30,32,37,42,44,50,60,64,68,75,77,92,102,112,118,122,128,134,139,147,151,159,163,168,174,178,184,187,193,198,203,205,212,219,222,231,236,240,244,249,263,269,277,283,289,298]
#testinstance_ids = [1,8,15,16,24,25,34,37,42,45,52,60,64,69,71,77,88,106,114,119,122,131,135,141,145,155,157,162,169,171,183,185,188,195,201,203,206,211,215,223,231,233,237,242,250,256,271,280,281,286,292]
#testinstance_ids = [4,10,12,17,22,30,31,38,41,47,50,59,62,70,74,82,87,103,110,116,125,131,136,141,146,153,160,163,167,174,176,186,187,197,198,202,206,211,220,226,227,233,239,246,252,262,265,277,281,286,293]
#testinstance_ids = [3,6,15,16,23,29,32,36,41,47,50,58,63,70,75,81,84,107,112,118,121,128,135,141,146,153,158,166,167,172,175,186,188,196,200,204,206,211,214,222,230,234,237,244,250,259,269,273,282,288,299]
#testinstance_ids = [2,7,14,16,22,30,33,38,42,43,50,59,62,68,73,81,91,99,110,115,124,129,135,142,148,151,158,163,169,174,177,184,188,194,199,203,207,212,214,222,232,235,239,242,252,255,272,273,282,288,291]
#testinstance_ids = [1,6,14,18,22,28,35,36,40,46,48,55,63,69,72,78,88,104,111,115,124,130,136,140,143,155,157,166,168,173,178,184,189,195,199,203,208,212,216,226,229,236,240,245,248,253,271,274,284,290,291]
#testinstance_ids = [4,7,15,16,23,26,33,37,39,45,48,57,62,67,74,83,88,108,110,118,124,128,134,140,148,152,160,162,167,172,176,185,187,194,200,204,210,213,220,226,232,235,238,243,248,257,268,274,281,290,293]
#testinstance_ids = [2,5,13,20,22,30,34,38,42,45,52,57,61,70,73,79,89,98,112,116,125,128,134,140,149,151,156,162,170,172,179,184,190,197,200,203,208,211,220,222,230,236,239,243,248,263,267,277,283,287,297]
#testinstance_ids = [2,5,12,19,23,29,35,38,39,43,48,59,63,68,74,78,89,99,112,118,123,128,137,140,146,154,159,164,167,173,178,185,189,195,199,203,206,212,215,226,232,233,237,243,248,263,265,276,285,287,295]
#RGBtrain, Dtrain, RGBtest, Dtest, RGBDtrain_label, RGBDtest_label = readRGBDData_sn(datasetpath, testinstance_ids, classnum)
RGBtrain, Dtrain, RGBtest, Dtest, RGBDtrain_label, RGBDtest_label = readJHUIT50Data_sn(datasetpath, classnum)
#RGBtrain, Dtrain, RGBtest, Dtest, RGBDtrain_label, RGBDtest_label = readocidData_sn(datasetpath, classnum)

print ("GENERATING DOMAIN DATA")

#total_train        = np.vstack([mnist_train, mnistm_train])
#total_test         = np.vstack([mnist_test, mnistm_test])
ntrain             = RGBtrain.shape[0]
ntest              = RGBtest.shape[0]
RGBtrain_domain = np.tile([1., 0.], [ntrain, 1])
Dtrain_domain = np.tile([0., 1.], [ntrain, 1])
RGBtest_domain = np.tile([1., 0.], [ntest, 1])
Dtest_domain = np.tile([0., 1.], [ntest, 1])
#total_test_domain  = np.vstack([np.tile([1., 0.], [ntest, 1]), np.tile([0., 1.], [ntest, 1])])
#n_total_train      = total_train.shape[0]
#n_total_test       = total_test.shape[0]

# GET PIXEL MEAN
pixel_mean_RGB = RGBtrain.mean((0, 1, 2))
pixel_mean_D = Dtrain.mean((0, 1, 2))
#np.save('./datasetpara/RGBD_mean_rgb.npy', pixel_mean_RGB)
#np.save('./datasetpara/RGBD_mean_d.npy', pixel_mean_D)

# PLOT IMAGES
# def imshow_grid(images, shape=[2, 8]):
#     from mpl_toolkits.axes_grid1 import ImageGrid
#     fig = plt.figure()
#     grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)
#     size = shape[0] * shape[1]
#     for i in range(size):
#         grid[i].axis('off')
#         grid[i].imshow(images[i])
#     plt.show()
# imshow_grid(mnist_train, shape=[5, 10])
# imshow_grid(mnistm_train, shape=[5, 10])


# def print_npshape(x, name):
#     print("SHAPE OF %s IS %s" % (name, x.shape,))


# SOURCE AND TARGET DATA
#source_train_img = mnist_train
#source_train_label = mnist_train_label
# source_train_img = np.concatenate((mnist_train,mnist_train),axis=0)
# source_train_label = np.concatenate((mnist_train_label,mnist_train_label),axis=0)
#
# source_test_img = mnist_test
# source_test_label = mnist_test_label

#target_train_img = mnistm_train
#target_train_label= mnistm_train_label
# target_train_img = np.concatenate((mnistm_train,mnistm_train),axis=0)
# target_train_label= np.concatenate((mnistm_train_label,mnistm_train_label),axis=0)
#
# target_test_img = mnistm_test
# target_test_label = mnistm_test_label




# DOMAIN ADVERSARIAL TRAINING
# domain_train_img = total_train
# domain_train_label = total_train_domain
#
# imgshape = source_train_img.shape[1:4]
# labelshape = source_train_label.shape[1]
#
# print_npshape(source_train_img, "source_train_img")
# print_npshape(source_train_label, "source_train_label")
# print_npshape(source_test_img, "source_test_img")
# print_npshape(source_test_label, "source_test_label")
# print_npshape(target_test_img, "target_test_img")
# print_npshape(target_test_label, "target_test_label")
# print_npshape(domain_train_img, "domain_train_img")
# print_npshape(domain_train_label, "domain_train_label")
#
# print
# imgshape
# print
# labelshape

RGBshape = RGBtrain.shape[1:4]
Dshape = Dtrain.shape[1:4]
labelshape = RGBDtrain_label.shape[1]


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0
    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            #return [tf.neg(grad) * l]
            return [tf.negative(grad) * l]
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
        self.num_calls += 1
        return y

flip_gradient = FlipGradientBuilder()

##Place Holder
RGB  = tf.placeholder(tf.uint8, [None, RGBshape[0], RGBshape[1], RGBshape[2]])
D  = tf.placeholder(tf.uint8, [None, Dshape[0], Dshape[1], Dshape[2]])

y  = tf.placeholder(tf.float32, [None, labelshape])
d_rgb  = tf.placeholder(tf.float32, [None, 2]) # DOMAIN LABEL
d_d = tf.placeholder(tf.float32, [None, 2]) # DOMAIN LABEL
lr = tf.placeholder(tf.float32, [])
dw = tf.placeholder(tf.float32, [])
istrain = tf.placeholder(tf.bool)

# FEATURE EXTRACTOR
# def shared_encoder(x, name='feat_ext', reuse=False):
#     with tf.variable_scope(name) as scope:
#         if reuse:
#             scope.reuse_variables()
#         x = (tf.cast(x, tf.float32) - pixel_mean) / 255.
#         net = slim.conv2d(x, 32, [5, 5], scope = 'conv1_shared_encoder')
#         net = slim.max_pool2d(net, [2, 2], scope='pool1_shared_encoder')
#         net = slim.conv2d(net, 48, [5, 5], scope='conv2_shared_encoder')
#         net = slim.max_pool2d(net, [2, 2], scope='pool2_shared_encoder')
#         feat = slim.flatten(net, scope='flat_shared_encoder')
#     return feat

def RGB_encoder(x, istrain, name='RGB_resnetfeat', reuse=False):
    with tf.variable_scope(name) as scope:
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            if reuse:
                scope.reuse_variables()
            x = tf.image.resize_bilinear(tf.cast(x, tf.float32), [224, 224])
            x = (x - pixel_mean_RGB) / 255.
            net, _ = resnet_v2.resnet_v2_50(x, is_training=istrain)
            #net = slim.flatten(net)
            #feat = slim.fully_connected(net, 1024)
            feat = slim.flatten(net)
    return feat

def D_encoder(x, istrain, name='D_resnetfeat', reuse=False):
    with tf.variable_scope(name) as scope:
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            if reuse:
                scope.reuse_variables()
            x = tf.image.resize_bilinear(tf.cast(x, tf.float32), [224, 224])
            x = (x - pixel_mean_D) / 255. #65535
            net, _ = resnet_v2.resnet_v2_50(x, is_training=istrain)
            #net = slim.flatten(net)
            #feat = slim.fully_connected(net, 1024)
            feat = slim.flatten(net)
    return feat


def RGB_share_encoder(resnetfeat, name='RGB_share_feat', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        net = slim.fully_connected(resnetfeat, 1024)
        net = slim.fully_connected(net, 512)
        feat = slim.fully_connected(net, 256)
    return feat

def D_share_encoder(resnetfeat, name='D_share_feat', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        net = slim.fully_connected(resnetfeat, 1024)
        net = slim.fully_connected(net, 512)
        feat = slim.fully_connected(net, 256)
    return feat

def share_encoder(feat1, feat2, name='share_feat', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        #feat = tf.add(feat1, feat2)
        #feat = tf.div(feat, 2.0)
        feat = tf.concat([feat1, feat2], 1)
    return feat


#Private Target Encoder
def private_RGB_encoder(resnetfeat, name='priviate_RGB_encoder', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        net = slim.fully_connected(resnetfeat, 1024)
        net = slim.fully_connected(net, 512)
        feat = slim.fully_connected(net, 256)
    return feat

#Private Source Encoder
def private_D_encoder(resnetfeat, name='priviate_D_encoder', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        net = slim.fully_connected(resnetfeat, 1024)
        net = slim.fully_connected(net, 512)
        feat = slim.fully_connected(net, 256)
    return feat

def total_feat(featshared, featprivate1, featprivate2, name='total_feat', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        feat = tf.concat([featshared, featprivate1, featprivate2], 1)
    return feat

# CLASS PREDICTION
def class_pred_net(feat, name='class_pred', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        #net = slim.fully_connected(feat, 100, scope='fc1')
        #net = slim.fully_connected(net, 100, scope='fc2')
        net = slim.fully_connected(feat, classnum, activation_fn = None, scope='out')
    return net

# DOMAIN PREDICTION
def domain_pred_net(feat, name='domain_pred', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        feat = flip_gradient(feat, dw) # GRADIENT REVERSAL
        net = slim.fully_connected(feat, 100, scope='fc1')
        net = slim.fully_connected(net, 2, activation_fn = None, scope='out')
    return net

def shared_decoder(feat,reuse=False, name='shared_decoder'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        net = slim.fully_connected(feat, 512)
        net = slim.fully_connected(net, 1024)
        net = slim.fully_connected(net, 2048, activation_fn = None)
    return net


# DIFFERENCE LOSS
def difference_loss(private_samples, shared_samples, weight=1.0, name=''):
  private_samples -= tf.reduce_mean(private_samples, 0)
  shared_samples -= tf.reduce_mean(shared_samples, 0)
  private_samples = tf.nn.l2_normalize(private_samples, 1)
  shared_samples = tf.nn.l2_normalize(shared_samples, 1)
  correlation_matrix = tf.matmul( private_samples, shared_samples, transpose_b=True)
  cost = tf.reduce_mean(tf.square(correlation_matrix)) * weight
  cost = tf.where(cost > 0, cost, 0, name='value')
  #tf.summary.scalar('losses/Difference Loss {}'.format(name),cost)
  assert_op = tf.Assert(tf.is_finite(cost), [cost])
  with tf.control_dependencies([assert_op]):
     tf.losses.add_loss(cost)
  return cost

# DOMAIN ADVERSARIAL NEURAL NETWORK
feat_RGB = RGB_encoder(RGB, istrain, name='RGB_resnet')
feat_D = D_encoder(D, istrain, name='D_resnet')
feat_RGB_shared  = RGB_share_encoder(feat_RGB, name='affn_shared_feat')
feat_D_shared  = D_share_encoder(feat_D, name='affn_shared_feat', reuse=True)
feat_shared = share_encoder(feat_RGB_shared, feat_D_shared)

domain_RGB_pred_affn = domain_pred_net(feat_RGB_shared, name='domain_pred')
domain_D_pred_affn = domain_pred_net(feat_D_shared, name='domain_pred', reuse=True)

#pretrain for resnet
class_pred_rgb = class_pred_net(feat_RGB, name='rgb_pretrain_class_pred')
class_pred_d = class_pred_net(feat_D, name='d_pretrain_class_pred')

#Private & Shared Encoder
feat_RGB_private = private_RGB_encoder(feat_RGB, name='RGB_private')
feat_D_private = private_D_encoder(feat_D, name='D_private')
#feat_total = feat_shared
feat_total = total_feat(feat_shared, feat_RGB_private, feat_D_private)
class_pred_affn  = class_pred_net(feat_total, name='affn_class_pred')
#class_pred_affn  = class_pred_net(feat_shared, name='affn_class_pred')

#Input for Decoder
RGB_concat_feat = tf.concat([feat_RGB_shared, feat_RGB_private],1)
D_concat_feat = tf.concat([feat_D_shared, feat_D_private],1)

#Decoder
#target_recon = small_decoder(target_concat_feat,28,28,3)
RGB_recon = shared_decoder(RGB_concat_feat, name='RGB_decoder')
D_recon = shared_decoder(D_concat_feat, name='D_decoder')

############################################################################################################################################################

print ("MODEL READY")

t_weights = tf.trainable_variables()

# TOTAL WEIGHTS
# print ("   TOTAL WEIGHT LIST")
# for i in range(len(t_weights)):  print ("[%2d/%2d] [%s]" % (i, len(t_weights), t_weights[i]))
"""
# FEATURE EXTRACTOR + CLASS PREDICTOR
print ("   WEIGHT LIST FOR CLASS PREDICTOR")
w_class = []
for i in range(len(t_weights)):
    if t_weights[i].name[:9] == 'dann_feat' or t_weights[i].name[:10] == 'dann_class':
        w_class.append(tf.nn.l2_loss(t_weights[i]))
        print ("[%s]    \t ADDED TO W_CLASS LIST" % (t_weights[i].name))

l2loss_dann_class = tf.add_n(w_class)

# FEATURE EXTRACTOR + DOMAIN CLASSIFIER
print ("\n   WEIGHT LIST FOR DOMAIN CLASSIFIER")
w_domain = []

for i in range(len(t_weights)):
    if t_weights[i].name[:8] == 'dann_feat' or t_weights[i].name[:11] == 'dann_domain':
        w_domain.append(tf.nn.l2_loss(t_weights[i]))
        print ("[%s]    \t ADDED TO W_DOMAIN LIST" % (t_weights[i].name))


l2loss_cnn_domain = tf.add_n(w_domain)
"""


# FUNCTIONS FOR DANN
class_loss_affn  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=class_pred_affn, labels=y))
domain_loss_affn = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=domain_RGB_pred_affn, labels=d_rgb)) + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=domain_D_pred_affn, labels=d_d))) / 2.0
RGB_recon_loss = tf.reduce_mean(tf.pow((feat_RGB - RGB_recon),2))
D_recon_loss = tf.reduce_mean(tf.pow((feat_D - D_recon),2))
RGB_diff_loss = difference_loss(feat_RGB_shared, feat_RGB_private)
D_diff_loss = difference_loss(feat_D_shared, feat_D_private)

class_loss_rgb  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=class_pred_rgb, labels=y))
class_loss_d  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=class_pred_d, labels=y))
############################################################################################################################################################
#losses = class_loss_affn + RGB_recon_loss + D_recon_loss + RGB_diff_loss + D_diff_loss
losses = class_loss_affn + RGB_recon_loss + D_recon_loss + RGB_diff_loss + D_diff_loss
domain_loss_affn_combine = domain_loss_affn + class_loss_affn
#losstry = class_loss_affn  + RGB_diff_loss + D_diff_loss
#optm_class_affn  = tf.train.MomentumOptimizer(lr, 0.9).minimize(losses)

#optm_domain_affn = tf.train.MomentumOptimizer(lr, 0.9).minimize(domain_loss_affn)

params_trainable = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='affn_shared_feat') \
                    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='share_feat') \
                    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='domain_pred') \
                    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='RGB_private') \
                    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D_private') \
                    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='total_feat') \
                    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='affn_class_pred') \
                    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='RGB_decoder') \
                    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D_decoder')


optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
#optimizer = tf.train.AdamOptimizer(learning_rate=lr)
optm_class_rgb = slim.learning.create_train_op(class_loss_rgb, optimizer)
optm_class_d = slim.learning.create_train_op(class_loss_d, optimizer)

optm_class_affn1 = slim.learning.create_train_op(class_loss_affn, optimizer, variables_to_train=params_trainable)
optm_class_affn = slim.learning.create_train_op(losses, optimizer, variables_to_train=params_trainable)
optm_domain_affn = slim.learning.create_train_op(domain_loss_affn, optimizer,  variables_to_train=params_trainable)
optm_domain_affn_combine = slim.learning.create_train_op(domain_loss_affn_combine, optimizer,  variables_to_train=params_trainable)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
if update_ops:
    updates = tf.group(*update_ops)
    class_loss_rgb = control_flow_ops.with_dependencies([updates], class_loss_rgb)
    class_loss_d = control_flow_ops.with_dependencies([updates], class_loss_d)
    class_loss_affn = control_flow_ops.with_dependencies([updates], class_loss_affn)
    losses = control_flow_ops.with_dependencies([updates], losses)
    domain_loss_affn = control_flow_ops.with_dependencies([updates], domain_loss_affn)
    domain_loss_affn_combine = control_flow_ops.with_dependencies([updates], domain_loss_affn_combine)

#optm_class_affn = tf.train.AdamOptimizer(learning_rate=lr).minimize(losses)
#optm_domain_affn = tf.train.AdamOptimizer(learning_rate=lr).minimize(domain_loss_affn)


accr_class_affn  = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(class_pred_affn, 1), tf.arg_max(y, 1)), tf.float32))
accr_domain_affn = (tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(domain_RGB_pred_affn, 1), tf.arg_max(d_rgb, 1)), tf.float32)) + tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(domain_D_pred_affn, 1), tf.arg_max(d_d, 1)), tf.float32))) / 2.0
accr_class_rgb  = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(class_pred_rgb, 1), tf.arg_max(y, 1)), tf.float32))
accr_class_d  = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(class_pred_d, 1), tf.arg_max(y, 1)), tf.float32))

print ("FUNCTIONS READY")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
tf.set_random_seed(0)
init = tf.global_variables_initializer()
sess.run(init)

def name_in_checkpoint(var):
    if "RGB_resnet" in var.op.name:
        return var.op.name.replace("RGB_resnet/","")
    if "D_resnet" in var.op.name:
        return var.op.name.replace("D_resnet/", "")
variables_to_restore = slim.get_model_variables()
variables_to_restore1 = {name_in_checkpoint(var):var for var in variables_to_restore if 'RGB_resnet/resnet' in var.op.name}
variables_to_restore2 = {name_in_checkpoint(var):var for var in variables_to_restore if 'D_resnet/resnet' in var.op.name}
restorer1 = tf.train.Saver(variables_to_restore1)
restorer2 = tf.train.Saver(variables_to_restore2)
restorer1.restore(sess, "./resnet_v2_50.ckpt")
restorer2.restore(sess, "./resnet_v2_50.ckpt")




print ("SESSION OPENED")



# PARAMETERS
batch_size = 64
training_epochs = 10
every_epoch = 1
#num_batch = int(ntrain / batch_size) + 1
num_batch = int(ntrain/ batch_size)
total_iter = training_epochs * num_batch
indexlist = range(0, ntrain)

for epoch in range(training_epochs):
    randpermlist = np.random.permutation(ntrain)
    for i in range(num_batch):
        curriter = epoch * num_batch + i
        p = float(curriter) / float(total_iter)
        dw_val = 2. / (1. + np.exp(-10. * p)) - 1
        lr_val = 0.01 / (1. + 10 * p) ** 0.75
        randidx_class = randpermlist[i * batch_size:min((i + 1) * batch_size, ntrain-1)]
        batch_RGB = RGBtrain[randidx_class]
        batch_D = Dtrain[randidx_class]
        batch_y_class = RGBDtrain_label[randidx_class, :]
        feeds_class = {RGB: batch_RGB, D: batch_D, y: batch_y_class, lr: lr_val, dw: dw_val, istrain: True}
        #feeds_class = {source: batch_source, target:batch_target, y: batch_y_class, lr: lr_val, dw: dw_val}
        _, lossclass_val_rgb, accr_trainrgb = sess.run([optm_class_rgb, class_loss_rgb, accr_class_rgb], feed_dict=feeds_class)
        _, lossclass_val_d, accr_traind = sess.run([optm_class_d, class_loss_d, accr_class_d], feed_dict=feeds_class)
        # print("[%d/%d][%d/%d] p: %.3f lossclass_rgb: %.3e, rgb_accuracy: %.3f, lossclass_d: %.3e, d_accuracy: %.3f"
        #       % (epoch, training_epochs, curriter, total_iter, p, lossclass_val_rgb, accr_trainrgb, lossclass_val_d, accr_traind))
    if epoch % every_epoch == 0:
        indexleft = 0
        resultsum_rgb = 0
        resultsum_d = 0
        while indexleft < ntest:
            indexright = min(indexleft + 32, ntest)
            batch_RGBtest = RGBtest[indexleft:indexright, ...]
            batch_Dtest = Dtest[indexleft:indexright, ...]
            batch_ytest = RGBDtest_label[indexleft:indexright, ...]
            feed_test = {RGB: batch_RGBtest, D: batch_Dtest, y: batch_ytest, istrain: False}
            accr_rgb_resnet = sess.run(accr_class_rgb, feed_dict=feed_test)
            accr_d_resnet = sess.run(accr_class_d, feed_dict=feed_test)
            resultsum_rgb += accr_rgb_resnet * (indexright-indexleft)
            resultsum_d += accr_d_resnet * (indexright - indexleft)
            indexleft = indexright
        resultaccr_rgb = resultsum_rgb / ntest
        resultaccr_d = resultsum_d / ntest
        print(" RGB resnet: CLASSIFY ACCURACY: %.3f; D resnet: CLASSIFY ACCURACY: %.3f"
              % (resultaccr_rgb, resultaccr_d))

# batch_size = 32
# training_epochs = 1
# every_epoch = 1
# #num_batch = int(ntrain / batch_size) + 1
# num_batch = int(ntrain/ batch_size)
# total_iter = training_epochs * num_batch
# indexlist = range(0, ntrain)
#
#
# for epoch in range(training_epochs):
#     randpermlist = np.random.permutation(ntrain)
#     for i in range(num_batch):
#         # REVERSAL WEIGHT AND LEARNING RATE SCHEDULE
#         curriter = epoch * num_batch + i
#         p = float(curriter) / float(total_iter)
#         dw_val = 2. / (1. + np.exp(-10. * p)) - 1
#         dw_val = 1.
#         lr_val = 0.01 * ((0.1 + 1 - p)*(0.5 * (1 + np.cos(np.pi*2*0.2*p))) + 0.1)
#         lr_val1 = 0.001 / (1. + 10 * p) ** 0.75
#         lr_val2 = 0.01 * ((1 - 0.1) * (0.5 * (1 + np.cos(np.pi * p))) + 0.1)
#         lr_val3 = 0.01 / (1. + 10 * p) ** 0.75
#
#         # OPTIMIZE DANN: CLASS-CLASSIFIER
#         #randidx_class = randpermlist[i * batch_size:min((i + 1) * batch_size, ntrain - 1)]
#         randidx_class = randpermlist[i * batch_size:min((i + 1) * batch_size, ntrain-1)]
#         #randidx_class = random.sample(indexlist, batch_size)
#         #randidx_domain = randpermlist[i * batch_size:min((i + 1) * batch_size, ntrain-1)]
#         # randidx_domain = np.random.permutation(n_total_train)[:batch_size]
#         batch_RGB = RGBtrain[randidx_class]
#         batch_D = Dtrain[randidx_class]
#         batch_y_class = RGBDtrain_label[randidx_class, :]
#         batch_RGB_domain = RGBtrain_domain[randidx_class]
#         batch_D_domain = Dtrain_domain[randidx_class]
#
#         feeds_domain = {RGB: batch_RGB, D: batch_D, y: batch_y_class, lr: 0.1, dw: dw_val, istrain: True, d_rgb: batch_RGB_domain, d_d: batch_D_domain}
#         _, lossdomain_val_affn, lossclass_val_affn, accr_domainbatch = sess.run([optm_domain_affn, domain_loss_affn, class_loss_affn, accr_domain_affn], feed_dict=feeds_domain)
#         # concatenate train
#         print("[%d/%d][%d/%d] p: %.3f lossclass_val: %.3e, lossdomain_val: %.3e, batch domain_accuracy: %.3f"
#          % (epoch, training_epochs, curriter, total_iter, p, lossclass_val_affn, lossdomain_val_affn, accr_domainbatch))


batch_size = 32
training_epochs = 10
every_epoch = 1
#num_batch = int(ntrain / batch_size) + 1
num_batch = int(ntrain/ batch_size)
total_iter = training_epochs * num_batch
indexlist = range(0, ntrain)

maxaccr = 0

classrecord = []
domainrecord = []
decomrecord = []
totalrecord = []
domainaccrecord = []

for epoch in range(2):
    randpermlist = np.random.permutation(ntrain)
    for i in range(num_batch):
        # REVERSAL WEIGHT AND LEARNING RATE SCHEDULE
        curriter = epoch * num_batch + i
        p = float(curriter) / float(total_iter)
        dw_val = 2. / (1. + np.exp(-10. * p)) - 1
        lr_val = 0.01 * ((0.1 + 1 - p)*(0.5 * (1 + np.cos(np.pi*2*0.2*p))) + 0.1)
        lr_val1 = 0.001 / (1. + 10 * p) ** 0.75
        lr_val2 = 0.01 * ((1 - 0.1) * (0.5 * (1 + np.cos(np.pi * p))) + 0.1)
        lr_val3 = 0.01 / (1. + 10 * p) ** 0.75

        # OPTIMIZE DANN: CLASS-CLASSIFIER
        #randidx_class = randpermlist[i * batch_size:min((i + 1) * batch_size, ntrain - 1)]
        randidx_class = randpermlist[i * batch_size:min((i + 1) * batch_size, ntrain-1)]
        #randidx_class = random.sample(indexlist, batch_size)
        #randidx_domain = randpermlist[i * batch_size:min((i + 1) * batch_size, ntrain-1)]
        # randidx_domain = np.random.permutation(n_total_train)[:batch_size]
        batch_RGB = RGBtrain[randidx_class]
        batch_D = Dtrain[randidx_class]
        #batch_x_domain = total_train[randidx_domain]
        batch_y_class = RGBDtrain_label[randidx_class, :]
        #batch_RGB, batch_D = data_aug(batch_RGB, batch_D)
        #print(batch_source.shape)
        #print(batch_target.shape)
        #print(batch_x_domain.shape)
        #print(batch_y_class.shape)
        batch_RGB_domain = RGBtrain_domain[randidx_class]
        batch_D_domain = Dtrain_domain[randidx_class]
        feeds_class = {RGB: batch_RGB, D: batch_D, y: batch_y_class, lr: lr_val, dw: dw_val, istrain: True, d_rgb: batch_RGB_domain, d_d: batch_D_domain}
        #feeds_class = {source: batch_source, target:batch_target, y: batch_y_class, lr: lr_val, dw: dw_val}
        _ = sess.run([optm_class_affn1], feed_dict=feeds_class)
        _, accr_trainbatch = sess.run([optm_class_affn, accr_class_affn], feed_dict=feeds_class)
        # OPTIMIZE DANN: DOMAIN-CLASSIFER
        # randidx_domain = np.random.permutation(n_total_train)[:batch_size]
        # batch_x_domain = total_train[randidx_domain]
        # batch_d_domain = total_train_domain[randidx_domain, :]
        #feeds_domain = {RGB: batch_RGB, D: batch_D, d_rgb: batch_RGB_domain, d_d: batch_D_domain, lr: lr_val, dw: dw_val, istrain: True}
        feeds_domain = {RGB: batch_RGB, D: batch_D, y: batch_y_class, lr: lr_val3, dw: dw_val, istrain: True, d_rgb: batch_RGB_domain, d_d: batch_D_domain}
        _, lossdomain_val_affn, lossclass_val_affn, total_loss, accr_domainbatch = sess.run([optm_domain_affn_combine, domain_loss_affn, class_loss_affn, losses, accr_domain_affn], feed_dict=feeds_domain)

        print(
            "[%d/%d][%d/%d] p: %.3f lossclass_val: %.3e, lossdomain_val: %.3e, lossdecom_val:  %.3e, losstotal_val: %.3e, batch accuracy: %.3f, batch domain_accuracy: %.3f"
            % (epoch, training_epochs, curriter, total_iter, p, lossclass_val_affn, lossdomain_val_affn, total_loss - lossclass_val_affn,
               total_loss + lossdomain_val_affn, accr_trainbatch, accr_domainbatch))

        classrecord.append(lossclass_val_affn)
        domainrecord.append(lossdomain_val_affn)
        decomrecord.append(total_loss - lossclass_val_affn)
        totalrecord.append(total_loss + lossdomain_val_affn)
        domainaccrecord.append(accr_domainbatch)

        #if epoch % every_epoch == 0:
        # CHECK BOTH LOSSES

        # print("[%d/%d][%d/%d] p: %.3f lossclass_val: %.3e, batch accuracy: %.3f"
        #       % (epoch, training_epochs, curriter, total_iter, p, lossclass_val_affn, accr_trainbatch))
        #print(batch_RGB.shape[0])
        # CHECK ACCUARACIES OF BOTH SOURCE AND TARGET
        #feed_source = {source: batch_source, source_target: batch_x_domain, target:batch_target, y: batch_y_class, lr: lr_val, dw: dw_val}
        #feed_target = {source: batch_source, source_target: batch_x_domain, target: batch_target, y: batch_y_class,lr: lr_val, dw: dw_val}
        #feed_test = {RGB: RGBtrain[0:ntrain:20], D:Dtrain[0:ntrain:20], y: RGBDtrain_label[0:ntrain:20, :]}
        if curriter % 20 == 0:
            indexleft = 0
            resultsum_affn = 0
            resultsum_rgbdc = 0
            while indexleft < ntest:
                indexright = min(indexleft + 32, ntest)
                batch_RGBtest = RGBtest[indexleft:indexright, ...]
                batch_Dtest = Dtest[indexleft:indexright, ...]
                batch_ytest = RGBDtest_label[indexleft:indexright, ...]
                feed_test = {RGB: batch_RGBtest, D: batch_Dtest, y: batch_ytest, istrain: False}
                accr_affn = sess.run(accr_class_affn, feed_dict=feed_test)
                resultsum_affn += accr_affn * (indexright-indexleft)
                indexleft = indexright
                #print("[%.3f/%.3f], [%.3f/%.3f]" % (accr_affn * batch_ytest.shape[0], batch_ytest.shape[0], accr_cnn * batch_ytest.shape[0], batch_ytest.shape[0]))
            resultaccr_affn = resultsum_affn / ntest
            print(" AFFN: CLASSIFY ACCURACY: %.3f" % resultaccr_affn)
            if resultaccr_affn > maxaccr:
                maxaccr = resultaccr_affn


np.save('seclassloss.npy',classrecord)
np.save('sedomainloss.npy',domainrecord)
np.save('sedecomloss.npy',decomrecord)
np.save('setotalloss.npy',totalrecord)
np.save('sedomainacc.npy',domainaccrecord)
print(maxaccr)
