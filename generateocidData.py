import os
from PIL import Image
import numpy as np
import tensorflow as tf

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)





def readocidData_sn(filePath, classnum):
    RGBtrain = []
    RGBtest = []
    Dtrain = []
    Dtest = []
    RGBDtrain_label = []
    RGBDtest_label = []


    txttrainfilepath = filePath + '/split_files_and_labels/arid20_clean_sync_instances.txt'
    txttestfilepath = filePath + '/split_files_and_labels/arid10_clean_sync_instances.txt'
    trainrgbimagepath = filePath + '/ARID20_crops/squared_rgb/'
    traindepthimagepath = filePath + '/ARID20_crops/surfnorm++/'
    testrgbimagepath = filePath + '/ARID10_crops/squared_rgb/'
    testdepthimagepath = filePath + '/ARID10_crops/surfnorm++/'

    img_paths = []
    labels = []

    with open(txttrainfilepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            img_paths.append(items[0])
            labels.append(int(items[1]))


    count  = 0
    for path in img_paths:
        img_paths_rgb = trainrgbimagepath + path
        img_paths_depth = traindepthimagepath + path
        if (os.path.exists(img_paths_rgb)) & (os.path.exists(img_paths_depth)):
            index = labels[count]
            img_rgb = Image.open(img_paths_rgb)
            if img_rgb.height < img_rgb.width:
                heightnew = max(round(img_rgb.height * 112 / img_rgb.width), 1)
                img_rgb = img_rgb.resize((112, heightnew))
                p1 = 0
                p2 = round((112 - heightnew) / 2)
                p3 = 112
                p4 = p2 + heightnew
            else:
                widthnew = max(round(img_rgb.width * 112 / img_rgb.height), 1)
                img_rgb = img_rgb.resize((widthnew, 112))
                p1 = round((112 - widthnew) / 2)
                p2 = 0
                p3 = p1 + widthnew
                p4 = 112
            imout = Image.new("RGB", (112, 112))
            imout.paste(img_rgb, (p1, p2, p3, p4))
            img_rgb = np.array(imout)
            img_rgb = np.reshape(img_rgb, (1,) + img_rgb.shape)
            img_d = Image.open(img_paths_depth)
            if img_d.height < img_d.width:
                heightnew = max(round(img_d.height * 112 / img_d.width), 1)
                img_d = img_d.resize((112, heightnew))
                p1 = 0
                p2 = round((112 - heightnew) / 2)
                p3 = 112
                p4 = p2 + heightnew
            else:
                widthnew = max(round(img_d.width * 112 / img_d.height), 1)
                img_d = img_d.resize((widthnew, 112))
                p1 = round((112 - widthnew) / 2)
                p2 = 0
                p3 = p1 + widthnew
                p4 = 112
            imout = Image.new("RGB", (112, 112), "#8080FF")
            imout.paste(img_d, (p1, p2, p3, p4))
            img_d = np.array(imout)
            img_d = np.reshape(img_d, (1,) + img_d.shape)
            # train
            if count > 0:
                labeltemp = np.zeros((1, classnum))
                labeltemp[0, index] = 1
                RGBDtrain_label = np.vstack([RGBDtrain_label, labeltemp])
                RGBtrain = np.vstack([RGBtrain, img_rgb])
                Dtrain = np.vstack([Dtrain, img_d])
            else:
                RGBDtrain_label = np.zeros((1, classnum))
                RGBDtrain_label[0, index] = 1
                RGBtrain = img_rgb
                Dtrain = img_d
        count += 1

    img_paths = []
    labels = []
    with open(txttestfilepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            img_paths.append(items[0])
            labels.append(int(items[1]))


    count  = 0
    for path in img_paths:
        img_paths_rgb = testrgbimagepath + path
        img_paths_depth = testdepthimagepath + path
        if (os.path.exists(img_paths_rgb)) & (os.path.exists(img_paths_depth)):
            index = labels[count]
            img_rgb = Image.open(img_paths_rgb)
            if img_rgb.height < img_rgb.width:
                heightnew = max(round(img_rgb.height * 112 / img_rgb.width), 1)
                img_rgb = img_rgb.resize((112, heightnew))
                p1 = 0
                p2 = round((112 - heightnew) / 2)
                p3 = 112
                p4 = p2 + heightnew
            else:
                widthnew = max(round(img_rgb.width * 112 / img_rgb.height), 1)
                img_rgb = img_rgb.resize((widthnew, 112))
                p1 = round((112 - widthnew) / 2)
                p2 = 0
                p3 = p1 + widthnew
                p4 = 112
            imout = Image.new("RGB", (112, 112))
            imout.paste(img_rgb, (p1, p2, p3, p4))
            img_rgb = np.array(imout)
            img_rgb = np.reshape(img_rgb, (1,) + img_rgb.shape)
            img_d = Image.open(img_paths_depth)
            if img_d.height < img_d.width:
                heightnew = max(round(img_d.height * 112 / img_d.width), 1)
                img_d = img_d.resize((112, heightnew))
                p1 = 0
                p2 = round((112 - heightnew) / 2)
                p3 = 112
                p4 = p2 + heightnew
            else:
                widthnew = max(round(img_d.width * 112 / img_d.height), 1)
                img_d = img_d.resize((widthnew, 112))
                p1 = round((112 - widthnew) / 2)
                p2 = 0
                p3 = p1 + widthnew
                p4 = 112
            imout = Image.new("RGB", (112, 112), "#8080FF")
            imout.paste(img_d, (p1, p2, p3, p4))
            img_d = np.array(imout)
            img_d = np.reshape(img_d, (1,) + img_d.shape)
            # test
            if count > 0:
                labeltemp = np.zeros((1, classnum))
                labeltemp[0, index] = 1
                RGBDtest_label = np.vstack([RGBDtest_label, labeltemp])
                RGBtest = np.vstack([RGBtest, img_rgb])
                Dtest = np.vstack([Dtest, img_d])
            else:
                RGBDtest_label = np.zeros((1, classnum))
                RGBDtest_label[0, index] = 1
                RGBtest = img_rgb
                Dtest = img_d
        count += 1

    return RGBtrain, Dtrain, RGBtest, Dtest, RGBDtrain_label, RGBDtest_label

def readocidData_tsne(filePath, tsnenum):
    RGBtrain = []
    RGBtest = []
    Dtrain = []
    Dtest = []
    RGBDtrain_label = []
    RGBDtest_label = []


    txttestfilepath = filePath + '/split_files_and_labels/arid10_clean_sync_instances.txt'
    testrgbimagepath = filePath + '/ARID10_crops/squared_rgb/'
    testdepthimagepath = filePath + '/ARID10_crops/surfnorm++/'


    img_paths = []
    labels = []
    with open(txttestfilepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            img_paths.append(items[0])
            labels.append(int(items[1]))


    count  = 0
    for path in img_paths:
        img_paths_rgb = testrgbimagepath + path
        img_paths_depth = testdepthimagepath + path
        if (os.path.exists(img_paths_rgb)) & (os.path.exists(img_paths_depth)):
            index = labels[count]
            if index >= tsnenum:
                count += 1
                continue
            img_rgb = Image.open(img_paths_rgb)
            if img_rgb.height < img_rgb.width:
                heightnew = max(round(img_rgb.height * 112 / img_rgb.width), 1)
                img_rgb = img_rgb.resize((112, heightnew))
                p1 = 0
                p2 = round((112 - heightnew) / 2)
                p3 = 112
                p4 = p2 + heightnew
            else:
                widthnew = max(round(img_rgb.width * 112 / img_rgb.height), 1)
                img_rgb = img_rgb.resize((widthnew, 112))
                p1 = round((112 - widthnew) / 2)
                p2 = 0
                p3 = p1 + widthnew
                p4 = 112
            imout = Image.new("RGB", (112, 112))
            imout.paste(img_rgb, (p1, p2, p3, p4))
            img_rgb = np.array(imout)
            img_rgb = np.reshape(img_rgb, (1,) + img_rgb.shape)
            img_d = Image.open(img_paths_depth)
            if img_d.height < img_d.width:
                heightnew = max(round(img_d.height * 112 / img_d.width), 1)
                img_d = img_d.resize((112, heightnew))
                p1 = 0
                p2 = round((112 - heightnew) / 2)
                p3 = 112
                p4 = p2 + heightnew
            else:
                widthnew = max(round(img_d.width * 112 / img_d.height), 1)
                img_d = img_d.resize((widthnew, 112))
                p1 = round((112 - widthnew) / 2)
                p2 = 0
                p3 = p1 + widthnew
                p4 = 112
            imout = Image.new("RGB", (112, 112), "#8080FF")
            imout.paste(img_d, (p1, p2, p3, p4))
            img_d = np.array(imout)
            img_d = np.reshape(img_d, (1,) + img_d.shape)
            # test
            if RGBDtest_label != []:
                labeltemp = np.zeros((1, tsnenum))
                labeltemp[0, index] = 1
                RGBDtest_label = np.vstack([RGBDtest_label, labeltemp])
                RGBtest = np.vstack([RGBtest, img_rgb])
                Dtest = np.vstack([Dtest, img_d])
            else:
                RGBDtest_label = np.zeros((1, tsnenum))
                RGBDtest_label[0, index] = 1
                RGBtest = img_rgb
                Dtest = img_d
        count += 1

    return RGBtest, Dtest, RGBDtest_label




def create_record(filePath):
    """
    :param filePath: your dataset path
    :return: tfrecords file
    """
    writer = tf.python_io.TFRecordWriter("./rgbd51test.tfrecords")

    index = 0
    for classname in os.listdir(filePath):
        for instancename in os.listdir(filePath + '/' + classname):
            instancenum = 0
            for filename in os.listdir(filePath + '/' + classname + '/' + instancename):
                image_path = filePath + '/' + classname + '/' + instancename + '/' + filename
                imageclassjudge = filename[-9]
                # _ rgb,k d
                if imageclassjudge == '_':
                    filename_base = filename[:-8]
                    filename_depth = filename_base + 'maskcrop.png'
                    if (os.path.exists(image_path))&(os.path.exists(filePath + '/' + classname + '/' + instancename + '/' + filename_depth)):
                        img_rgb = Image.open(image_path).convert("RGB")
                        img_rgb = img_rgb.resize((256, 256))
                        img_rgb_raw = img_rgb.tostring()
                        img_d = Image.open(filePath + '/' + classname + '/' + instancename + '/' + filename_depth)
                        img_d = img_d.resize((256, 256))
                        img_d_raw = img_d.tostring()
                        if (len(img_rgb_raw) == 196608) & (len(img_d_raw) == 65536):
                            # print (len(img_raw))
                            instancenum += 1
                            if instancenum > 11:
                                break
                            if instancenum < 11:
                                continue
                            example = tf.train.Example(features=tf.train.Features(
                                feature={"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                                        "img_rgb_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_rgb_raw])),
                                        "img_d_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_d_raw]))}
                             ))
                            writer.write(example.SerializeToString())

        index += 1
    writer.close()
    print ("create record is finished")



#
def read_and_decode(filename, batch_size, is_batch = True):
    """

    :param filename: tfrecords path
    :param batch_size:
    :param is_batch: train or test
    :return: batch of iamges and  labels
    """
    if(is_batch):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_rgb_raw': tf.FixedLenFeature([], tf.string),
                                           'img_d_raw': tf.FixedLenFeature([], tf.string)
                                       })

        img_rgb = tf.decode_raw(features['img_rgb_raw'], tf.uint8)
        img_rgb = tf.cast(img_rgb, tf.float32)
        #print(img.shape)
        img_rgb = tf.reshape(img_rgb, [256, 256, 3])
        img_d = tf.decode_raw(features['img_d_raw'], tf.uint8)
        img_d = tf.reshape(img_d, [256, 256, 1])
        img_d = tf.cast(img_d, tf.float32)
        # print(img.shape)


        # data argumentation
        image_rgb = tf.random_crop(img_rgb, [227, 227, 3])# randomly crop the image size to 224 x 224
        image_rgb = tf.image.random_flip_left_right(image_rgb)
        image_d = tf.random_crop(img_d, [227, 227])  # randomly crop the image size to 224 x 224
        image_d = tf.image.random_flip_left_right(image_d)
        #image = tf.image.random_flip_up_down(image)
        #image = tf.image.random_brightness(image, max_delta=63)
        #image = tf.image.random_contrast(image,lower=0.2,upper=1.8)

       # img = tf.image.per_image_standardization(image)
        # img = 2*( tf.cast(image, tf.float32) * (1. / 255) - 0.5)
        img_rgb = tf.subtract(image_rgb, IMAGENET_MEAN )
        img_d = tf.subtract(image_d, (255 / 2.))
        # RGB -> BGR, for using pretrained alexnet
        img_bgr = img_rgb[..., ::-1]
        # d from grey to bgr
        img_d = tf.tile(img_d, [1, 1, 3])
        label = tf.cast(features['label'], tf.int32)

        images_bgr, images_d, label_batch = tf.train.shuffle_batch(
                                        [img_bgr, img_d, label],
                                        batch_size = batch_size,
                                        num_threads= 16,
                                        capacity = 5000,
                                        min_after_dequeue = 2000)
        ## ONE-HOT
        n_classes = 51
        label_batch = tf.one_hot(label_batch, depth= n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        labels = tf.reshape(label_batch, [batch_size, n_classes])
    else:

        labels = []
        images_bgr = []
        images_d = []
        for s_example in tf.python_io.tf_record_iterator(filename):
            features = tf.parse_single_example(s_example,
                                               features={
                                              'label':tf.FixedLenFeature([],tf.int64),
                                              'img_rgb_raw': tf.FixedLenFeature([], tf.string),
                                              'img_d_raw': tf.FixedLenFeature([], tf.string)
                                               })
            img_rgb = tf.decode_raw(features['img_rgb_raw'], tf.uint8)
            img_rgb = tf.cast(img_rgb, tf.float32)
            img_rgb = tf.reshape(img_rgb, [256, 256, 3])
            img_d = tf.decode_raw(features['img_d_raw'], tf.uint8)
            img_d = tf.reshape(img_d, [256, 256, 1])
            img_d = tf.cast(img_d, tf.float32)
            #image = tf.random_crop(img,[227,227,3])
            image_rgb = tf.image.resize_images(img_rgb, (224,224), method=0)
            image_d = tf.image.resize_images(img_d, (224, 224), method=0)
            img_rgb = tf.subtract(image_rgb, IMAGENET_MEAN)
            img_d = tf.subtract(image_d, (255 / 2.))
            # RGB -> BGR, for using pretrained alexnet
            img_bgr = img_rgb[..., ::-1]
            # d from grey to bgr
            img_d = tf.tile(img_d, [1,1,3])
            images_bgr.append(tf.expand_dims(img_bgr,0))
            images_d.append(tf.expand_dims(img_d, 0))

            ##labels
            n_classes =51
            label = tf.cast(features['label'], tf.int32)
            label = tf.one_hot(label, depth=n_classes)
            labels.append(tf.expand_dims(label,0))
    return tf.concat(images_bgr,0), tf.concat(images_d,0), tf.concat(labels,0)

# ## test
if __name__ == '__main__' :
    #create_record('./rgbd-dataset')
    #print('create done!')
    datasetpath = "./ocid"
    classnum = 59
    # Loading Data
    print("LOADING JHUIT50 DATASET")
    RGBtrain, Dtrain, RGBtest, Dtest, RGBDtrain_label, RGBDtest_label = readocidData_sn(datasetpath, classnum)
    print(RGBtrain.shape)
    print(RGBtrain.dtype)
    print(Dtrain.shape)
    print(Dtrain.dtype)
    print(RGBtest.shape)
    print(RGBtest.dtype)
    print(Dtest.shape)
    print(Dtest.dtype)
    print(RGBDtrain_label.shape)
    print(RGBDtrain_label.dtype)
    print(RGBDtest_label.shape)
    print(RGBDtest_label.dtype)

