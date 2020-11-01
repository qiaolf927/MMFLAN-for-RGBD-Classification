import os
from PIL import Image
import numpy as np
import tensorflow as tf

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
rgbdclass = ['apple', 'ball', 'banana', 'bell_pepper', 'binder', 'bowl', 'calculator', 'camera', 'cap', 'cell_phone',
             'cereal_box', 'coffee_mug', 'comb', 'dry_battery', 'flashlight', 'food_bag', 'food_box', 'food_can',
             'food_cup', 'food_jar', 'garlic', 'glue_stick', 'greens', 'hand_towel', 'instant_noodles', 'keyboard',
             'kleenex', 'lemon', 'lightbulb', 'lime', 'marker', 'mushroom', 'notebook', 'onion', 'orange', 'peach',
             'pear', 'pitcher', 'plate', 'pliers', 'potato', 'rubber_eraser', 'scissors', 'shampoo', 'soda_can',
             'sponge', 'stapler', 'tomato', 'toothbrush', 'toothpaste', 'water_bottle']

def readRGBDData(filePath, testinstance_ids, classnum):
    RGBpool = []
    Dpool = []
    labelpool = []
    index = 0
    instancenum = 0
    for classname in rgbdclass:
        instancelist = os.listdir(filePath + '/' + classname)
        instancelist.sort(key=lambda x: int(x[len(classname) + 1:]))
        for instancename in instancelist:
            RGBtemp = []
            Dtemp = []
            labeltemp = []
            haselement = False
            for filename in os.listdir(filePath + '/' + classname + '/' + instancename):
                image_path = filePath + '/' + classname + '/' + instancename + '/' + filename
                imageclassjudge = filename[-9]
                # _ rgb,h d,k mask
                if imageclassjudge == '_':
                    filename_base = filename[:-8]
                    filename_depth = filename_base + 'depthcrop.png'
                    if (os.path.exists(image_path))&(os.path.exists(filePath + '/' + classname + '/' + instancename + '/' + filename_depth)):
                        #img_rgb = Image.open(image_path).convert("RGB")
                        img_rgb = Image.open(image_path)
                        img_rgb = img_rgb.resize((112, 112))
                        img_rgb = np.array(img_rgb)
                        img_rgb = np.reshape(img_rgb, (1,) + img_rgb.shape)
                        img_d = Image.open(filePath + '/' + classname + '/' + instancename + '/' + filename_depth)
                        img_d = img_d.resize((112, 112))
                        img_d = np.array(img_d)
                        img_d = np.reshape(img_d, (1,) + img_d.shape)
                        img_d = np.reshape(img_d, img_d.shape + (1,))
                        img_d = np.tile(img_d, [1, 1, 1, 3])
                        if haselement :
                            labeltemptemp = np.zeros((1, classnum))
                            labeltemptemp[0, index] = 1
                            labeltemp = np.vstack([labeltemp, labeltemptemp])
                            RGBtemp = np.vstack([RGBtemp, img_rgb])
                            Dtemp = np.vstack([Dtemp, img_d])
                        else:
                            labeltemp = np.zeros((1, classnum))
                            labeltemp[0, index] = 1
                            RGBtemp = img_rgb
                            Dtemp = img_d
                            haselement = True

            RGBpool.append(RGBtemp)
            Dpool.append(Dtemp)
            labelpool.append(labeltemp)
            instancenum += 1
        index += 1

    totalinstanceids = [i for i in range(instancenum)]
    traininstanceids = set(totalinstanceids).difference(set(testinstance_ids))
    RGBtrain = np.vstack(RGBpool[i] for i in traininstanceids)
    Dtrain = np.vstack(Dpool[i] for i in traininstanceids)
    RGBDtrain_label = np.vstack(labelpool[i] for i in traininstanceids)
    RGBtest = np.vstack(RGBpool[i] for i in testinstance_ids)
    Dtest = np.vstack(Dpool[i] for i in testinstance_ids)
    RGBDtest_label = np.vstack(labelpool[i] for i in testinstance_ids)


    return RGBtrain, Dtrain, RGBtest, Dtest, RGBDtrain_label, RGBDtest_label

def readRGBDData_sn(filePath, testinstance_ids, classnum):
    RGBpool = []
    Dpool = []
    labelpool = []
    index = 0
    instancenum = 0
    imgsize = 112 #112
    for classname in rgbdclass:
        instancelist = os.listdir(filePath + '/' + classname)
        instancelist.sort(key=lambda x: int(x[len(classname) + 1:]))
        for instancename in instancelist:
            RGBtemp = []
            Dtemp = []
            labeltemp = []
            haselement = False
            for filename in os.listdir(filePath + '/' + classname + '/' + instancename):
                image_path = filePath + '/' + classname + '/' + instancename + '/' + filename
                imageclassjudge = filename[-9]
                # _ rgb,h d,k mask
                if imageclassjudge == '_':
                    filename_base = filename[:-8]
                    filename_depth = filename_base + 'depthsn.png'
                    if (os.path.exists(image_path))&(os.path.exists(filePath + '/' + classname + '/' + instancename + '/' + filename_depth)):
                        #img_rgb = Image.open(image_path).convert("RGB")
                        img_rgb = Image.open(image_path)
                        if img_rgb.height < img_rgb.width:
                            heightnew = max(round(img_rgb.height * imgsize / img_rgb.width), 1)
                            img_rgb = img_rgb.resize((imgsize, heightnew))
                            p1 = 0
                            p2 = round((imgsize - heightnew) / 2)
                            p3 = imgsize
                            p4 = p2 + heightnew
                        else:
                            widthnew = max(round(img_rgb.width * imgsize / img_rgb.height), 1)
                            img_rgb = img_rgb.resize((widthnew, imgsize))
                            p1 = round((imgsize - widthnew) / 2)
                            p2 = 0
                            p3 = p1 + widthnew
                            p4 = imgsize
                        imout = Image.new("RGB", (imgsize, imgsize))
                        imout.paste(img_rgb, (p1, p2, p3, p4))
                        img_rgb = np.array(imout)
                        img_rgb = np.reshape(img_rgb, (1,) + img_rgb.shape)
                        img_d = Image.open(filePath + '/' + classname + '/' + instancename + '/' + filename_depth)
                        if img_d.height < img_d.width:
                            heightnew = max(round(img_d.height * imgsize / img_d.width), 1)
                            img_d = img_d.resize((imgsize, heightnew))
                            p1 = 0
                            p2 = round((imgsize - heightnew) / 2)
                            p3 = imgsize
                            p4 = p2 + heightnew
                        else:
                            widthnew = max(round(img_d.width * imgsize / img_d.height), 1)
                            img_d = img_d.resize((widthnew, imgsize))
                            p1 = round((imgsize - widthnew) / 2)
                            p2 = 0
                            p3 = p1 + widthnew
                            p4 = imgsize
                        imout = Image.new("RGB", (imgsize, imgsize), "#8080FF")
                        imout.paste(img_d, (p1, p2, p3, p4))
                        img_d = np.array(imout)
                        img_d = np.reshape(img_d, (1,) + img_d.shape)
                        if haselement :
                            labeltemptemp = np.zeros((1, classnum))
                            labeltemptemp[0, index] = 1
                            labeltemp = np.vstack([labeltemp, labeltemptemp])
                            RGBtemp = np.vstack([RGBtemp, img_rgb])
                            Dtemp = np.vstack([Dtemp, img_d])
                        else:
                            labeltemp = np.zeros((1, classnum))
                            labeltemp[0, index] = 1
                            RGBtemp = img_rgb
                            Dtemp = img_d
                            haselement = True

            RGBpool.append(RGBtemp)
            Dpool.append(Dtemp)
            labelpool.append(labeltemp)
            instancenum += 1
        index += 1

    totalinstanceids = [i for i in range(instancenum)]
    traininstanceids = set(totalinstanceids).difference(set(testinstance_ids))
    RGBtrain = np.vstack(RGBpool[i] for i in traininstanceids)
    Dtrain = np.vstack(Dpool[i] for i in traininstanceids)
    RGBDtrain_label = np.vstack(labelpool[i] for i in traininstanceids)
    RGBtest = np.vstack(RGBpool[i] for i in testinstance_ids)
    Dtest = np.vstack(Dpool[i] for i in testinstance_ids)
    RGBDtest_label = np.vstack(labelpool[i] for i in testinstance_ids)


    return RGBtrain, Dtrain, RGBtest, Dtest, RGBDtrain_label, RGBDtest_label


def readRGBDData_sn_perclass(filePath, testinstance_ids, classnum):
    RGBpool = []
    Dpool = []
    labelpool = []
    index = 0
    instancenum = 0
    imgsize = 112 #112
    for classname in rgbdclass:
        instancelist = os.listdir(filePath + '/' + classname)
        instancelist.sort(key=lambda x: int(x[len(classname) + 1:]))
        for instancename in instancelist:
            RGBtemp = []
            Dtemp = []
            labeltemp = []
            haselement = False
            for filename in os.listdir(filePath + '/' + classname + '/' + instancename):
                image_path = filePath + '/' + classname + '/' + instancename + '/' + filename
                imageclassjudge = filename[-9]
                # _ rgb,h d,k mask
                if imageclassjudge == '_':
                    filename_base = filename[:-8]
                    filename_depth = filename_base + 'depthsn.png'
                    if (os.path.exists(image_path))&(os.path.exists(filePath + '/' + classname + '/' + instancename + '/' + filename_depth)):
                        #img_rgb = Image.open(image_path).convert("RGB")
                        img_rgb = Image.open(image_path)
                        if img_rgb.height < img_rgb.width:
                            heightnew = max(round(img_rgb.height * imgsize / img_rgb.width), 1)
                            img_rgb = img_rgb.resize((imgsize, heightnew))
                            p1 = 0
                            p2 = round((imgsize - heightnew) / 2)
                            p3 = imgsize
                            p4 = p2 + heightnew
                        else:
                            widthnew = max(round(img_rgb.width * imgsize / img_rgb.height), 1)
                            img_rgb = img_rgb.resize((widthnew, imgsize))
                            p1 = round((imgsize - widthnew) / 2)
                            p2 = 0
                            p3 = p1 + widthnew
                            p4 = imgsize
                        imout = Image.new("RGB", (imgsize, imgsize))
                        imout.paste(img_rgb, (p1, p2, p3, p4))
                        img_rgb = np.array(imout)
                        img_rgb = np.reshape(img_rgb, (1,) + img_rgb.shape)
                        img_d = Image.open(filePath + '/' + classname + '/' + instancename + '/' + filename_depth)
                        if img_d.height < img_d.width:
                            heightnew = max(round(img_d.height * imgsize / img_d.width), 1)
                            img_d = img_d.resize((imgsize, heightnew))
                            p1 = 0
                            p2 = round((imgsize - heightnew) / 2)
                            p3 = imgsize
                            p4 = p2 + heightnew
                        else:
                            widthnew = max(round(img_d.width * imgsize / img_d.height), 1)
                            img_d = img_d.resize((widthnew, imgsize))
                            p1 = round((imgsize - widthnew) / 2)
                            p2 = 0
                            p3 = p1 + widthnew
                            p4 = imgsize
                        imout = Image.new("RGB", (imgsize, imgsize), "#8080FF")
                        imout.paste(img_d, (p1, p2, p3, p4))
                        img_d = np.array(imout)
                        img_d = np.reshape(img_d, (1,) + img_d.shape)
                        if haselement :
                            labeltemptemp = np.zeros((1, classnum))
                            labeltemptemp[0, index] = 1
                            labeltemp = np.vstack([labeltemp, labeltemptemp])
                            RGBtemp = np.vstack([RGBtemp, img_rgb])
                            Dtemp = np.vstack([Dtemp, img_d])
                        else:
                            labeltemp = np.zeros((1, classnum))
                            labeltemp[0, index] = 1
                            RGBtemp = img_rgb
                            Dtemp = img_d
                            haselement = True

            RGBpool.append(RGBtemp)
            Dpool.append(Dtemp)
            labelpool.append(labeltemp)
            instancenum += 1
        index += 1

    totalinstanceids = [i for i in range(instancenum)]
    traininstanceids = set(totalinstanceids).difference(set(testinstance_ids))
    RGBtrain = np.vstack(RGBpool[i] for i in traininstanceids)
    Dtrain = np.vstack(Dpool[i] for i in traininstanceids)
    RGBDtrain_label = np.vstack(labelpool[i] for i in traininstanceids)

    RGBtestpool = []
    Dtestpool = []
    labeltestpool = []

    for i in testinstance_ids:
        RGBtestpool.append(RGBpool[i])
        Dtestpool.append(Dpool[i])
        labeltestpool.append(labelpool[i])



    return RGBtrain, Dtrain, RGBDtrain_label, RGBtestpool, Dtestpool, labeltestpool


def readRGBDData_tsne(filePath, testinstance_ids, tsnenum):
    RGBpool = []
    Dpool = []
    labelpool = []
    index = -1
    instancenum = 0
    imgsize = 112 #112
    instanceindex = 0
    currentinstance = testinstance_ids[instanceindex]
    #tsneclassset = ['apple', 'banana', 'cell_phone', 'greens', 'scissors', 'shampoo', 'soda_can']
    for classname in rgbdclass: #tsneclassset:
        instancelist = os.listdir(filePath + '/' + classname)
        instancelist.sort(key=lambda x: int(x[len(classname) + 1:]))
        for instancename in instancelist:
            if instancenum < currentinstance:
                instancenum += 1
                continue
            instanceindex += 1
            if instanceindex > tsnenum:
                break
            currentinstance = testinstance_ids[instanceindex]
            index += 1
            RGBtemp = []
            Dtemp = []
            labeltemp = []
            haselement = False
            count = 10
            for filename in os.listdir(filePath + '/' + classname + '/' + instancename):
                image_path = filePath + '/' + classname + '/' + instancename + '/' + filename
                imageclassjudge = filename[-9]
                # _ rgb,h d,k mask
                if imageclassjudge == '_':
                    filename_base = filename[:-8]
                    filename_depth = filename_base + 'depthsn.png'
                    if (os.path.exists(image_path))&(os.path.exists(filePath + '/' + classname + '/' + instancename + '/' + filename_depth)):
                        #img_rgb = Image.open(image_path).convert("RGB")
                        if count < 10:
                            count += 1
                            continue
                        count = 1
                        img_rgb = Image.open(image_path)
                        if img_rgb.height < img_rgb.width:
                            heightnew = max(round(img_rgb.height * imgsize / img_rgb.width), 1)
                            img_rgb = img_rgb.resize((imgsize, heightnew))
                            p1 = 0
                            p2 = round((imgsize - heightnew) / 2)
                            p3 = imgsize
                            p4 = p2 + heightnew
                        else:
                            widthnew = max(round(img_rgb.width * imgsize / img_rgb.height), 1)
                            img_rgb = img_rgb.resize((widthnew, imgsize))
                            p1 = round((imgsize - widthnew) / 2)
                            p2 = 0
                            p3 = p1 + widthnew
                            p4 = imgsize
                        imout = Image.new("RGB", (imgsize, imgsize))
                        imout.paste(img_rgb, (p1, p2, p3, p4))
                        img_rgb = np.array(imout)
                        img_rgb = np.reshape(img_rgb, (1,) + img_rgb.shape)
                        img_d = Image.open(filePath + '/' + classname + '/' + instancename + '/' + filename_depth)
                        if img_d.height < img_d.width:
                            heightnew = max(round(img_d.height * imgsize / img_d.width), 1)
                            img_d = img_d.resize((imgsize, heightnew))
                            p1 = 0
                            p2 = round((imgsize - heightnew) / 2)
                            p3 = imgsize
                            p4 = p2 + heightnew
                        else:
                            widthnew = max(round(img_d.width * imgsize / img_d.height), 1)
                            img_d = img_d.resize((widthnew, imgsize))
                            p1 = round((imgsize - widthnew) / 2)
                            p2 = 0
                            p3 = p1 + widthnew
                            p4 = imgsize
                        imout = Image.new("RGB", (imgsize, imgsize), "#8080FF")
                        imout.paste(img_d, (p1, p2, p3, p4))
                        img_d = np.array(imout)
                        img_d = np.reshape(img_d, (1,) + img_d.shape)
                        if haselement :
                            labeltemptemp = np.zeros((1, tsnenum))
                            labeltemptemp[0, index] = 1
                            labeltemp = np.vstack([labeltemp, labeltemptemp])
                            RGBtemp = np.vstack([RGBtemp, img_rgb])
                            Dtemp = np.vstack([Dtemp, img_d])
                        else:
                            labeltemp = np.zeros((1, tsnenum))
                            labeltemp[0, index] = 1
                            RGBtemp = img_rgb
                            Dtemp = img_d
                            haselement = True

            RGBpool.append(RGBtemp)
            Dpool.append(Dtemp)
            labelpool.append(labeltemp)
            instancenum += 1


    RGBtsne = np.vstack(RGBpool[i] for i in range(tsnenum))
    Dtsne = np.vstack(Dpool[i] for i in range(tsnenum))
    RGBDtsne_label = np.vstack(labelpool[i] for i in range(tsnenum))

    return RGBtsne, Dtsne, RGBDtsne_label

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
    datasetpath = "./rgbd-dataset_eval"
    classnum = 51
    # Loading Data
    print("LOADING RGB-D DATASET")
    testinstance_ids = [2,5,15,21,24,29,31,37,39,44,50,54,61,67,73,76,88,106,111,118,122,132,135,139,145,151,159,164,169,173,175,185,189,195,201,204,212,213,217,223,227,235,240,243,248,257,270,280,283,290,292]  # 10 settings according to article
    RGBtrain, Dtrain, RGBtest, Dtest, RGBDtrain_label, RGBDtest_label = readRGBDData(datasetpath, testinstance_ids,
                                                                                     classnum)
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

