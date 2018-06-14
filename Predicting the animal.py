import tensorflow as tf
import numpy as np
import os, glob, cv2
import sys, argparse


import tensorflow as tf
import dataset
import time
from datetime import timedelta
import math
import random
import os
import numpy as np

img_size =128
num_input_channels = 3
batch_size = 32
classes = ['dogs' , 'cats']
num_classes = len(classes)

x = tf.placeholder(tf.float32 , shape = [None , img_size , img_size , num_input_channels])
#labels
y_true = tf.placeholder(tf.float32 , shape = [None , num_classes])
y_true_cls = tf.argmax(y_true)

#network parameters:
conv_filter_size_1 = 3
num_filters_conv1 = 32

conv_filter_size_2 = 3
num_filters_conv2 = 32

conv_filter_size_3 = 3
num_filters_conv3 = 32

fc_layer_size = 128

      
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev = 0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05,shape = [size]))




def create_conv_layer(input , num_input_channels , conv_filter_size, num_filters):
    weights = create_weights(shape = [conv_filter_size , conv_filter_size , num_input_channels , num_filters])
    biases = create_biases(num_filters)
    layer = tf.nn.conv2d(input = input ,
                         filter = weights ,
                         strides = [1,1,1,1] ,
                         padding = 'SAME' )
    layer+=biases
    layer = tf.nn.max_pool(value = layer ,
                           ksize = [1,2,2,1],
                           strides = [1,2,2,1] ,
                           padding = 'SAME')
    layer = tf.nn.relu(layer)
    return layer

def create_flatten_layer(layer):
    shape = layer.get_shape()
    print(shape)
    num_features = shape[1:4].num_elements()
    print(num_features)
    layer = tf.reshape(layer,shape = [-1,num_features])
    return layer

def create_fc_layer(input , num_inputs , num_outputs , use_relu = True):
    weights = create_weights(shape = [num_inputs , num_outputs])
    biases = create_biases(num_outputs)
    layer = tf.matmul(input , weights)
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer




#creating convolutional layers for the network inshort network building
# layer 1
layer_conv1 = create_conv_layer(input = x,
                                num_input_channels = num_input_channels ,
                                conv_filter_size = conv_filter_size_1,
                                num_filters = num_filters_conv1)
# layer 2
layer_conv2 = create_conv_layer(input = layer_conv1,
                                num_input_channels = num_filters_conv1 ,
                                conv_filter_size = conv_filter_size_2,
                                num_filters = num_filters_conv2)
# layer 3 
layer_conv3 = create_conv_layer(input = layer_conv2,
                                num_input_channels = num_filters_conv2 ,
                                conv_filter_size = conv_filter_size_3,
                                num_filters = num_filters_conv3)
#flatten layer
layer_flat = create_flatten_layer(layer_conv3)

#fully connected layers
#layer layer_fc1
layer_fc1  =create_fc_layer(input = layer_flat ,
                            num_inputs = layer_flat.get_shape()[1:4].num_elements()  ,
                            num_outputs = fc_layer_size ,
                            use_relu = False)
#layer_fc_2
layer_fc2  =create_fc_layer(input = layer_fc1 ,
                            num_inputs = fc_layer_size ,
                            num_outputs = num_classes ,
                            use_relu = False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred[0] )

# optimizing the cost of the result

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = layer_fc2 ,
                                                        labels = y_true)

cost = tf.reduce_mean(cross_entropy)
#optimization function

optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
#sess.run(tf.global_variables_initializer())

def show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss):
      acc = sess.run(accuracy, feed_dict = feed_dict_tr)
      val_acc = sess.run(accuracy, feed_dict = feed_dict_val)
      msg = "Training Epoch {0}---Training accuracy: {1:>6.1%} , Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
      print(msg.format(epoch+1, acc, val_acc, val_loss))

      
tot_iterations = 0

saver = tf.train.Saver()

def train(num_iteration):
    global tot_iterations

    #adding seed so that random initialization is consistent
    from numpy.random import seed
    seed(1)
    from tensorflow import set_random_seed
    set_random_seed(2)
    classes = ['dogs' , 'cats']
    num_classes = len(classes)

    train_path = 'training_data'
    # 20% of the data will automatically be used for valiadtion
    validation_size = 0.2
    img_size =128
    num_input_channels = 3
    batch_size = 32

    #this is line is left blank to insert code for the reading the data from the file
    data = dataset.read_train_sets(train_path, img_size, classes, validation_size = validation_size)

    print("completed reading the data. Now will print snippet of the traing\n")
    print("Num of files in Training set:\t\t{}".format(len(data.train.labels)))
    print("Num of files in Validation set:\t\t{}".format(len(data.valid.labels)))
    print("traing started...!!")
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    for i in range(tot_iterations,tot_iterations+num_iteration):
          x_batch, y_true_batch,  _, cls_batch = data.train.next_batch(batch_size)
          x_valid_batch, y_valid_true, _, cls_valid_batch = data.valid.next_batch(batch_size)

          feed_dict_tr = {x: x_batch, y_true: y_true_batch}
          feed_dict_val = {x: x_valid_batch, y_true: y_valid_true}

          sess.run(optimizer, feed_dict = feed_dict_tr)

          if i >3 and i%int(data.train.num_examples/batch_size) == 0:
              val_loss = sess.run(cost, feed_dict = feed_dict_val)
              epoch = int(i/int(data.train.num_examples/batch_size))
              print("progress going on ",i," out of ",tot_iterations+num_iteration,"\n")
              show_progress(epoch = epoch, feed_dict_tr = feed_dict_tr, feed_dict_val = feed_dict_val, val_loss = val_loss )
              saver.save(sess, os.path.join(os.getcwd(),'dogs-cats-model'))
    tot_iterations+=num_iterations


def predict():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print("name of the image for the prediction: \n")
    image_path = input()
    file_name = dir_path+'/'+image_path
    image_size = 128
    images = []

    #reading the image using openCV

    image = cv2.imread(file_name)

    #Resizing the image to our desired  size and preprocessing will be done as exactly done in training
     
    image = cv2.resize(image, (image_size, image_size),0,0,cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)

    #The image to the network is of the shape [None imagesize, imagesize, num_input_channels ]. Hence we reshape.
    x_batch = images.reshape(1, image_size, image_size, num_input_channels)
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    y_test_image = np.zeros((1,2))
    feed_dict_pred= {x: x_batch, y_true: y_test_image}
    result = sess.run(y_pred_cls, feed_dict = feed_dict_pred)
    print('prediction: ',result)
def choose_the_option(train1):
    if train1:
        train(num_iteration = 3000)
        print("thanks man finished training...!!!")
    else:
        while(True):
            print("Do you want to continue yes y else n")
            ch= input()
            if(ch == "N"):
                break
            predict()

def main():
    print("we started in main....yuh!!")
    choose_the_option(train1=False)
    
if __name__=="__main__":
    main()
   

          










