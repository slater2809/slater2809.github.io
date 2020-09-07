---
title: "Digit Recognizer competiton by Kaggle Attempt"
date: 2020-09-06
tags: [mnist, convolutional neural network, tensorflow, keras, digit recognizer, python]
header:
excerpt: "Digit Recognizer"
mathjax: "true"
---

I recently completed the Deep Learning course by Kaggle. The course contained an introduction to convolutional neural networks, with application to object detection. I learnt about transfer learning, how to build an image classification model from scratch, and how to improve computational efficiency and accuracy using dropout and different amounts of strides.

In ths article, I look to apply what I learnt to the Digit Recognizer challenge by Kaggle. The problem entails classifying 28x28 pixel images of hand written digits (0 to 9) from the MNIST dataset. Since the problem is quite specific, I will be building my own image classification models from scratch and I will be testing if the use of a higher than defualt values of strides, and dropout has an effect on the accuracy of my model.

Let's begin by importing the necessary libraries.
```
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import random

from sklearn.model_selection import train_test_split

from google.colab import files

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
```

Next, we import and explore the training set.


```
# Import the training set
train = np.loadtxt("train.csv", skiprows=1, delimiter=',')

# Examine the shape of the training set
train.shape
```




    (42000, 785)



We see that the traning set contans 42000 labelled images.


```
# Peek at the training set
train[0:10, :]
```




    array([[1., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [1., 0., 0., ..., 0., 0., 0.],
           ...,
           [3., 0., 0., ..., 0., 0., 0.],
           [5., 0., 0., ..., 0., 0., 0.],
           [3., 0., 0., ..., 0., 0., 0.]])



 Now, I look at what the first image in the training set looks like, along with its associated label.


```
# Display the first image in the training set
first_image = train[0, 1:]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
```

<img src="{{ site.url }}{{ site.baseurl }}/images/digit recognizer/digit_recognizer_7_0.png" alt="digit">


```
# Display the label
train[0, 0]
```




    1.0



The first training image is of a hand-written 1. From the description of the problem on Kaggle, I know that each image is 28x28 pixels, and each pixel has a single pixel-value associated with it between 0 and 255. Since normalising the inputs to any neural network to be between 0 and 1 is recommended, I preprocess the data to achieve this goal.


```
img_rows, img_cols = 28, 28
num_classes = 10

# Prep the data
def prep_data(raw):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:, 1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

X, y = prep_data(train)
```

We see that the training set is now a tensor of dimension (42000, 28, 28, 1), i.e. 42000 28x28 matrices of normalised pixel-values. The response has been coded as a binary class matrix. Let's go on to splitting the training set into a training and validation set. 


```
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.25, random_state=0)

# Check the split
X_train.shape
```




    (31500, 28, 28, 1)




```
y_train.shape
```




    (31500, 10)




```
X_validate.shape
```




    (10500, 28, 28, 1)




```
y_validate.shape
```




    (10500, 10)



We see that the new training set has 31500 images (75% of the training data), and the validation set has 10500 images (25% of the training data).

Now for the fun bit. I am going to start with a simple feedforward neural network for classification. This will serve as my baseline model to which I will compare more complex models. For the baseline model, I flatten the input to a one-dimensional array. I have a single hidden layer with 128 neurons.


```
baseline_model = Sequential()

# Flatten the input tensor to a 2D matrix
baseline_model.add(Flatten(input_shape=(28, 28, 1)))

# Add the hidden layer
baseline_model.add(Dense(128, activation="relu"))

# Add the output layer
baseline_model.add(Dense(num_classes, activation="softmax"))

# View the model summary
baseline_model.summary()
```

    Model: "sequential_7"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_6 (Flatten)          (None, 784)               0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 128)               100480    
    _________________________________________________________________
    dense_13 (Dense)             (None, 10)                1290      
    =================================================================
    Total params: 101,770
    Trainable params: 101,770
    Non-trainable params: 0
    _________________________________________________________________
    

The baseline model has 101 770 trainable parameters ('simple' is relative here...). We now go on to compile the model using log loss as the loss function and the Adam optimizer as the optimizer.


```
baseline_model.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics = ['accuracy'])
```

Now for training. I used Google Colab to train all models with the GPU accelarator.


```
random.seed(0)
baseline_model.fit(X_train, y_train, validation_data=(X_validate, y_validate), epochs=10)
```

    Epoch 1/10
    985/985 [==============================] - 3s 3ms/step - loss: 0.3355 - accuracy: 0.9044 - val_loss: 0.2038 - val_accuracy: 0.9428
    Epoch 2/10
    985/985 [==============================] - 3s 3ms/step - loss: 0.1498 - accuracy: 0.9568 - val_loss: 0.1501 - val_accuracy: 0.9587
    Epoch 3/10
    985/985 [==============================] - 3s 3ms/step - loss: 0.1028 - accuracy: 0.9690 - val_loss: 0.1262 - val_accuracy: 0.9635
    Epoch 4/10
    985/985 [==============================] - 3s 3ms/step - loss: 0.0769 - accuracy: 0.9777 - val_loss: 0.1220 - val_accuracy: 0.9651
    Epoch 5/10
    985/985 [==============================] - 3s 3ms/step - loss: 0.0580 - accuracy: 0.9829 - val_loss: 0.1039 - val_accuracy: 0.9689
    Epoch 6/10
    985/985 [==============================] - 3s 3ms/step - loss: 0.0459 - accuracy: 0.9851 - val_loss: 0.1023 - val_accuracy: 0.9712
    Epoch 7/10
    985/985 [==============================] - 3s 3ms/step - loss: 0.0348 - accuracy: 0.9896 - val_loss: 0.1090 - val_accuracy: 0.9698
    Epoch 8/10
    985/985 [==============================] - 3s 3ms/step - loss: 0.0283 - accuracy: 0.9920 - val_loss: 0.1103 - val_accuracy: 0.9697
    Epoch 9/10
    985/985 [==============================] - 3s 3ms/step - loss: 0.0230 - accuracy: 0.9935 - val_loss: 0.1146 - val_accuracy: 0.9694
    Epoch 10/10
    985/985 [==============================] - 3s 3ms/step - loss: 0.0176 - accuracy: 0.9954 - val_loss: 0.1123 - val_accuracy: 0.9718
    




    <tensorflow.python.keras.callbacks.History at 0x7f282dfeb668>



Now, we obtain the accuracy of the  baseline model on the validation set.


```
validation_loss, validation_acc = baseline_model.evaluate(X_validate,  y_validate, verbose=2)

print('\nTest accuracy:', validation_acc)
```

    329/329 - 1s - loss: 0.1123 - accuracy: 0.9718
    
    Test accuracy: 0.9718095064163208
    

We see that the baseline model obtained an accuracy of 0.97 on the validation set. This accuracy may seem high but a 3% missclassfication rate can result in dire consequences, depending on the application of the model. Therefore, let's consider a convolutional neural network to see if we can improve on this validation accuracy.


```
# Create the model and add the layers
conv_base_model = Sequential()
conv_base_model.add(Conv2D(12, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
conv_base_model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',))
conv_base_model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',))
# Flatten data coming in from the third convolutional layer into a 2-D tensor
conv_base_model.add(Flatten())
conv_base_model.add(Dense(100, activation='relu'))
conv_base_model.add(Dense(num_classes, activation='softmax'))

# View the model summary
conv_base_model.summary()
```

    Model: "sequential_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_9 (Conv2D)            (None, 26, 26, 12)        120       
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 24, 24, 20)        2180      
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 22, 22, 20)        3620      
    _________________________________________________________________
    flatten_4 (Flatten)          (None, 9680)              0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 100)               968100    
    _________________________________________________________________
    dense_9 (Dense)              (None, 10)                1010      
    =================================================================
    Total params: 975,030
    Trainable params: 975,030
    Non-trainable params: 0
    _________________________________________________________________
    

There are now 975 030 trainable parameters, which is 9 times more than our baseline model. Let's see if this increase in parameter count makes a difference.

We now compile and train the model using the same loss function and optimizer as the baseline model.


```
conv_base_model.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics = ['accuracy'])

conv_base_model.fit(X_train, y_train, batch_size=100, epochs=10)
```

    Epoch 1/10
    315/315 [==============================] - 1s 4ms/step - loss: 0.2467 - accuracy: 0.9241
    Epoch 2/10
    315/315 [==============================] - 1s 4ms/step - loss: 0.0642 - accuracy: 0.9802
    Epoch 3/10
    315/315 [==============================] - 1s 4ms/step - loss: 0.0406 - accuracy: 0.9875
    Epoch 4/10
    315/315 [==============================] - 1s 4ms/step - loss: 0.0289 - accuracy: 0.9907
    Epoch 5/10
    315/315 [==============================] - 1s 4ms/step - loss: 0.0194 - accuracy: 0.9939
    Epoch 6/10
    315/315 [==============================] - 1s 4ms/step - loss: 0.0158 - accuracy: 0.9947
    Epoch 7/10
    315/315 [==============================] - 1s 4ms/step - loss: 0.0123 - accuracy: 0.9962
    Epoch 8/10
    315/315 [==============================] - 1s 4ms/step - loss: 0.0095 - accuracy: 0.9970
    Epoch 9/10
    315/315 [==============================] - 1s 4ms/step - loss: 0.0064 - accuracy: 0.9977
    Epoch 10/10
    315/315 [==============================] - 1s 4ms/step - loss: 0.0076 - accuracy: 0.9976
    




    <tensorflow.python.keras.callbacks.History at 0x7f282d706b70>




```
validation_loss, validation_acc = conv_base_model.evaluate(X_validate,  y_validate, verbose=2)

print('\nTest accuracy:', validation_acc)
```

    329/329 - 1s - loss: 0.0706 - accuracy: 0.9850
    
    Test accuracy: 0.9850476384162903
    

We see from the above output that the convolutional neural network took slightly longer to train than the baseline model. This can be attributed to the substantial increase in the number of trainable parameters.

The validation set accuracy is now 0.99. We now attempt to make use of higher strides and dropout to speed up the training of the convolutional neural network, while maintaining the same level of validation set accuracy.


```
# Create the model and add the layers
conv_model = Sequential()
conv_model.add(Conv2D(12, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
conv_model.add(Conv2D(20, kernel_size=(3, 3), strides=2,
                 activation='relu',))
conv_model.add(Dropout(0.5))
conv_model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu', strides=2))
conv_model.add(Dropout(0.5))
# Flatten data coming in from the third convolutional layer into a 2-D tensor
conv_model.add(Flatten())
conv_model.add(Dense(100, activation='relu'))
conv_model.add(Dense(num_classes, activation='softmax'))

# View the model summary
conv_model.summary()
```

    Model: "sequential_6"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_12 (Conv2D)           (None, 26, 26, 12)        120       
    _________________________________________________________________
    conv2d_13 (Conv2D)           (None, 12, 12, 20)        2180      
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 12, 12, 20)        0         
    _________________________________________________________________
    conv2d_14 (Conv2D)           (None, 5, 5, 20)          3620      
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 5, 5, 20)          0         
    _________________________________________________________________
    flatten_5 (Flatten)          (None, 500)               0         
    _________________________________________________________________
    dense_10 (Dense)             (None, 100)               50100     
    _________________________________________________________________
    dense_11 (Dense)             (None, 10)                1010      
    =================================================================
    Total params: 57,030
    Trainable params: 57,030
    Non-trainable params: 0
    _________________________________________________________________
    

We see that the number of trainable parameters has decreased substantially to 57 030. Let's train and validate.


```
conv_model.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics = ['accuracy'])

conv_model.fit(X_train, y_train, batch_size=100, epochs=10)
```

    Epoch 1/10
    315/315 [==============================] - 1s 3ms/step - loss: 0.7248 - accuracy: 0.7624
    Epoch 2/10
    315/315 [==============================] - 1s 3ms/step - loss: 0.2825 - accuracy: 0.9138
    Epoch 3/10
    315/315 [==============================] - 1s 3ms/step - loss: 0.2016 - accuracy: 0.9372
    Epoch 4/10
    315/315 [==============================] - 1s 3ms/step - loss: 0.1680 - accuracy: 0.9460
    Epoch 5/10
    315/315 [==============================] - 1s 3ms/step - loss: 0.1529 - accuracy: 0.9525
    Epoch 6/10
    315/315 [==============================] - 1s 3ms/step - loss: 0.1283 - accuracy: 0.9598
    Epoch 7/10
    315/315 [==============================] - 1s 3ms/step - loss: 0.1251 - accuracy: 0.9605
    Epoch 8/10
    315/315 [==============================] - 1s 3ms/step - loss: 0.1141 - accuracy: 0.9637
    Epoch 9/10
    315/315 [==============================] - 1s 3ms/step - loss: 0.1058 - accuracy: 0.9662
    Epoch 10/10
    315/315 [==============================] - 1s 3ms/step - loss: 0.0995 - accuracy: 0.9680
    




    <tensorflow.python.keras.callbacks.History at 0x7f282d2f3ba8>




```
validation_loss, validation_acc = conv_model.evaluate(X_validate,  y_validate, verbose=2)

print('\nTest accuracy:', validation_acc)
```

    329/329 - 1s - loss: 0.0517 - accuracy: 0.9842
    
    Test accuracy: 0.9841904640197754
    

Now, we have been able to obtain a similar validation set accuracy of 0.98 using higher strides and dropout, while speeding up model training. It should be noted that even though the training time difference between the model that makes use of increased strides and dropout, and the model that does not is small in this case, when the problem increases in size, this time difference can become very large.

Since the validation set accuracy is marginally lower for the convolutional neural network that makes use of increased strides and dropout, but the training time is faster, I select conv_model as the model I will use to obtain predictions on the test set.

Now, I retrain conv_model on the full training set and then apply the model to the test set to obtain predictions that I will upload to Kaggle.


```
# Retrain conv_model on the full training set
conv_model.fit(X, y, batch_size=100, epochs=10)
```

    Epoch 1/10
    420/420 [==============================] - 1s 3ms/step - loss: 0.0967 - accuracy: 0.9694
    Epoch 2/10
    420/420 [==============================] - 1s 3ms/step - loss: 0.0899 - accuracy: 0.9721
    Epoch 3/10
    420/420 [==============================] - 1s 3ms/step - loss: 0.0857 - accuracy: 0.9730
    Epoch 4/10
    420/420 [==============================] - 1s 3ms/step - loss: 0.0811 - accuracy: 0.9746
    Epoch 5/10
    420/420 [==============================] - 1s 3ms/step - loss: 0.0756 - accuracy: 0.9757
    Epoch 6/10
    420/420 [==============================] - 1s 3ms/step - loss: 0.0735 - accuracy: 0.9765
    Epoch 7/10
    420/420 [==============================] - 1s 3ms/step - loss: 0.0709 - accuracy: 0.9772
    Epoch 8/10
    420/420 [==============================] - 1s 3ms/step - loss: 0.0714 - accuracy: 0.9769
    Epoch 9/10
    420/420 [==============================] - 1s 3ms/step - loss: 0.0669 - accuracy: 0.9784
    Epoch 10/10
    420/420 [==============================] - 1s 3ms/step - loss: 0.0608 - accuracy: 0.9800
    




    <tensorflow.python.keras.callbacks.History at 0x7f282c6a2fd0>




```
# Import and prep the test set
test = np.loadtxt("test.csv", skiprows=1, delimiter=',')

num_images = test.shape[0]
X_test = test.reshape(num_images, img_rows, img_cols, 1)
X_test = X_test / 255

X_test.shape
```




    (28000, 28, 28, 1)



We see that there are 28000 images in the test set. Let's obtain the predictions on the test set.


```
preds = conv_model.predict(X_test)
preds.shape
```




    (28000, 10)




```

```

These predictions are in the form of arrays of probabilities per case. We would like the category associated with the maximum probablity per case to be the prediction for that case.


```
preds = [np.argmax(preds[i]) for i in range(preds.shape[0])]
```

Finally, let's create the submission file.


```
ImageId = np.arange(1, test.shape[0] + 1)

submission = pd.DataFrame({"ImageId":ImageId, "Label":preds})

submission.to_csv('preds.csv')
files.download('preds.csv')
```


    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>


According to Kaggle, conv_model achieved an accuracy of 0.988 on the test set.
