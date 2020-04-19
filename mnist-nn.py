## pip3 install Keras
## pip3 install Tensorflow


from keras.datasets import minst   # one of keras dataset
from keras import models
from keras import layers
from keras.utils import to_categorical
# split data in test and train
(train_images, train_labels),(test_images, test_labels) = minist.load_data


network = models.Sequential()
network.add(layers.Dense(784,activation ='relu' , input_shape(28*28,)))  # 784 =28*28  with relu activation function
network.add(layes.Dense(784,activation ='relu' , input_shape(28*28,)))
network.add(layers.Dense(10, activation = 'softmax'))   # 10 possible categories (number 0-9)
network.compile(optimizer ='adam', loss= 'categorical_crossentropy', metrics = ['accuracy']

# reshape our data wuth [60000] images for train and [10000] for test
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# categorical encoding to turns a number of features in numerical representations
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# train the model
network.fit(train_images, train_labels, epochs=5, batch_size=128)









