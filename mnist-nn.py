from keras.datasets import mnist   # one of keras dataset
from keras import models
from keras import layers
from keras.utils import to_categorical
# split data in test and train
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential()  # use sequential model
network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))    # activation function is relu 
network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))  # [28 * 28] refers to the image’s pixel width and height
network.add(layers.Dense(10, activation='softmax'))   # 10 as we want to have at the end ten categories, 0-9
network.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])train_images = train_images.reshape((60000, 28 * 28))




train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))  
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


network.fit(train_images, train_labels, epochs=5, batch_size=128)


test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc, 'test_loss', test_loss)

