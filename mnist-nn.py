{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.datasets import mnist   # one of keras dataset\n",
    "from keras import models\n",
    "from keras import layers\n",
    "from keras.utils import to_categorical\n",
    "# split data in test and train\n",
    "(train_images, train_labels), (test_images, test_labels) = mnist.load_data()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "network = models.Sequential()  # use sequential model\n",
    "network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))    # activation function is relu \n",
    "network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))  # [28 * 28] refers to the image’s pixel width and height\n",
    "network.add(layers.Dense(10, activation='softmax'))   # 10 as we want to have at the end ten categories, 0-9\n",
    "network.compile(optimizer='adam',\n",
    "                loss='categorical_crossentropy',\n",
    "                metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_images = train_images.reshape((60000, 28 * 28))\n",
    "train_images = train_images.astype('float32') / 255\n",
    "test_images = test_images.reshape((10000, 28 * 28))  \n",
    "test_images = test_images.astype('float32') / 255"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_labels = to_categorical(train_labels)\n",
    "test_labels = to_categorical(test_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "60000/60000 [==============================] - 6s 98us/step - loss: 0.3242 - accuracy: 0.9037\n",
      "Epoch 2/5\n",
      "60000/60000 [==============================] - 6s 101us/step - loss: 0.1012 - accuracy: 0.9715\n",
      "Epoch 3/5\n",
      "60000/60000 [==============================] - 6s 101us/step - loss: 0.0688 - accuracy: 0.9809\n",
      "Epoch 4/5\n",
      "60000/60000 [==============================] - 6s 99us/step - loss: 0.0521 - accuracy: 0.9851\n",
      "Epoch 5/5\n",
      "60000/60000 [==============================] - 6s 103us/step - loss: 0.0435 - accuracy: 0.9874\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.callbacks.History at 0x7f98988dab50>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "network.fit(train_images, train_labels, epochs=5, batch_size=128)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "10000/10000 [==============================] - 1s 59us/step\n",
      "test_acc: 0.9783999919891357 test_loss 0.08120215325243771\n"
     ]
    }
   ],
   "source": [
    "test_loss, test_acc = network.evaluate(test_images, test_labels)\n",
    "print('test_acc:', test_acc, 'test_loss', test_loss)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
