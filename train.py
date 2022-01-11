from googlenet import MiniGoogLeNet
from keras.datasets import fashion_mnist
import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
trainX = trainX.astype('float')/255.0
testX = testX.astype('float')/255.0

#convert the label
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

optimization = SGD(learning_rate=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniGoogLeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=optimization, metrics=['accuracy'])

history = model.fit(trainX, trainY, validation_data=(testX, testY),
                    batch_size=64, epochs=40, verbose=2)

predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=labelNames))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), history.history["loss"], label="train_loss")


plt.plot(np.arange(0, 40), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on fashion-mnist")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])







