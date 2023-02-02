from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor, SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import ShallowNet
from tensorflow.python.keras.optimizers.gradient_descent_v2 import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset_path', required=True, help="path to input images")
args = vars(ap)

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset_path"]))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

(trainX, trainY, testX, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testX = LabelBinarizer().fit_transform(testX)

print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32,
                         height=32,
                         depth=3,
                         classes=3)
model.compile(loss="categorical_crossentropy",
              optimizer=opt)

print("[INFO] training network...")
H = model.fit(trainX,
              trainY,
              validation_data=(testX, testY),
              batch_size=32,
              epochs=100,
              verbose=1)

print("[INFO] evaluating network")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], labe="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], labe="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], labe="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], labe="val_acc")
plt.title("training loss and accuracy")
plt.xlabel("epochs")
plt.ylabel("loss/accuracy")
plt.legend()
plt.show()

