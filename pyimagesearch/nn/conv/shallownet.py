from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Activation, Flatten, Dense
from tensorflow.python.keras import backend as K


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (height, width, width)

        model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model