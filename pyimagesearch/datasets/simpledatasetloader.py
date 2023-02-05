import numpy as np
import cv2
import os


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        #store preprocessors
        self.preprocessors = preprocessors
        # in case no preprocessors, start it as empty list
        if preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # initialize list of features (data) and labels
        data = list()
        labels = list()

        # loop over the input images paths and for each:
        for (i, image_path) in enumerate(imagePaths):
            # (i) read image
            image = cv2.imread(image_path)
            # (ii) read label (separate imagePath and take -2 element which stores class)
            label = image_path.split(os.path.sep)[-2]
            # label = imagePath.split(os.path.sep)[-2]

            # if preprocessors aren't empty, loop each of them and preprocess image
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # append image to data and append label to labels
            data.append(image)
            labels.append(label)

            # in case verbos != 0 print the {}/{} how many images were processed
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i+1, len(imagePaths)))
            # return an np array of data and labels

        return (np.array(data), np.array(labels))