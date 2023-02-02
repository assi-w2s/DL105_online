import cv2


class SimplePreprocessor:
    def __init__(self, w, h, inter=cv2.INTER_AREA):
        self.width = w
        self.height = h
        self.inter = inter

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)