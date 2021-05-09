import numpy as np
import cv2
import pafy
import time
import pytesseract

class OCR(object):
    input_frame, temp_frame, output_frame = None, None, None
    m, dim = None, None
    ocr_text, ocr_status = None, None


    def __init__(self, experiment):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.m = experiment.data["OCR_m"]
        self.ocr_status = False

    def preprocessor(self, frame):
        self.input_frame = frame
        dim = (self.input_frame.shape[1] * self.m, self.input_frame.shape[0] * self.m)
        self.input_frame = cv2.resize(self.input_frame, dim)
        self.temp_frame = OCR.grayscale(self.input_frame)
        self.temp_frame = OCR.remove_noise(self.temp_frame)
        self.temp_frame = OCR.threshold(self.temp_frame)
        self.temp_frame = OCR.dilate(self.temp_frame)
        self.temp_frame = OCR.erode(self.temp_frame)
        self.temp_frame = OCR.opening(self.temp_frame)

        self.output_frame = self.temp_frame

    def mytesseract(self, frame):
        try:
            self.ocr_text = pytesseract.image_to_string(frame, config='-l eng --oem 1 -c tessedit_char_whitelist=\|/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789 --tessdata-dir .').replace(" ","_").replace("|","_")
            self.ocr_text = self.ocr_text.split('_', 1)
            self.ocr_text[1] = self.ocr_text[1].split()
            self.ocr_text[1] = self.ocr_text[1][1].split('_', 1)
            self.ocr_text = str(self.ocr_text[0]) +'\n'+ str(self.ocr_text[1][0])
            self.ocr_status = True
            return self.ocr_text
        except:
            self.ocr_status = False


    # get grayscale image
    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    def remove_noise(image):
        return cv2.medianBlur(image, 3)

    # thresholding
    def threshold(image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # dilation
    def dilate(image):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    # erosion
    def erode(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    # opening - erosion followed by dilation
    def opening(image):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # canny edge detection
    def canny(image):
        return cv2.Canny(image, 10, 200)

    # skew correction
    def deskew(image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated