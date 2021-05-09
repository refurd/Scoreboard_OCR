import numpy as np
import cv2


class Separator(object):
    max_iter, cur_iter = None, None
    input_frame, temp_frame, output_frame = None, None, None
    lim = None
    scoreboard = None
    contours, hierarchy, area_min, area_max = None, None, None, None
    x, y, w, h = None, None, None, None
    find_status, crop_status = None, None

    def __init__(self, experiment):
        self.max_iter = experiment.data["search_iter"]
        self.area_min = experiment.data["area_min"]
        self.area_max = experiment.data["area_max"]
        self.cur_iter = 0
        self.find_status = True
        self.crop_status = False

    def find_scoreboard(self, frame):
        # if current iteration less than the max iteration
        # motion recording in progress, find_status = True
        if self.cur_iter < self.max_iter:
            self.input_frame = frame
            if self.cur_iter == 0:
                self.temp_frame = self.input_frame
                self.output_frame = cv2.cvtColor(self.input_frame, cv2.COLOR_BGR2GRAY)
                self.output_frame = 0
            else:
                self.temp_frame = cv2.cvtColor(cv2.absdiff(self.temp_frame, self.input_frame), cv2.COLOR_BGR2GRAY)
                self.lim, self.temp_frame = cv2.threshold(self.temp_frame, 20, 255, cv2.THRESH_BINARY)
                self.output_frame = self.output_frame + self.temp_frame
                self.temp_frame = self.input_frame

            self.cur_iter = self.cur_iter + 1
            self.find_status = True
        # if max iteration less than the current iteration
        # motion recording inactive, find_status = False
        else:
            self.find_status = False
            self.crop_status = True

    def crop_scoreboard(self, frame):
        try:
            # threshold
            self.lim, self.temp_frame = cv2.threshold(self.output_frame, 200, 255, 0)
            # contours
            self.contours, self.hierarchy = cv2.findContours(self.temp_frame, 1, 2)
            cnt = self.contours[0]
            self.temp_frame = cv2.cvtColor(self.output_frame, cv2.COLOR_GRAY2RGB)
            for c in self.contours:
                area = cv2.contourArea(c)
                cv2.drawContours(self.temp_frame, [c], -1, (0, 255, 0), 3)
                if self.area_min < area < self.area_max:
                    cv2.drawContours(self.temp_frame, [c], -1, (0, 0, 255), 2)
                    # bounding boxes
                    self.x, self.y, self.w, self.h = cv2.boundingRect(c)
                    cv2.rectangle(self.temp_frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 0, 255), 2)
            # cropping
            self.temp_frame = frame[self.y:self.y + self.h, self.x:self.x + self.w]
            # preprocessing
            self.crop_status = False
        except:
            self.crop_status = False
            self.cur_iter = 0
            self.find_status = True
