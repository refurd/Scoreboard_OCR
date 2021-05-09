import cv2
import pafy

class Stream(object):

    cap = None
    url = None

    def __init__(self, experiment):
        self.url = experiment.data["yt_link"]
        self.video = pafy.new(self.url)
        self.best = self.video.getbest(preftype="mp4")
        self.video.viewcount, self.video.author, self.video.length
        self.cap = cv2.VideoCapture(self.best.url)  # create connection

    def youtube_frame(self):
        # -------------- settings
        self.ret, self.frame = self.cap.read()  # ref picture
        return self.frame