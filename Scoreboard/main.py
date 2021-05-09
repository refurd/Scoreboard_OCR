from stream import Stream
from separator import Separator
from ocr import OCR
import cv2


class Experiment(object):
    data = {
        "yt_link": "https://www.youtube.com/watch?v=TUikJi0Qhhw",
        "search_iter": 600,
        "area_min": 5500,
        "area_max": 12500,
        "OCR_m": 3
    }
m = 0.3
dim = (640,360)

def main():
    # read video input frames
    while True:
        # read frames
        in_frame = stream.youtube_frame()

        # create a white bg picture, where the scoreboard is black
        # return with scoreboard finder frame / separator.output_frame
        if separator.find_status:
            separator.find_scoreboard(in_frame)

            img = cv2.resize(separator.output_frame, dim)
            cv2.imshow('Finding', img)

        # crop the black rectangle out of the input image
        # processed the output by crop_scoreboard
        if separator.crop_status:
            separator.crop_scoreboard(in_frame)

        # when the scoreboard cropped, run the OCR
        # optimazed for 2 line and 2 name
        if not(separator.find_status or separator.crop_status):
            ocr.preprocessor(separator.temp_frame)
            ocr.mytesseract(ocr.output_frame)

            # if ocr_status True, finding the scoreboard was correctly
            # and crop the frame again at the previous position
            if ocr.ocr_status:
                separator.crop_status = True
                cv2.imshow('ocr', ocr.output_frame)
                print(ocr.ocr_text)
            # else find again the scoreboard
            else:
                separator.cur_iter = 0
                separator.find_status = True

        img2 = cv2.resize(in_frame, dim)
        cv2.imshow('input', img2)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # "q" to escape
            break
    stream.cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ocr = OCR(Experiment)
    stream = Stream(Experiment)
    separator = Separator(Experiment)
    main()
