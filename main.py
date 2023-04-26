import os

import cv2
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty, NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy._version import __version__
print(__version__)
from detection import Detect
from utils.plots import Annotator, colors

# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p": (512, 512),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}


class KivyCamera(BoxLayout):
    filename = StringProperty('video.avi')
    frames_per_second = NumericProperty(30.0)
    video_resolution = StringProperty('480p')
    line_thickness = 2

    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.img1 = Image()
        self.add_widget(self.img1)
        self.detection = Detect()
        self.capture = cv2.VideoCapture(0)
        self.out = cv2.VideoWriter(self.filename, self.get_video_type(self.filename), self.frames_per_second,
                                   self.get_dims(self.capture, self.video_resolution))
        self.counter = 0
        Clock.schedule_interval(self.update, 1 / self.frames_per_second)
        self.last_pred = []
        self.last_labels = []
        self.last_boxes = []
        self.last_colors = []

        self.detection_freq = 20

    def update(self, *args):
        ret, frame = self.capture.read()
        annotator = Annotator(frame, line_width=self.line_thickness, example=str(self.detection.names))

        if self.counter % self.detection_freq == 0:
            self.counter = 1
            self.last_pred, self.last_labels, self.last_boxes, self.last_colors = self.detection(frame)
        if len(self.last_pred):
            if self.last_pred[0].shape[0]:
                for label, c, xyxy in zip(self.last_labels, self.last_colors, self.last_boxes):
                    annotator.box_label(xyxy, label, color=colors(c, True))

        self.counter += 1
        self.out.write(frame)
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.img1.texture = texture

    # Set resolution for the video capture
    # Function adapted from https://kirr.co/0l6qmh
    def change_resolution(self, cap, width, height):
        self.capture.set(3, width)
        self.capture.set(4, height)

    # grab resolution dimensions and set video capture to it.
    def get_dims(self, cap, video_resolution='1080p'):
        width, height = STD_DIMENSIONS["480p"]
        if self.video_resolution in STD_DIMENSIONS:
            width, height = STD_DIMENSIONS[self.video_resolution]
        ## change the current caputre device
        ## to the resulting resolution
        self.change_resolution(cap, width, height)
        return width, height

    def get_video_type(self, filename):
        filename, ext = os.path.splitext(filename)
        if ext in VIDEO_TYPE:
            return VIDEO_TYPE[ext]
        return VIDEO_TYPE['avi']


class CamApp(App):
    def build(self):
        return KivyCamera()


if __name__ == '__main__':
    CamApp().run()
