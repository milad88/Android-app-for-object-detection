import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.general import (Profile, non_max_suppression)
from utils.plots import Annotator


class Detect:
    def __init__(self, weights_path='model/weights/best.pt', iou_thres=0.35, max_det=1000):
        self.model = DetectMultiBackend(weights_path)
        self.names = self.model.names

        self.conf_thres = .25 # confidence threshold
        self.iou_thres = iou_thres  # NMS IOU threshold
        self.max_det = max_det

        self.classes = None  # filter by class: --class 0  or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = True  # augmented inference
        self.visualize = False  # visualize features
        self.project_dir = 'detections/'  # save results to project/name
        self.line_thickness = 1  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.vid_stride = 1

    def __call__(self, im):
        org_im = im

        im = np.moveaxis(np.array(im), 2, 0)
        im = np.array([im])

        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255.0
        pred = self.model.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, False, max_det=self.max_det)
        s = ''
        # for i, det in enumerate(pred):  # per image
        seen += 1
        s += ''
        labels = []
        XYs = []
        colors = []

        if pred[0].shape[0]:
            # Rescale boxes from img_size to im0 size
            # det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im.shape).round()
            # Print results
            for c in pred[0][:, 5].unique():
                n = (pred[0][:, 5] == c).sum()  # detections per class
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "

            for *xyxy, conf, cls in reversed(pred[0]):
                c = int(cls)  # integer class
                label = None if self.hide_labels else (
                    self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                labels.append(label)
                colors.append(c)
                XYs.append(xyxy)

            print(f"{s} {dt[1].t * 1E3:.1f}ms")

        # Stream results

        return pred, labels, XYs, colors

# weights_path = 'exp38/weights/best.pt'
# detect = Detect()
# cap = cv2.VideoCapture(0)
#
# # Check if the webcam is opened correctly
# if not cap.isOpened():
#     raise IOError("Cannot open webcam")
#
# dim = (512, 512)
#
# # resize image
# while True:
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
#
#     frame = detect( frame)
#
#     cv2.imshow('Input', frame)
#
#     c = cv2.waitKey(1)
#     if c == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
