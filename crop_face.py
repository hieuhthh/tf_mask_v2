import cv2
import mediapipe as mp
import numpy as np

def get_crop_face_tool(im_size=160, scale=1.3):
    FaceDetection = mp.solutions.face_detection.FaceDetection
    detector = FaceDetection(min_detection_confidence=0.5)

    def crop_face(img: np.ndarray, box):
        if box is None:
            return None
            
        x, y, w, h = box
        x, y = abs(x), abs(y)

        return cv2.resize(img[y: y + h, x: x + w], (im_size, im_size))

    def extract_face(img: np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        results = detector.process(img)
        if results.detections:
            for detection in results.detections:
                bb = detection.location_data.relative_bounding_box
                bb_box = [
                    int(max(0.0, bb.xmin - bb.width*0.1) * w),
                    int(max(0.0, bb.ymin - bb.height*0.1) * h),
                    int(bb.width *  w * scale),
                    int(bb.height * h * scale)
                ]
                return bb_box

    return crop_face, extract_face