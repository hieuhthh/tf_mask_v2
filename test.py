import cv2 
from glasses_augment.augment_modules import do_augment_mixup_glasses
import dlib
import mediapipe as mp 

#call all models
mp_face_mesh = mp.solutions.face_mesh
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./glasses_augment/shape_predictor_68_face_landmarks.dat")


img = cv2.imread("huy.jpg")

#input img array
out = do_augment_mixup_glasses(img, mp_face_mesh,detector, predictor)

cv2.imwrite("out1.jpg",out)
