import dlib
import numpy as np
import os 
import cv2
import secrets
import mediapipe as mp 
import random

from glasses_augment.augment_modules import do_augment_mixup_glasses

def build_gen_glasses(path_to_dlib_model):
    if path_to_dlib_model is None:
        return None

    mp_face_mesh = mp.solutions.face_mesh
    detector = dlib.get_frontal_face_detector()
    predictor =  dlib.shape_predictor(path_to_dlib_model)
    print('Loaded gen glasses model')

    def func_gen(image):
        image_out = do_augment_mixup_glasses(image, mp_face_mesh, detector, predictor)
        if (len(image_out) == 0):
            return image
        return image_out

    return func_gen

if __name__ == '__main__':
    path_to_dlib_model = 'download/shape_predictor_68_face_landmarks.dat'
    tool_gen_glasses = build_gen_glasses(path_to_dlib_model)
    img = cv2.imread("/storage/hieunmt/tf_nonmask/unzip/VN-celeb/1/1.png")
    cv2.imwrite("faceglasses_input.jpg",img)
    img = tool_gen_glasses(img)
    cv2.imwrite("faceglasses_output.jpg",img)
