import mediapipe as mp
import cv2 
from glasses_augment.get_landmark import getRightEyeRect, getLeftEyeRect, getLandmarks
from glasses_augment.augment_utils import mixup_eyes, mixup_background
import time
import random
from glasses_augment.glasses2 import augment_glasses

# img = cv2.imread("4.jpg")
# bg = cv2.imread("./glare/b1.jpg")



def do_mixup_eyes(mp_face_mesh, img, bg, mode = 0):

    landmark = getLandmarks(img, mp_face_mesh)

    # start = time.time()
    xRightEye, yRightEye, rightEyeWidth, rightEyeHeight, crop_eyeRight = getRightEyeRect(img,landmark)
    xLeftEye, yLeftEye, leftEyeWidth, leftEyeHeight ,crop_eyeLeft = getLeftEyeRect(img, landmark)

    eyeRightBg = mixup_background(bg, crop_eyeRight, target_width = rightEyeWidth, target_height = rightEyeHeight)
    eyeLeftBg = mixup_background(bg, crop_eyeLeft, target_width = leftEyeWidth, target_height = leftEyeHeight)
    
    if mode == 0:

        img = mixup_eyes(img, eyeLeftBg,xLeftEye, yLeftEye, x_diff = 15, y_diff= -15)
        img = mixup_eyes(img, eyeLeftBg,xLeftEye, yLeftEye, x_diff = -15, y_diff= 15)
        img = mixup_eyes(img, eyeLeftBg,xLeftEye, yLeftEye, x_diff = 0, y_diff= -15)
        img = mixup_eyes(img, eyeLeftBg,xLeftEye, yLeftEye, x_diff = -15, y_diff= 0)
        img = mixup_eyes(img, eyeLeftBg,xLeftEye, yLeftEye, x_diff = 0, y_diff= 0)

        img = mixup_eyes(img, eyeRightBg,xRightEye, yRightEye, x_diff = 15, y_diff= 0)
        img = mixup_eyes(img, eyeRightBg,xRightEye, yRightEye, x_diff = 0, y_diff= 15)
        img = mixup_eyes(img, eyeRightBg,xRightEye, yRightEye, x_diff = -15, y_diff= 15)
        img = mixup_eyes(img, eyeRightBg,xRightEye, yRightEye, x_diff = 15, y_diff= -15)
        img = mixup_eyes(img, eyeRightBg,xRightEye, yRightEye, x_diff = 0, y_diff= 0)
    
    if mode == 1:

        img = mixup_eyes(img, eyeLeftBg,xLeftEye, yLeftEye, x_diff = 0, y_diff= 0)
        img = mixup_eyes(img, eyeRightBg,xRightEye, yRightEye, x_diff = 0, y_diff= 0)

    if mode == 2:

        img = mixup_eyes(img, eyeLeftBg,xLeftEye, yLeftEye, x_diff = 0, y_diff= -15)
        img = mixup_eyes(img, eyeLeftBg,xLeftEye, yLeftEye, x_diff = -15, y_diff= 0)
        img = mixup_eyes(img, eyeLeftBg,xLeftEye, yLeftEye, x_diff = 15, y_diff= 0)
        img = mixup_eyes(img, eyeLeftBg,xLeftEye, yLeftEye, x_diff = 0, y_diff= 15)

        img = mixup_eyes(img, eyeRightBg,xRightEye, yRightEye, x_diff = 15, y_diff= 0)
        img = mixup_eyes(img, eyeRightBg,xRightEye, yRightEye, x_diff = 0, y_diff= 15)
        img = mixup_eyes(img, eyeRightBg,xRightEye, yRightEye, x_diff = 0, y_diff= -15)
        img = mixup_eyes(img, eyeRightBg,xRightEye, yRightEye, x_diff = -15, y_diff= 0)

    # stop = time.time()
    # print("Time augment:" , stop - start)
    return img 

def mixup_augmentation(img, mp_face_mesh):
    # ,"./glare/b2.jpg"
    try:
        bg_list  = ["./glasses_augment/glare/b1.jpg","./glasses_augment/glare/b2.jpg"]
        mode_list = [0, 1, 2]

        mode = random.choice(mode_list)
        bg_name = random.choice(bg_list)

        bg = cv2.imread(bg_name)

        img_res = do_mixup_eyes(mp_face_mesh,img,bg, mode = mode)
        
        return img_res
    except:
        return img

def do_augment_mixup_glasses(img, mp_face_mesh,detector, predictor): #input is img array    
    try:
        out = augment_glasses(img, detector, predictor)
        out_img = mixup_augmentation(out, mp_face_mesh)
        return out_img
    except:
        return img
    
# out_img = do_augment_mixup_glasses(img)
# cv2.imwrite("save_aug_random.png", out_img)




# print(landmark)



# print(img.shape)
# print(xLeftEye,yRightEye)
# print(crop_eyeLeft.shape)




# cv2.imwrite("save2.jpg",eyeLeft_mix)