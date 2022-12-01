import dlib
import cv2
import numpy as np
from scipy import ndimage
import random



def resize(img, width):
    r = float(width) / img.shape[1]
    dim = (width, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

def blend_transparent(face_img, sunglasses_img):

    overlay_img = sunglasses_img[:,:,:3]
    overlay_mask = sunglasses_img[:,:,3:]
    
    background_mask = 255 - overlay_mask

    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

def augment_glasses(img, detector, predictor):
    try:
        glasses_list = ["./glasses_augment/glare/h.png", "./glasses_augment/glare/k.png", "./glasses_augment/glare/l.png"]
        glasses_img = random.choice(glasses_list)
        glasses = cv2.imread(glasses_img, -1)
        
        img_copy = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect faces
        dets = detector(gray, 1)

        #find face box bounding points
        for d in dets:

            x = d.left()
            y = d.top()
            w = d.right()
            h = d.bottom()

        dlib_rect = dlib.rectangle(x, y, w, h)

        ##############   Find facial landmarks   ##############
        detected_landmarks = predictor(gray, dlib_rect).parts()
        # print(detected_landmarks)
        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            if idx == 0:
                eye_left = pos
            elif idx == 16:
                eye_right = pos

            try:

                degree = np.rad2deg(np.arctan2(eye_left[0] - eye_right[0], eye_left[1] - eye_right[1]))

            except:
                pass

        eye_center = (eye_left[1] + eye_right[1]) / 2

        glass_trans = int(.3 * (eye_center - y))
        # glass_trans = 0
        face_width = w - x

        glasses_resize = resize(glasses, face_width)

        yG, xG, cG = glasses_resize.shape
        glasses_resize_rotated = ndimage.rotate(glasses_resize, (degree+90))
        glass_rec_rotated = ndimage.rotate(img[y + glass_trans:y + yG + glass_trans, x:w], (degree+90))

        h5, w5, s5 = glass_rec_rotated.shape
        rec_resize = img_copy[y + glass_trans:y + h5 + glass_trans, x:x + w5]
        blend_glass3 = blend_transparent(rec_resize , glasses_resize_rotated)
        img_copy[y + glass_trans:y + h5 + glass_trans, x:x+w5 ] = blend_glass3
        return img_copy
    except:
        return img
    

# img = cv2.imread("messi.jpg")
# out = augment_glasses(img)
# cv2.imwrite("out.jpg",out)