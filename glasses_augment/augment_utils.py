import cv2 

def mixup_eyes(bg, fg, xEyes, yEyes, x_diff = 1, y_diff = 1 ,opaque = 0.5, gamma = 0):
    # bg_h, bg_w = bg.shape
    fg_h, fg_w, _ = fg.shape
    fg_h , fg_w = fg_h , fg_w

    yEyes = y_diff + yEyes
    xEyes = x_diff +xEyes

    # print(fg_h, fg_w)
    patch = bg[yEyes:yEyes+fg_h, xEyes:xEyes+fg_w]
    
    # print(patch.shape)
    # cv2.imwrite("patch.jpg",patch)
    # cv2.imwrite("bg.jpg",bg)

    blended = cv2.addWeighted(src1=patch, alpha=1-opaque, src2=fg, beta=opaque, gamma=gamma)
    
    res = bg.copy()
    res[yEyes:yEyes+fg_h, xEyes:xEyes+fg_w] = blended
    return res

def mixup_background(bg, fg, opaque = 0.4, gamma = 0.2, target_height = 256, target_width = 256):
    
    bg_new = cv2.resize(bg, (target_width,target_height))
    fg_new = cv2.resize(fg, (target_width, target_height))
    
    res = cv2.addWeighted(src1 = bg_new, alpha= 1 - opaque, src2 = fg_new, beta= opaque, gamma= gamma)
    return res