import os
import shutil
import multiprocessing
import cv2
import random
import time

from gen_mask import *
from gen_classes import *

def make_mask_dataset(route, to_des, im_size, tool_gen_mask, tool_gen_glasses=None, glasses_prob=0, 
                      n_mask=2, do_resize=True, n_thres=5):
    """
    using multiprocessing
    """

    print(route)

    if tool_gen_mask is None:
        print('tool_gen_mask is None')
        return

    global task

    sign = "mask" if tool_gen_glasses is None else "glasses_mask"

    def task(route, list_cls, to_des, im_size):
        for cl in list_cls:
            path2cl = os.path.join(route, cl)

            if len(os.listdir(path2cl)) < 1:
                continue

            des_class = os.path.join(to_des, sign + '_' + cl)

            try:
                os.mkdir(des_class)
            except:
                pass

            if len(os.listdir(path2cl)) > n_thres:
                n_do = 1
            else:
                n_do = n_mask

            for imfile in os.listdir(path2cl):
                impath = os.path.join(path2cl, imfile)

                for i in range(n_do):
                    try:
                        img = cv2.imread(impath)
                        img = cv2.resize(img, (im_size, im_size)) if do_resize else img
                        img = tool_gen_mask(img)
                        # if tool_gen_glasses is not None:
                        #     if random.random() < glasses_prob:
                        #         temp_img = tool_gen_glasses(img)
                        #         if temp_img is not None:
                        #             img = temp_img
                        imsave = os.path.join(des_class, f"{i}_" + imfile)
                        cv2.imwrite(imsave, img)
                    except:
                        print(i, impath)

    all_class = sorted(os.listdir(route))
    cpu_count = multiprocessing.cpu_count()
    cpu_count = cpu_count * max(min(80, len(all_class) // 2000), 1)

    pool = multiprocessing.Pool(cpu_count)
    processes = []

    n_labels = len(all_class)
    n_per = int(n_labels // cpu_count + 1)

    print('cpu_count', cpu_count)

    start_time = time.time()
    for i in range(cpu_count):
        start_pos = i * n_per
        end_pos = (i + 1) * n_per
        list_cls = all_class[start_pos:end_pos]
     
        p = pool.apply_async(task, args=(route,list_cls,to_des,im_size))
        processes.append(p)

        # task(route,all_class,list_cls)

    result = [p.get() for p in processes]

    pool.close()
    pool.join()

    print("Time:", time.time() - start_time)

if __name__ == '__main__':
    from utils import *
    from multiprocess_dataset import *

    os.environ["CUDA_VISIBLE_DEVICES"]=""

    settings = get_settings()
    globals().update(settings)
    
    tool_gen_mask = build_gen_mask(path_to_dlib_model, from_cv2=True)
    # tool_gen_glasses = build_gen_glasses(path_to_dlib_model)

    des = path_join(route, 'mask_dataset')
    mkdir(des)

    route = 'unzip/VN-celeb'
    make_mask_dataset(route, des, im_size, tool_gen_mask, tool_gen_glasses=None, glasses_prob=0.0,
                      n_mask=4, do_resize=True, n_thres=1000)

    route = 'unzip/gnv_dataset'
    make_mask_dataset(route, des, im_size, tool_gen_mask, tool_gen_glasses=None, glasses_prob=0.0,
                      n_mask=4, do_resize=True, n_thres=1000)

    route = 'unzip/processed_crop'
    make_mask_dataset(route, des, im_size, tool_gen_mask, tool_gen_glasses=None, glasses_prob=0.0,
                      n_mask=4, do_resize=True, n_thres=1000)

    route = 'unzip/glint360k_224'
    make_mask_dataset(route, des, im_size, tool_gen_mask, tool_gen_glasses=None, glasses_prob=0.0,
                      n_mask=2, do_resize=True, n_thres=1000)

