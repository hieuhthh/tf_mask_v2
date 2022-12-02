import os
import shutil
import multiprocessing
import cv2

from crop_face import *

def clean_image(route, to_des, im_size, do_crop_face=False):
    """
    using multiprocessing
    input:
        route to main directory and phrase ("train", "valid", "test")
        or just route to the directory that its subfolder are classes
    output:
        X_path: path to img
        Y_int: int label
        all_class: list of string class name
    """
    do_multiprocess = False if do_crop_face else True

    global task

    print(route)

    if do_crop_face:
        crop_face, extract_face = get_crop_face_tool(im_size=im_size, scale=1.2)

    def task(route, all_class, list_cls, to_des, im_size):
        sign = route.split('/')[-1]

        for cl in list_cls:
            path2cl = os.path.join(route, cl)

            if len(os.listdir(path2cl)) < 1:
                continue

            des_class = os.path.join(to_des, sign + '_' + cl)

            try:
                os.mkdir(des_class)
            except:
                pass

            for imfile in os.listdir(path2cl):
                impath = os.path.join(path2cl, imfile)
                imsave = os.path.join(des_class, imfile)

                try:
                    img = cv2.imread(impath)

                    if do_crop_face:
                        bb_box = extract_face(img)
                        img = crop_face(img, bb_box)
    
                    img = cv2.resize(img, (im_size, im_size))
                    cv2.imwrite(imsave, img)
                except:
                    print(impath)

            if len(os.listdir(des_class)) < 1:
                try:
                    os.mkdir(des_class)
                except:
                    pass

    all_class = sorted(os.listdir(route))
    n_labels = len(all_class)

    if do_multiprocess:    
        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpu_count)
        processes = []
        print('cpu_count', cpu_count)

        n_per = int(n_labels // cpu_count + 1)

        for i in range(cpu_count):
            start_pos = i * n_per
            end_pos = (i + 1) * n_per
            list_cls = all_class[start_pos:end_pos]

            p = pool.apply_async(task, args=(route,all_class,list_cls,to_des,im_size))
            processes.append(p)

        result = [p.get() for p in processes]

        pool.close()
        pool.join()

    else:
        task(route,all_class,all_class,to_des,im_size)

if __name__ == '__main__':
    from utils import *

    settings = get_settings()
    globals().update(settings)

    # mask

    des = path_join(route, 'clean_tinh_dataset')
    mkdir(des)

    route = 'unzip/ImgOut2'
    clean_image(route, des, im_size)

    # des = path_join(route, 'mask_dataset')
    # mkdir(des)

    # route = 'unzip/ImgOut2'
    # clean_image(route, des, im_size)

    # route = 'unzip/AFDB_masked_face_dataset'
    # clean_image(route, des, im_size)

    # route = 'unzip/RWMFD_part_2_pro'
    # clean_image(route, des, im_size, do_crop_face=True)

    # route = 'unzip/final'
    # clean_image(route, des, im_size)

    # route = 'unzip/masked_ms1m'
    # clean_image(route, des, im_size)

    # route = 'unzip/Real_faces_align_masked'
    # clean_image(route, des, im_size)

    # nonmask

    # des = path_join(route, 'nonmask_dataset')
    # mkdir(des)

    # route = 'unzip/VN-celeb'
    # clean_image(route, des, im_size)

    # route = 'unzip/gnv_dataset'
    # clean_image(route, des, im_size)

    # route = 'unzip/glint360k_224'
    # clean_image(route, des, im_size)