import zipfile
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
import time

from utils import *

settings = get_settings()
globals().update(settings)

from_dir = path_join(route, 'download')
des = path_join(route, 'unzip')

mkdir(des)

def fast_unzip(zip_path, out_path):
    print(zip_path)
    try:
        start = time.time()
        with ZipFile(zip_path) as handle:
            with ThreadPoolExecutor(2) as exe:
                _ = [exe.submit(handle.extract, m, out_path) for m in handle.namelist()]
    except:
        pass
    finally:
        print('Unzip', zip_path, 'Time:', time.time() - start)

# filename = 'vnceleb'
# zip_path = path_join(from_dir, filename + '.zip')
# fast_unzip(zip_path, des)

# filename = 'gnv_dataset'
# zip_path = path_join(from_dir, filename + '.zip')
# fast_unzip(zip_path, des)

# filename = 'mask_tinh'
# zip_path = path_join(from_dir, filename + '.zip')
# fast_unzip(zip_path, des)

# filename = 'AFDB_masked_face_dataset'
# zip_path = path_join(from_dir, filename + '.zip')
# fast_unzip(zip_path, des)

# filename = 'RWMFD_part_2_pro'
# zip_path = path_join(from_dir, filename + '.zip')
# fast_unzip(zip_path, des)

# filename = 'final'
# zip_path = path_join(from_dir, filename + '.zip')
# fast_unzip(zip_path, des)

# filename = 'masked_ms1m'
# zip_path = path_join(from_dir, filename + '.zip')
# fast_unzip(zip_path, des)

# filename = 'Real_faces_align_masked'
# zip_path = path_join(from_dir, filename + '.zip')
# fast_unzip(zip_path, des)

# filename = 'processed_crop'
# zip_path = path_join(from_dir, filename + '.zip')
# fast_unzip(zip_path, des)

# zip_path = '/home/minint-t14g3hk-local/face_bucket_huy/glint360k_224_copy.zip'
# fast_unzip(zip_path, des)

