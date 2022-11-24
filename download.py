import gdown

from utils import *

settings = get_settings()
globals().update(settings)

des = path_join(route, 'download')
mkdir(des)

# url = "https://drive.google.com/file/d/1gxcvRNl6K3hCDaTq7JZcgSn9rdmLh2wK/view?usp=share_link"
# output =  f"{des}/shape_predictor_68_face_landmarks.dat"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/1VCNkiFvNhxVTDklMkT3GaVwX0C624RE4/view?usp=sharing"
# output =  f"{des}/vnceleb.zip"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/1-VYjrPIoVWkE7uwsNO6exayh7yhpPo0n/view?usp=share_link"
# output =  f"{des}/gnv_dataset.zip"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = 'https://drive.google.com/file/d/1Gd1FDFiJ6RyK4mpUc3SohAdgxOExExmS/view?usp=share_link'
url = "https://drive.google.com/file/d/12OOOrzRUDogH868q99qtHWxUDhCxXFix/view?usp=share_link"
output =  f"{des}/mask_tinh.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)