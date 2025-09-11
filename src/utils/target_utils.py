import os
from .fileformat_handler import FFHandler
from PIL import Image


def create_pap(or_image_path, prediction_path, save_img=False, image_kb_max=1300000):
    file_h = FFHandler()

    out_fld = os.path.dirname(prediction_path)
    img_name = os.path.basename(or_image_path)

    image = Image.open(or_image_path)
    image = image.convert("RGB") 

    image_path = os.path.join(out_fld, img_name)

    quality = 100
    image.save(image_path,quality=quality,optimize=True)

    file_size = os.path.getsize(image_path)

    while (file_size > image_kb_max and quality > 1):
        quality -= 5
        image.save(image_path, quality=quality, optimize=True)
        file_size = os.path.getsize(image_path)
        #print(name, file_size)
   
    file_h.save_formattedfile(image_path, prediction_path, out_fld)

    if not save_img:
        os.remove(image_path)
    