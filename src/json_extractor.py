# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import json
import base64
import os

root = '/workspace/'
dataset_path = root + 'Dataset/pegon-ocr-master/Annotations/'
mujarobat_doa = dataset_path + 'Mujarobat Doa/'

arabic_ocr_src_path = root + 'Arabic-OCR/src/'
test_path  = arabic_ocr_src_path + 'test/'  # img files
truth_path = arabic_ocr_src_path + 'truth/' # txt files

with open(mujarobat_doa + 'Cover-1.json', encoding="utf8") as jsonfile:
    data = json.load(jsonfile)

# txt_file = data['imagePath'][:-3] + 'txt'
# with open(txt_file, 'w', encoding="utf8") as res:
#     for shape in data['shapes']:
#         res.write(shape['label'])
#         res.write(' ')

img_data = data['imageData']
with open(data['imagePath'], "wb") as img:
    img.write(base64.decodebytes(bytes(img_data, "utf-8")))
# -


