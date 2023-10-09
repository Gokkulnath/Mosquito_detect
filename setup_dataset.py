import importlib
import os
from pathlib import Path
import shutil
import pandas as pd 
import json
import yaml
from tqdm import tqdm
import zipfile
from sklearn.model_selection import train_test_split


dataset_dir = Path("./Mosquito_Dataset")
train_zip = dataset_dir/"train_images.zip"
test_zip = dataset_dir/"test_images_phase1.zip"
IMAGE_DIR = "images/{}_images"
LABEL_DIR = "labels/{}_images"
ASSET_DIR = Path('assets')

if not dataset_dir.exists():
    gdown = importlib.import_module('gdown')
    url = "https://drive.google.com/drive/folders/1T5wkBC43CLYBPJKh4wToyq4W5Md6UtwC"
    gdown.download_folder(url, quiet=False, use_cookies=False)

def extract_zip(zip_path,split):
    if not os.path.exists(f"{dataset_dir}/{IMAGE_DIR.format(split)}"):
        zf = zipfile.ZipFile(zip_path)
        total_items = sum((1 for file in zf.infolist()))
        extracted_size = 0
        for file in tqdm(zf.infolist(),total=total_items):
            extracted_size += file.file_size
            zf.extract(file,path=f"{dataset_dir}/{IMAGE_DIR.format(split)}/")


extract_zip(test_zip,"test")
extract_zip(train_zip,"train")


idx2lab = { 0: "culex",
 1: "albopictus",
 2: "culiseta",
 3: "japonicus/koreicus",
 4: "anopheles",
 5: "aegypti"}
lab2idx = {v:k for k,v in idx2lab.items()}


train_df = pd.read_csv(f"{dataset_dir}/train.csv")
test_df = pd.read_csv(f"{dataset_dir}/test_phase1_v2.csv")

train_df['fname']=train_df['img_fName'].apply(lambda x: x.rstrip(".jpeg"))
train_df['class_code']=train_df['class_label'].apply(lambda x : lab2idx[x])

# Prepare the mosquito.yaml : YOLO Dataset Config 
if not ASSET_DIR.exists():
    ASSET_DIR.mkdir(exist_ok=True)

with open(ASSET_DIR/"class_mapping.json",'w') as f:
    json.dump({"idx2lab": idx2lab, "lab2idx":lab2idx}, f)


#  val images (relative to 'path') 128 images 
dataset_config_yaml = {
    "path": str(dataset_dir.absolute()),
    "train": IMAGE_DIR.format('train'),
    "val": IMAGE_DIR.format('val'),
    "test": IMAGE_DIR.format('test'),
    "names": idx2lab

}

with open(ASSET_DIR/"mosquito.yaml",'w') as f:
    yaml.dump(dataset_config_yaml, f,indent=4)



print("Setting up train and validation split ")
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42, shuffle=True, stratify=train_df.class_code)


# Prepare Label


# https://medium.com/analytics-vidhya/basics-of-bounding-boxes-94e583b5e16c
# (x1, y1): Corresponds to the x and y coordinate of the top left corner of the rectangle.
# (x2, y2): Corresponds to the x and y coordinate of the bottom right corner of the rectangle.
# (xc, yc): Corresponds to the x and y coordinate of the center of the bounding box.
# Width: Represents the width of the bounding box.
# Height: Represents the height of the bounding box.

# xc = ( a.bbx_xtl + a.bbx_xbr ) / 2
# yc = ( a.bbx_ytl + a.bbx_ybr ) / 2
# width = ( a.bbx_xbr — a.bbx_xtl)
# height = (a.bbx_ybr — a.bbx_ytl)

from PIL import Image
import cv2
import os 
def perpare_label(a,split='train', debug=False):
    os.makedirs(f"{dataset_dir}/{LABEL_DIR.format(split)}",exist_ok=True)
    with open(os.path.join(f"{dataset_dir}/{LABEL_DIR.format(split)}",f'{a.fname}.txt'),'w') as f:
        xc = (( a.bbx_xtl + a.bbx_xbr ) / 2) / a.img_w
        yc = (( a.bbx_ytl + a.bbx_ybr ) / 2) / a.img_h
        width = ( a.bbx_xbr - a.bbx_xtl)/a.img_w
        height = (a.bbx_ybr - a.bbx_ytl)/a.img_h
        row = [a.class_code, xc, yc, width,height]
        label_str = " ".join([str(r) for r in row])
        # print(label_str)
        f.writelines(label_str)
    if debug:
        os.makedirs(f"{dataset_dir}/{IMAGE_DIR.format(split)}/debug",exist_ok=True)
        img_path = f"{dataset_dir}/{IMAGE_DIR.format(split)}/{a.img_fName}"
        img_debug_path = f"{dataset_dir}/{IMAGE_DIR.format(split)}/debug/{a.img_fName}"
        img = cv2.imread(img_path)
        start_point = (int(a.bbx_xtl), int(a.bbx_ytl))
        end_point = (int(a.bbx_xbr), int(a.bbx_ybr))
        cv2.rectangle(img, start_point, end_point, color=(0,255,0), thickness=2)   
        cv2.imwrite(f"{img_debug_path}", img)



# for _,row in tqdm(train_df.iterrows()):
#     perpare_label(row,split='train')
    
for _,row in tqdm(val_df.iterrows()):
    try:
        shutil.move(f"{dataset_dir}/{IMAGE_DIR.format('train')}/{row.img_fName}", f"{dataset_dir}/{IMAGE_DIR.format('val')}/{row.img_fName}")
    except:
        pass
    perpare_label(row,split="val")
