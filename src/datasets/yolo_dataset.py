import os
import json, shutil
import random
from tqdm import tqdm
from .categories import CATEGORIES, CATEGORIES_ENCODING

def make_label_map(all_labels):
    """Create stable int ids for each unique character label."""
    return {lbl: idx for idx, lbl in enumerate(sorted(all_labels))}


def convert_bbox_to_yolo(box, img_w, img_h):
    """
    Converts the bbox in YOLO format:
    (x_min,y_min,w,h)->(x_c,y_c,w,h) normalized [0,1]
    """
    x_min, y_min, w, h = box
    x_c = x_min + w / 2
    y_c = y_min + h / 2
    return [x_c / img_w, y_c / img_h, w / img_w, h / img_h]


def create_YOLO_train_val(in_dataset_path, out_dataset_path, val_perc=0.2, bt_filter=[], encoding_categories=CATEGORIES, seed=42):
    """
    Divides the dataset in COCO format in training and validation. It also heberates the .yaml descriptor file.
    Params:
      - in_dataset_path: path of the dataset to divide
      - out_dataset_path: path of the generated dataset
      - val_perc: [default=0.2] percentage of data to use as validation set
      - bt_filter: bt type included in the lista are not included in the generated dataset
      - encoding_categories: [default=None] if not None, it is used as category for the generated dataset
    """
    # initialize seed
    random.seed(seed)

    # initialize output folder
    if os.path.exists(out_dataset_path):
        shutil.rmtree(out_dataset_path)
    os.makedirs(os.path.join(out_dataset_path, "train", "images"))
    os.makedirs(os.path.join(out_dataset_path, "val", "images"))
    os.makedirs(os.path.join(out_dataset_path, "train", "labels"))
    os.makedirs(os.path.join(out_dataset_path, "val", "labels"))

    # read data
    with open(os.path.join(in_dataset_path, "HomerCompTraining.json")) as f:
        in_data = json.load(f)
    images = in_data['images']

    # split data
    len_train_images = int(len(images)*(1-val_perc))
    len_val_images = len(images) - len_train_images

    random.shuffle(images)
    images_train = images[:len_train_images]
    images_val = images[len_train_images:]

    train_IDs = []
    val_IDs = []

    # copy images
    id_to_img = {}
    for img in tqdm(images_train, desc="Prepare Training Images", colour="GREEN"):
        img_name = os.path.basename(img['file_name'])
        shutil.copyfile(src=os.path.join(in_dataset_path, img['file_name'].replace("./", "")),
                        dst=os.path.join(out_dataset_path, "train", "images", img_name))

        img['file_name'] = img_name
        img['img_url'] = os.path.join("./", "images", img_name)

        train_IDs.append(img['id'])
        id_to_img[img['id']] = img

    for img in tqdm(images_val, desc="Prepare Validation Images", colour="GREEN"):
        img_name = os.path.basename(img['file_name'])
        ### ERROR IN DATASET!!!
        if img_name == "P_Mich_inv_13.jpg":
            img['file_name'] = "./images/homer2/txt171/P_Mich_inv_133.jpg"

        shutil.copyfile(src=os.path.join(in_dataset_path, img['file_name'].replace("./", "")),
                        dst=os.path.join(out_dataset_path, "val", "images", img_name))

        img['file_name'] = img_name
        img['img_url'] = os.path.join("./", "images", img_name)

        val_IDs.append(img['id'])
        id_to_img[img['id']] = img

    # Split Annotations
    if encoding_categories is None:
        categories = in_data['categories']
    else:
        categories = encoding_categories

    cat_ids = []
    for cat in categories:
        cat_ids.append(cat['id'])

    annotations_train = []
    annotations_val = []

    for ann in tqdm(in_data['annotations'], desc="Prepare Annotations", colour="MAGENTA"):
        if encoding_categories is not None:
            try:
                ann['category_id'] = CATEGORIES_ENCODING[ann['category_id']]
            except:
                pass

        if ann['category_id']  in cat_ids:
            img = id_to_img[ann['image_id']]
            yolo_bbox = convert_bbox_to_yolo(ann['bbox'], img['width'], img['height'])

            if ann['tags']['BaseType'][0] not in bt_filter:
                
                if ann['image_id'] in train_IDs:
                    annotations_train.append(ann)
                    with open(os.path.join(out_dataset_path, "train", "labels", f"{os.path.splitext(img['file_name'])[0]}.txt"), "a") as file:
                        file.write(f"{ann['category_id']} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")

                elif ann['image_id'] in val_IDs:
                    annotations_val.append(ann)
                    with open(os.path.join(out_dataset_path, "val", "labels", f"{os.path.splitext(img['file_name'])[0]}.txt"), "a") as file:
                        file.write(f"{ann['category_id']} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")

                else:
                    print(f"⚠️ Annotation id:{ann['id']} not in training or validation split.")


    # Write JSONs
    training_data = {
        "info":{"version":1, "description":"Homer training set for YOLO"},
        "licenses": in_data["licenses"],
        "categories": categories,
        "images": images_train,
        "annotations": annotations_train
    }            
    validation_data = {
        "info":{"version":1, "description":"Homer validation set for YOLO"},
        "licenses": in_data["licenses"],
        "categories": categories,
        "images": images_val,
        "annotations": annotations_val
    }

    with open(os.path.join(out_dataset_path, "train", "annotations.json"), "w") as json_file:
        json.dump(training_data, json_file, indent=4, ensure_ascii=False)
    
    with open(os.path.join(out_dataset_path, "val", "annotations.json"), "w") as json_file:
        json.dump(validation_data, json_file, indent=4, ensure_ascii=False)

    # Write yaml file
    with open(os.path.join(out_dataset_path, "dataset.yaml"), "w") as yaml_file:
        yaml_file.write(f"path: {out_dataset_path}\n")
        yaml_file.write("train: train/images\n")
        yaml_file.write("val: val/images\n\n")
        yaml_file.write("# COCO-style JSON annotations\n")
        yaml_file.write("train_annotations: train/annotations.json\n")
        yaml_file.write("val_annotations: val/annotations.json\n\n")
        yaml_file.write("names:\n")
        for cat in categories:
            yaml_file.write(f"  {cat['id']}: {cat['name']}\n")


def converto_out_COCO(result, img_id=0, ann_id=0):
    """
    Converts the YOLO ultralytics results for a single image in a COCO-JSON format.

    Params:
        - result: the YOLO ultralytics result
        - img_id: [default:0] image id
        - amm_id: [default:0] first id for the annotations

    Returns:
        the dictionary of the coco-json
    """
    
    coco_output = {
        "categories": [],
        "images": [],
        "annotations": []
    }

    # add categories
    for cls_id, name in result.names.items():
        coco_output["categories"].append({"id": cls_id, "name": name})

    # Add image
    coco_output["images"].append({
        "id": img_id,
        "file_name": os.path.basename(result.path),
        "width": result.orig_shape[1],
        "height": result.orig_shape[0]
    })

    # add annotations
    
    for box in result.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = box.xyxy.tolist()[0]

        # Converti in formato COCO [x,y,w,h]
        w, h = x2 - x1, y2 - y1
        coco_output["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": cls_id,
            "bbox": [int(x1), int(y1), int(w), int(h)],
            "score": conf
        })
        ann_id += 1

    return coco_output