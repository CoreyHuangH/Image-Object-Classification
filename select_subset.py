import os
import shutil
import json
import random
from pycocotools.coco import COCO

# Specify the root directory of the COCO dataset
dataDir = "./COCO_data" # Modify this path to the location of the COCO dataset

# Initialize COCO API for instance annotations
annFile = os.path.join(dataDir, "annotations/instances_train2014.json") # Modify this path to the location of the annotations
coco = COCO(annFile)

# Specify the new category IDs to extract
# Corresponding to bird, cat, dog, horse, sheep
# Ensure these IDs are correct in the COCO dataset
catIds = [16, 17, 18, 19, 20]
catNames = ["bird", "cat", "dog", "horse", "sheep"]

# Create a directory to store the subset
subset_dir = "my_coco_subset"
os.makedirs(subset_dir, exist_ok=True)

# Create directories for each category
category_dirs = {catName: os.path.join(subset_dir, catName) for catName in catNames}
for dir_path in category_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Set the target number of images per category
target_num_images_per_category = 1200

# Create dictionaries to store image paths and selected image IDs for each category
imgPaths_by_category = {catName: [] for catName in catNames}
selected_imgIds_by_category = {catName: [] for catName in catNames}

# Get all image IDs
imgIds = coco.getImgIds()

# Iterate through each image ID to collect images and annotations
for imgId in imgIds:
    # Get the annotation IDs for this image
    annIds = coco.getAnnIds(imgIds=imgId, iscrowd=None)
    anns = coco.loadAnns(annIds)

    # Create a set of categories present in this image
    present_catIds = {ann["category_id"] for ann in anns}

    # Check if the image contains instances of any of the specified categories
    for catId, catName in zip(catIds, catNames):
        if catId in present_catIds:
            imgPath = os.path.join(
                dataDir, "train2014", coco.loadImgs(imgId)[0]["file_name"] # Modify this path to the location of the images
            )
            imgPaths_by_category[catName].append(imgPath)
            selected_imgIds_by_category[catName].append(imgId)
            # Stop collecting for this image if we have enough images for this category
            if len(imgPaths_by_category[catName]) >= target_num_images_per_category:
                break

# Copy images to the corresponding category directories
for catName in catNames:
    imgPaths = imgPaths_by_category[catName]
    if len(imgPaths) > target_num_images_per_category:
        imgPaths = random.sample(imgPaths, target_num_images_per_category)

    for img_path in imgPaths:
        dst = os.path.join(category_dirs[catName], os.path.basename(img_path))
        shutil.copy(img_path, dst)

print(f"Copied images to {subset_dir}")

# Create a dictionary to store the subset annotations
subset_anns = {
    "info": coco.dataset["info"],
    "licenses": coco.dataset["licenses"],
    "categories": coco.loadCats(catIds),
    "images": [],
    "annotations": [],
}

# Iterate through the selected image IDs to collect annotations
for catName in catNames:
    imgIds = selected_imgIds_by_category[catName]
    for img_id in imgIds:
        # Add image information to the subset annotations
        img_info = coco.loadImgs(img_id)[0]
        subset_anns["images"].append(img_info)

        # Get the annotation IDs for this image
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=catIds, iscrowd=None)

        # Add annotation information to the subset annotations
        anns = coco.loadAnns(ann_ids)
        subset_anns["annotations"].extend(anns)

# Save the subset annotations to a JSON file
subset_ann_file = os.path.join(subset_dir, "subset_annotation.json")
with open(subset_ann_file, "w") as f:
    json.dump(subset_anns, f)

print(f"Saved subset annotations to {subset_ann_file}")
