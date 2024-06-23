# Image-Object-Classification
My Intern Project 1 @USTC: Image object classification

## Get the demo from Huggingface Space:
```
git clone https://huggingface.co/spaces/CoreyHuangH/ResNet50-5-Class
```

## Instruction:
**Dataset**
- COCO [https://cocodataset.org/#explore], the images are labeled. Prepare a dataset by picking 5
kinds of images (bird, cat, dog, horse and sheep were chosen in this project), and each kind contains a few hundred images. Split the dataset into 3 batches:
training set, validation set, and test set.

**Tool**
- Use PyTorch to do model training and inference (prediction).

**Model**
- Implement a basic Convolutional Neural Network (CNN) for object detection.
- Use a pre-trained model like Faster R-CNN or YOLO as a starting point.

**Training & Evaluation**
- Use appropriate regularization to avoid over-training.
- Compute precision/recall/F1 for the training result.

**Machine**
Start with CPU on local laptop with small dataset. If it works well, try to switch GPU server
(CUDA)
