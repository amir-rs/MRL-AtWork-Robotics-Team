# PPT (Precision Placement Tables)


>## Challanges:
>       . Similar cavity shapes with different sizes.
>       . Working with Realsense sensors.
>       . Fusing Depth and RGB data.

---

>## Algorithm's sequence order:
>        1. Try to config only color filters.
>        2. Add post-processing.
>        3. Align depth and color images.
>        4. Calculate weight on ROIs.

---

> # What's been done:
>  - ## Create a dataset of images of cavities classified into 10 classes.
>  - ## Preprocess the dataset by performing operations like resizing, normalization, and data augmentation.
>  - ##  Split the dataset into training and testing sets.
>  - ##  Use TensorFlow's deep learning models like Convolutional Neural Networks (CNNs) or Transfer Learning models like ResNet, Inception, or VGG to train on the dataset.
>  - ##  Evaluate the models' performance by predicting labels for the test set and computing accuracy, precision, recall, F1 scores, and confusion matrix.
>  - ##  Fine-tune the model by adjusting hyperparameters like learning rate, batch size, number of epochs, optimizer, and regularization.
>  - ## Use OpenCV to load and preprocess new images, and then pass them to the trained model for classification.

---
> python3 boundingBox_detector_ros.py -i ../../dataset/PPT/BAG/183847/20230315_183847.bag -m models/MobileNetV2_v2.h5
---

## **Results analysis:**
> ### **bounding-box detector** *v.1* : /bbox_classification/boundingBox_detector_ros.py

| Model | example | **16-04-2023_17-59-15.h5** | **classification_VGG16.h5** | **EfficientNetV2B0_v1.h5** | **EfficientNetV2B0_v2.h5** | **MobileNetV2_v2.h5** | **Pretrained_EfficientNetB0.h5** | **VGG16_v1.h5** | **VGG16_v2.h5** 
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F20_20_horizontal**| <img src="../dataset/PPT/cavity_images_100perCavity/F20_20_horizontal/F20_20_h_001.jpg" width="100" height="100"> | NA | NA | NA | NA | NA | NA | NA | NA |
| **M20_30_horizontal**| <img src="../dataset/PPT/cavity_images_100perCavity/M20_30_horizontal/M20_h_001.jpg " width="100" height="100"> | NA | NA | NA | NA | NA | NA | NA | NA |
| **M20_30_vertical**| <img src="../dataset/PPT/cavity_images_100perCavity/M20_30_vertical/M20_M30_v_001.jpg " width="100" height="100"> | NA | NA | NA | NA | NA | NA | NA | NA |
| **M20_100_horizontal**| <img src="../dataset/PPT/cavity_images_100perCavity/M20_100_horizontal/M20_100_h_001.jpg " width="100" height="100"> | NA | NA | NA | NA | NA | NA | NA | NA |
| **R20_horizontal**| <img src="../dataset/PPT/cavity_images_100perCavity/R20_horizontal/R20_h_001.jpg " width="100" height="100"> | NA | NA | NA | NA | NA | NA | NA | NA |
| **R20_vertical**| <img src="../dataset/PPT/cavity_images_100perCavity/R20_vertical/R20_v_001.jpg " width="100" height="100"> | NA | NA | NA | NA | NA | NA | NA | NA |
| **S40_40_horizontal**| <img src="../dataset/PPT/cavity_images_100perCavity/S40_40_horizontal/S40_40_h_001.jpg " width="100" height="100"> | NA | NA | NA | NA | NA | NA | NA | NA |
| **S40_40_virtical**| <img src="../dataset/PPT/cavity_images_100perCavity/S40_40_virtical/S40_40_v_001.jpg " width="100" height="100"> | NA | NA | NA | NA | NA | NA | NA | NA |

 
 
 
 
 
 
 
 


