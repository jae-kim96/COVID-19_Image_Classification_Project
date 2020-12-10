# COVID-19_Image_Classification_Project
A project for classifying chest X-Ray images for COVID-19, pneumonia, and normal X-rays using a CNN classifier. This project was for Georgia Tech OMSCS CSE6250: Big Data for Health Course. Project was completed by Jaeyong Kim, Muhammad Hassan and Omar Mohammadi. This GitHub repository highlights the work completed on this project by me (Jaeyong Kim). I mainly developed the code for Preprocessing of the images and a Spark/Keras CNN to classify the images.

## Image Preprocessing
Images were pulled from a Kaggle Dataset (https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) and a GitHub (https://github.com/ieee8023/covid-chestxray-dataset).

X-Ray Images were preprocessed using a Python Script to resize the images to 224 x 224 x 3 and introduce random horizontal flip to the images to increase the possibility of symptoms occuring in the Lung Region.

## Spark/Keras CNN
The CNN was created utilizing Spark in the form of PySpark and the Keras Python package. 

PySpark was used to load the images into dataframes and extract the features like pixel values. The CNN contains 13 Convolutional Layers, each followed by a Leaky Relu activation and Max Pooling Layer. A soft-max activation is added at the end to classify the images. The layers and dimensions are shown below:

The CNN took aboout 48 minutes to train, with the Validation Accuracy coming out to 0.8936 and Loss coming out to 1.815.

Covid X-Rays were predicted with a Precision of 0.86 and Recall of 0.75. I learned that this model had the tendency to overfit because the Loss curve began to increase for the Validation set, while the Accuracy showed a plateau. There was a small number of COVID images to work with, so there was a high chance of over-fitting. For future works, I would look into Transfer Learning with a pretrained model, and using that to help with feature extraction for this model.

# Model: CNN for Covid X-Ray Classification
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_191 (Conv2D)          (None, 224, 224, 8)       224       
_________________________________________________________________
leaky_re_lu_99 (LeakyReLU)   (None, 224, 224, 8)       0         
_________________________________________________________________
max_pooling2d_101 (MaxPoolin (None, 112, 112, 8)       0         
_________________________________________________________________
conv2d_192 (Conv2D)          (None, 112, 112, 16)      1168      
_________________________________________________________________
leaky_re_lu_100 (LeakyReLU)  (None, 112, 112, 16)      0         
_________________________________________________________________
max_pooling2d_102 (MaxPoolin (None, 56, 56, 16)        0         
_________________________________________________________________
conv2d_193 (Conv2D)          (None, 56, 56, 32)        4640      
_________________________________________________________________
leaky_re_lu_101 (LeakyReLU)  (None, 56, 56, 32)        0         
_________________________________________________________________
max_pooling2d_103 (MaxPoolin (None, 28, 28, 32)        0         
_________________________________________________________________
conv2d_194 (Conv2D)          (None, 28, 28, 16)        4624      
_________________________________________________________________
leaky_re_lu_102 (LeakyReLU)  (None, 28, 28, 16)        0         
_________________________________________________________________
max_pooling2d_104 (MaxPoolin (None, 14, 14, 16)        0         
_________________________________________________________________
conv2d_195 (Conv2D)          (None, 14, 14, 32)        4640      
_________________________________________________________________
leaky_re_lu_103 (LeakyReLU)  (None, 14, 14, 32)        0         
_________________________________________________________________
max_pooling2d_105 (MaxPoolin (None, 7, 7, 32)          0         
_________________________________________________________________
conv2d_196 (Conv2D)          (None, 7, 7, 64)          18496     
_________________________________________________________________
leaky_re_lu_104 (LeakyReLU)  (None, 7, 7, 64)          0         
_________________________________________________________________
max_pooling2d_106 (MaxPoolin (None, 4, 4, 64)          0         
_________________________________________________________________
conv2d_197 (Conv2D)          (None, 4, 4, 32)          18464     
_________________________________________________________________
leaky_re_lu_105 (LeakyReLU)  (None, 4, 4, 32)          0         
_________________________________________________________________
max_pooling2d_107 (MaxPoolin (None, 2, 2, 32)          0         
_________________________________________________________________
conv2d_198 (Conv2D)          (None, 2, 2, 64)          18496     
_________________________________________________________________
leaky_re_lu_106 (LeakyReLU)  (None, 2, 2, 64)          0         
_________________________________________________________________
max_pooling2d_108 (MaxPoolin (None, 1, 1, 64)          0         
_________________________________________________________________
conv2d_199 (Conv2D)          (None, 1, 1, 128)         73856     
_________________________________________________________________
leaky_re_lu_107 (LeakyReLU)  (None, 1, 1, 128)         0         
_________________________________________________________________
max_pooling2d_109 (MaxPoolin (None, 1, 1, 128)         0         
_________________________________________________________________
conv2d_200 (Conv2D)          (None, 1, 1, 64)          73792     
_________________________________________________________________
leaky_re_lu_108 (LeakyReLU)  (None, 1, 1, 64)          0         
_________________________________________________________________
max_pooling2d_110 (MaxPoolin (None, 1, 1, 64)          0         
_________________________________________________________________
conv2d_201 (Conv2D)          (None, 1, 1, 256)         147712    
_________________________________________________________________
leaky_re_lu_109 (LeakyReLU)  (None, 1, 1, 256)         0         
_________________________________________________________________
max_pooling2d_111 (MaxPoolin (None, 1, 1, 256)         0         
_________________________________________________________________
conv2d_202 (Conv2D)          (None, 1, 1, 128)         295040    
_________________________________________________________________
leaky_re_lu_110 (LeakyReLU)  (None, 1, 1, 128)         0         
_________________________________________________________________
max_pooling2d_112 (MaxPoolin (None, 1, 1, 128)         0         
_________________________________________________________________
conv2d_203 (Conv2D)          (None, 1, 1, 256)         295168    
_________________________________________________________________
leaky_re_lu_111 (LeakyReLU)  (None, 1, 1, 256)         0         
_________________________________________________________________
max_pooling2d_113 (MaxPoolin (None, 1, 1, 256)         0         
_________________________________________________________________
flatten_10 (Flatten)         (None, 256)               0         
_________________________________________________________________
dense_19 (Dense)             (None, 256)               65792     
_________________________________________________________________
leaky_re_lu_112 (LeakyReLU)  (None, 256)               0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_20 (Dense)             (None, 3)                 771       
=================================================================
Total params: 1,022,883
Trainable params: 1,022,883
Non-trainable params: 0
_________________________________________________________________












