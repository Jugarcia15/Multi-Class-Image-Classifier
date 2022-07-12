# Multi-Class-Image-Classifier
 Updated Mushroom Image Classifier. Only includes the streamlit python file, and the notebook that was used in Google Colab.  Project Backup repository holds the dataset and log data generated from the creation of multiple models.  


# End to End Multiclass Mushroom Species Classification (GitHub)
This Notebook builds an end to end multiclass image classifier using TensorFlow and TensorFlow Hub.

## Problem
Identifying the species of mushrooms given an image of a mushroom.

When I come across a wild mushroom, I typically take a picture of it so that I can determine what species it is. 

## Data
The data we're utilizing is available on Kaggle.
Available at the following urls:
https://www.kaggle.com/datasets/harperd17/mushroom-pictures

https://www.kaggle.com/datasets/derekkunowilliams/mushrooms

## Features
Some information about the data:
The dataset consists of numerous images (unstructured data).
There are approximately 2800+ images in the dataset. We expect to use .1 for validation.

This data set consists of 6 different species ( this means there are 6 different classes) with quantities specified.

Amanita bisporigera - 606

Amanita muscaria - 367

Boletus edulis - 444

Cantharellus - 1183

Omphalotus olearius - 59

Russula mariae - 235

2894 -1 = 2893 photos in first data set.

A part of a second dataset was merged with the first set:

48 photos of Amanita Bisporigera

51 photos of Amanita Muscaria

64 BOletus Edulis

50 Cantharellus

33 Omphalotus orealius

246 images from second dataset

3139 total images

Omphalotus Orealius was omitted due to low number of images.

### **Total Number of Images = 3047**


1. Use any modern web browser, go to https://mush-image-classifier-2.herokuapp.com/

![image](https://user-images.githubusercontent.com/41842178/178397432-1770275c-97c2-4b17-a2ab-e063407ef65f.png)


 
2. Click ‘Browse Files’ in the side bar.	

 ![image](https://user-images.githubusercontent.com/41842178/178397417-0b5d8e9e-58dd-4558-a84f-4b626bc14fcd.png)

 
 
3. Select a mushroom image you want to classify. Then click ‘Open’. There is a test image labeled ‘sample_image_rm.jpg’ in the submission folder. It is an image of a Russula Mariae mushroom. 
 
![image](https://user-images.githubusercontent.com/41842178/178397373-6f7c0219-7fa5-4cfb-b19c-1e81d8856ba2.png)

  
  
3. Click ‘Predict Image’ in the sidebar to predict the image you have uploaded. 

![image](https://user-images.githubusercontent.com/41842178/178397355-92e54bcd-0057-4096-ad0a-c37d58dd4437.png)

 
 
4. Prediction results will be displayed underneath the site title.

 ![image](https://user-images.githubusercontent.com/41842178/178397340-540feac4-926f-4d4e-bf42-3c542659d88d.png)

