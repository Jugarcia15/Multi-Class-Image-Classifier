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

### Access via Heroku
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



###Access via Google Colab (Gradio Implementation, this method requires a Google Account):


1.	Use any modern web browser, go to https://colab.research.google.com/drive/1kxXufKbPevp8Ewb1095ZJqBx5aDPLfeV?usp=sharing



2.	Sign into a Google Account. After signing in, your browser should look similar to below:
 
 ![image](https://user-images.githubusercontent.com/41842178/178398024-1ef58896-058d-4580-bd62-6274b2d1b083.png)

 
3.	On the top right-hand side of the page, click ‘Connect’ to connect to Colab runtime. You will be connected once you see ‘RAM’ and ‘Disk’ replace the ‘Connect’ button. 
 	 
   ![image](https://user-images.githubusercontent.com/41842178/178398008-52aa84c6-9801-4c97-87ac-00cdede69cde.png)
![image](https://user-images.githubusercontent.com/41842178/178398017-47d46b4a-ba84-4bc1-88e2-edbdf7d5cb72.png)

   
4.	Once Connected, click on ‘Runtime’ on the top left, and select ‘Run All’ from the drop-down menu. 
 	 
![image](https://user-images.githubusercontent.com/41842178/178397981-385070fe-1a1f-450b-9254-453d405a5544.png)
![image](https://user-images.githubusercontent.com/41842178/178397991-2b87885f-cd65-48f9-ae7b-a763271b25af.png)

   
5.	You will receive this warning from Google. Click ‘Run Anyway’ to proceed. The Colab notebook will clone files from a personal GitHub repo that contains the trained model and a folder of test images in zipped format. The notebook immediately unzips them. The files are temporarily stored in the virtual runtime environment and will be deleted once the user disconnects. The files can be found in the sidebar on the left.
  
![image](https://user-images.githubusercontent.com/41842178/178397932-34cc44c5-73e2-4c87-a341-1ed1a85a8dd6.png)
![image](https://user-images.githubusercontent.com/41842178/178397938-55dd0285-729a-4d93-a14d-54a2b18c2d44.png)

  
6.	Scroll down to the last cell, the cell containing the Gradio interface should be running. The notebook should look like this: 
 
 ![image](https://user-images.githubusercontent.com/41842178/178397914-2e7d819d-a56f-4df3-98b0-7e2d97ce7c44.png)

 
7.	Drag and drop or click ‘Click to Upload’ an Image in the image box in the Gradio Interface.
 
 ![image](https://user-images.githubusercontent.com/41842178/178397900-f894eb24-8dca-4b17-9b60-b65edc632d6e.png)

 
8.	Select an image to upload and click ‘Open’.

![image](https://user-images.githubusercontent.com/41842178/178397887-fdf75c39-17ca-4088-bd2c-7a26163fd7fb.png)

 
9.	Click ‘Submit’ to submit your image and generate predictions based on that image. 
 
 ![image](https://user-images.githubusercontent.com/41842178/178397875-7d56f015-465d-44d4-b6dd-bd381c46e2af.png)

 
10.	Image predictions will be displayed on the right in the ‘output’ section. 
 
 ![image](https://user-images.githubusercontent.com/41842178/178397866-0ad1c261-f038-491e-a94a-1161b0612a56.png)

 
11.	Click ‘Clear’ to start over from step 7 or click the cell stop button located on the top left area of the cell. 
 
 ![image](https://user-images.githubusercontent.com/41842178/178397861-0b03271b-7be2-4b81-967e-65de43fcd815.png)

 
12.	Click the down arrow located next to where the ‘Connect’ button was located and select ‘Disconnect and delete runtime’ in order to conclude the session. 
 
![image](https://user-images.githubusercontent.com/41842178/178397850-de4c222d-497a-447d-b14b-4f17f720ab58.png)

