##importing libraries
import numpy as np
import pandas as pd

##load dataset
dataset = pd.read_csv('digit.csv') #MNIST Digit Dataset

print(dataset.shape)

##segregating data into X and y
X=dataset.iloc[:,1:]
y=dataset.iloc[:,0]

# print(X.shape)
# print(y)

##splitting data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


##model Training
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

##model prediction
y_pred = clf.predict(X_test)

print(clf.score(X_test, y_test))

##accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:{0}".format(accuracy*100))



import cv2  

def resize_image(image_path, num_rows, num_cols):  
    # Read the image  
    image = cv2.imread(image_path)  
    
    # Get current dimensions  
    height, width = image.shape[:2]  
    
    # Calculate the aspect ratio  
    aspect_ratio = width / height  
    
    # Compute the new dimensions  
    new_width = int(num_cols * (height / num_rows) * aspect_ratio)  
    new_height = int(num_rows * (width / num_cols) / aspect_ratio)  
    
    # Resize the image  
    resized_image = cv2.resize(image, (new_width, new_height))  
    
    return  resized_image

resize_image('images.jpg', 28, 28)

from PIL import Image  

# Load the image  
img = Image.open('resized_image.jpg')

# Convert image to grayscale (optional, if you need grayscale values)  
img = img.convert('L')  

# Get pixel values  
pixel_values = list(img.getdata())  

# Print pixel values  
# print(pixel_values)

y_pred1 = clf.predict((pixel_values))
print(y_pred1)