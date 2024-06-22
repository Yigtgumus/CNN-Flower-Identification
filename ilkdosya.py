import numpy as np  
import pandas as pd 
import cv2 
import matplotlib.pyplot as plt 
from PIL import Image 
import keras.preprocessing.image 
from keras.preprocessing.image import ImageDataGenerator, load_img 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.optimizers import Adam 
import tensorflow as tf 
import os
from bs4 import BeautifulSoup
import requests

base_dir = r'D:\flowers'
  
img_size = 224
batch = 64

# Create a data augmentor 
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2,  
                                   zoom_range=0.2, horizontal_flip=True, 
                                   validation_split=0.2) 
  
test_datagen = ImageDataGenerator(rescale=1. / 255, 
                                  validation_split=0.2) 
  
# Create datasets 
train_datagen = train_datagen.flow_from_directory(base_dir, 
                                                  target_size=( 
                                                      img_size, img_size), 
                                                  subset='training', 
                                                  batch_size=batch) 
test_datagen = test_datagen.flow_from_directory(base_dir, 
                                                target_size=( 
                                                    img_size, img_size), 
                                                subset='validation', 
                                                batch_size=batch) 

# # modelling starts using a CNN. 
  
model = Sequential() 
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', 
                 activation='relu', input_shape=(224, 224, 3))) 
model.add(MaxPooling2D(pool_size=(2, 2)))


  
  
model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 
  
  
model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 
  
model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 
  
model.add(Flatten()) 
model.add(Dense(512)) 
model.add(Activation('relu')) 
model.add(Dense(8, activation="softmax")) 
#model.summary()


"""model.compile(optimizer=keras.optimizers.Adam(), 
        loss='categorical_crossentropy', metrics=['accuracy'])
epochs=40
model.fit(train_datagen,epochs=epochs,validation_data=test_datagen)"""



from keras.models import load_model 

"""model.save('Model.keras')
print("Model Saved !")"""


# load model 
savedModel=load_model('Model.keras')
print("Model Loaded !")
print(train_datagen.class_indices)
 

from keras.preprocessing import image 
  
 
list_ = ['Zambak','Lotus','Orkide','Papatya', 'Karahindiba','Gül','Ayçiçeği','Lale'] 
  

test_image = image.load_img('bitirme_projesi_vs/sunflower-1627193_1280.jpg',target_size=(224,224)) 
  

plt.imshow(test_image)
test_image = image.img_to_array(test_image) 
test_image = np.expand_dims(test_image,axis=0) 
  

result = savedModel.predict(test_image) 
print(result) 
  

i=0
for i in range(len(result[0])): 
  if(result[0][i]==1): 
    
    break
c_ismi=str(list_[i])
print(c_ismi)
print("-------------------------")



url="https://www.wikipedia.tr-tr.nina.az/"+c_ismi+".html"
print("-ÇİÇEĞİN ÖZELLİKLERİ-")

r = requests.get(url)
html_page=r.content

soup = BeautifulSoup(html_page, 'html.parser')
texts = soup.find_all('p')

for text in texts:
    
    if text ==texts[0]:pass
    else:
     print("--------------")
     print(text.get_text())









