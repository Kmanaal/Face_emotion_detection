#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import os


# In[4]:


print("NumPy:", np.__version__)


# In[5]:


import numpy as np, pandas as pd

arr = np.arange(9).reshape(3,3)
df = pd.DataFrame(arr, columns=["A","B","C"])
print(df)
print(df.describe())


# In[6]:


import pyarrow, numexpr, bottleneck


# In[7]:


print("PyArrow:", pyarrow.__version__)
print("NumExpr:", numexpr.__version__)
print("Bottleneck:", bottleneck.__version__)


# In[8]:


print("Pandas version:", pd.__version__)


# In[9]:


import tensorflow as tf
print(tf.__version__)


# In[11]:


get_ipython().system('pip install opendatasets')


# In[10]:


import opendatasets as od
od.download('https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/data')


# In[11]:


from tensorflow import keras


# In[12]:


from keras.utils import to_categorical
from keras.utils import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


# In[13]:


Train_dir = '/content/face-expression-recognition-dataset/images/train'
test_dir = '/content/face-expression-recognition-dataset/images/validation'


# In[14]:


Train_dir = 'images/train'
test_dir = 'images/test'


# In[15]:


def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,label)):
            image_paths.append(os.path.join(dir,label,imagename))
            labels.append(label)
        print(label,"completed")
    return image_paths,labels


# In[16]:


train = pd.DataFrame()
train['image'], train['label'] = createdataframe(Train_dir)


# In[17]:


print(train)


# In[18]:


test = pd.DataFrame()
test['image'], test['label'] = createdataframe(test_dir)


# In[19]:


print(test)


# In[20]:


from tqdm.notebook import tqdm


# In[21]:


from tqdm.notebook import tqdm
for i in tqdm(range(10)):
    pass


# In[22]:


def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode="grayscale")
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features),48,48,1)
    return features


# In[23]:


train_features = extract_features(train['image'])


# In[24]:


test_features = extract_features(test['image'])


# In[25]:


x_train = train_features/255.0
x_test = test_features/255.0


# **Our model will work in Supervised Learning, because hum uss input kay saath uss input ka label bh deingy, jaisey yeh angry ki image hei yeh sad ki hei yeh happy ki hei. So in supervised learning we give the label along with the image.**

# **Toh label banane kay liye hum label encoder ka use kreingy jo sklearn module mein hota hei**
# 

# In[27]:


from sklearn.preprocessing import LabelEncoder


# In[28]:


le = LabelEncoder()
le.fit(train['label'])


# In[29]:


y_train = le.transform(train['label'])
y_test = le.transform(test['label'])


# In[30]:


y_train = to_categorical(y_train,num_classes=7)
y_test = to_categorical(y_test,num_classes=7)


# In[31]:


model = Sequential()

# Convolutional Layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

# Flatten layer
model.add(Flatten())

# Fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

# Output layer
model.add(Dense(7, activation='softmax'))


# In[32]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test))


# In[35]:


model_json = model.to_json()
with open("emotiondetector.json",'w') as json_file:
  json_file.write(model_json)
model.save("emotiondetector.h5")


# In[32]:


label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# In[33]:


def ef(image):
    img = load_img(image, color_mode="grayscale")   # updated from grayscale=True
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


# In[34]:


image = '/content/face-expression-recognition-dataset/images/validation/angry/10095.jpg'
print("Original image is of angry")

img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]

print("Model prediction is =", pred_label)


# In[35]:


image = '/content/face-expression-recognition-dataset/images/validation/happy/10096.jpg'
print("Original image is of happy")

img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]

print("Model prediction is =", pred_label)


# In[37]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[40]:


image = '/content/face-expression-recognition-dataset/images/validation/disgust/10053.jpg'
print("Original image is of disgust")

img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]

print("Model prediction is =", pred_label)
plt.imshow(img.reshape(48,48), cmap='gray')


# In[42]:


image = '/content/face-expression-recognition-dataset/images/validation/happy/10237.jpg'
print("Original image is of happy")

img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]

print("Model prediction is =", pred_label)
plt.imshow(img.reshape(48,48), cmap='gray')


# In[41]:


image = '/content/face-expression-recognition-dataset/images/validation/fear/10263.jpg'
print("Original image is of fear")

img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]

print("Model prediction is =", pred_label)
plt.imshow(img.reshape(48,48), cmap='gray')


# In[45]:


image = '/content/face-expression-recognition-dataset/images/validation/surprise/1033.jpg'
print("Original image is of surprise")

img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]

print("Model prediction is =", pred_label)
plt.imshow(img.reshape(48,48), cmap='gray')


# In[ ]:




