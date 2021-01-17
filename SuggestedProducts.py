import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random
import tensorflow as tf


#keys of the dictionary are the image id
allItems = {}
path = './dataset'
flag = 0
labels = {}

#function to add a new item to the dictionary
def additem(label,i,price):
    #new_item = Shopify_Item(label,i,price)
    item1 = { 'id': i, 'label': label, 'price': price}
    allItems[i] = item1

#get ride of pesky file
for subdir, dirs, files in os.walk(path):
    for img in files:
        if img == ".DS_Store":
            os.remove(os.path.join(subdir,img))

for file in os.listdir("./test_images"):
    if file == ".DS_Store":
            print("here")
            os.remove(os.path.join("./test_images", file))


for subdir, dirs, files in os.walk(path):
    curr_label = os.path.basename(subdir)
    labels[curr_label] = []
    for img in files:
        price =  random.randint(20, 60)
        additem(curr_label,img,price)
        labels[curr_label].append(img)


CATEGORIES = ['black_shoes','red_dress','black_shirt', 'white_dress', 'black_pants', 'blue_dress',
              'red_shoes', 'blue_pants', 'white_shoes', 'blue_shirt', 'blue_shoes']


def prepare(file):
    IMG_SIZE = 32
    path = './test_images'
    path=os.path.join(path,file)
    img_array = cv2.imread(path)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)).flatten()
    return new_array.reshape(1, IMG_SIZE, IMG_SIZE, -1)


@tf.function(experimental_relax_shapes=True)
def predict(x):
     return model(x)


#testing
flag = 0
model = tf.keras.models.load_model("CNN.model")
path = './test_images'
IMG_SIZE = 300

for files in os.listdir(path):
    label_act = files.split("_test")
    image = prepare(files)
    prediction = predict([image])
    prediction = list(prediction[0])
    label_pred = CATEGORIES[prediction.index(max(prediction))]

    #display query product
    query = cv2.imread(os.path.join(path,files))
    image = cv2.resize(query, (IMG_SIZE, IMG_SIZE))
    cv2.imshow("Query", image)

    #set text parameters
    window_name = 'Suggested products'
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    color = (0, 0, 255)
    fontScale = 1

    path2 = "./dataset/"+label_pred
    items1 = labels[label_pred]

    #display 3 suggested products
    index = random.randint(20, len(items1)-1)
    product = allItems[items1[index]]
    price1 = "$"+str(product['price'])+".00"
    org1 = (15, 30)
    result = cv2.imread(os.path.join(path2,items1[index]))
    image1 = cv2.resize(result, (IMG_SIZE, IMG_SIZE))

    index = random.randint(20, len(items1)-1)
    product = allItems[items1[index]]
    price2 = "$"+str(product['price'])+".00"
    org2 = (15, 30)
    result = cv2.imread(os.path.join(path2,items1[index]))
    image2 = cv2.resize(result, (IMG_SIZE, IMG_SIZE))

    index = random.randint(20, len(items1)-1)
    product = allItems[items1[index]]
    price3 = "$"+str(product['price'])+".00"
    org3 = (15, 30)
    result = cv2.imread(os.path.join(path2,items1[index]))
    image3 = cv2.resize(result, (IMG_SIZE, IMG_SIZE))

    image1 = cv2.putText(image1, price1, org1, font, fontScale,
                 color, thickness, cv2.LINE_AA, False)
    image2 = cv2.putText(image2, price2, org2, font, fontScale,
                 color, thickness, cv2.LINE_AA, False)
    image3 = cv2.putText(image3, price3, org3, font, fontScale,
                 color, thickness, cv2.LINE_AA, False)

    Together = np.concatenate((image1, image2, image3), axis=0)

    cv2.imshow('Suggested products', Together)

    cv2.waitKey(0)

cv2.destroyAllWindows()
