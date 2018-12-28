#You can input the image path directly from the command line
#For example: python predict.py image.jpg

from keras.models import load_model
from PIL import Image as PilImage
from keras.preprocessing import image
import numpy
import os
import sys

classifier = load_model("vehicle_cnn.h5")
img_path = sys.argv[1]

img = PilImage.open(img_path)
img = img.resize((32,32), PilImage.ANTIALIAS)
tmp_path= "tmp3245675289499032485.jpg"
img.save(tmp_path)

prediction_image = image.load_img(tmp_path,(32,32))
prediction_image = image.img_to_array(prediction_image)
prediction_image = numpy.expand_dims(prediction_image,axis=0)
os.remove("tmp3245675289499032485.jpg")
prediction = classifier.predict(prediction_image)[0]

print({"Airplane":prediction[0],"Automobile":prediction[1],"Ship":prediction[2],"Truck":prediction[3]})
