import cv2
from keras.models import load_model
import tensorflow as tf
import numpy as np

model = load_model('mnist.h5')
img = cv2.imread("5.png")
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
new_img = tf.keras.utils.normalize(resized, axis=1)
new_img = np.array(new_img).reshape(-1, 28, 28, 1)
pred = model.predict(new_img)

print(np.argmax(pred))
