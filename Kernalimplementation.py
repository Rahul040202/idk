#Kernal implementation

import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
x=train_images[1]
plt.imshow(x)
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
loaded_image = cv2.imread("plot.png")
loaded_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(gray_image,(3,3),0)
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
img_prewitt=img_prewittx+img_prewitty
plt.subplot(2,2,3),plt.imshow(img_prewittx,cmap = 'gray')
plt.title('Prewitt X')
plt.subplot(2,2,3),plt.imshow(img_prewitty,cmap = 'gray')
plt.title('Prewitt Y')
plt.subplot(2,2,3),plt.imshow(img_prewitt,cmap = 'gray')
plt.title('Prewitt XY')
plt.show()
img = cv2.medianBlur(gray_image,3)
kernelxx = np.array([[1, 0], [0, -1]])
kernelyy = np.array([[0, 1], [-1, 0]])
img_robertx = cv2.filter2D(img, -1, kernelxx)
img_roberty = cv2.filter2D(img, -1, kernelyy)
grad = cv2.addWeighted(img_robertx, 0.5, img_roberty, 0.5, 0)
plt.imshow(grad,cmap='gray')
plt.title("RobertEdgeDetection")
plt.show()
