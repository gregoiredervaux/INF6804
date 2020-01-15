import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os

data_root = "./dataset"
results_root = "./results"
dataset_categories = os.listdir(data_root)
dict_video = {}
for folder in dataset_categories:
    dict_video[folder] = os.listdir(data_root + "/" + folder)
path = data_root + "/" + 'badWeather' + "/" + dict_video['badWeather'][0] + "/" + "input" + "/"
print(path)
image1 = cv2.imread(path + 'in000100.jpg',cv2.IMREAD_GRAYSCALE)
plt.imshow(image1, cmap = plt.get_cmap('gray'))
plt.show()
image2 = cv2.imread(path + 'in000200.jpg',cv2.IMREAD_GRAYSCALE)
plt.imshow(image2, cmap = plt.get_cmap('gray'))
plt.show()
# On calcule le flux optique. On obtiendra une valeur de Vx, Vy pour chaque pixel
flot = cv2.calcOpticalFlowFarneback(image1,image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
print("forme du flot: " + str(flot.shape))# on recupère la nomre du vecteur V, composé de Vx et Vy,
# en passant en coordonnées polaires
mag, _ = cv2.cartToPolar(flot[...,0], flot[...,1])
# on peut reconstruire notre image en nuance de gris
img = np.uint8(cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX))
plt.imshow(img, cmap = plt.get_cmap('gray'))
plt.show()
# On essaye de supprimer les pixels presques noirs
histogram = np.bincount(img.flatten())
plt.plot(range(0,255), histogram)
plt.xlabel('pixels')
plt.ylabel('numbers of pixels')
plt.title('Histogram of the number of pixel by luminosity')
plt.show()

seuil = 50

def flot_optique(img1, img2, seuil, print_img=True):
    image1 = cv2.imread(img1,cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(img2,cv2.IMREAD_GRAYSCALE)
    # On calcule le flux optique. Cela nous donnera une valeur de Vx, Vy pour chaque pixel
    flot = cv2.calcOpticalFlowFarneback(image1,image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flot[...,0], flot[...,1])
    # on peut reconstruire notre image en nuance de gris
    img = np.uint8(cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX))
    for index, pix in np.ndenumerate(img):
        img[index] = pix if pix > seuil else 0
    if print_img:
        fig = plt.figure(figsize=(16,16))
        original = fig.add_subplot(1,2,1)
        original_plot = plt.imshow(image2, cmap = plt.get_cmap('gray'))
        original.set_title('Original')

        processed = fig.add_subplot(1,2,2)
        processed_plot = plt.imshow(img, cmap = plt.get_cmap('gray'))
        processed.set_title('Processed')
        plt.show()
    return img

img = flot_optique(path + 'in000500.jpg', path + 'in000510.jpg', seuil, False)
path_gt = data_root + "/" + 'badWeather' + "/" + dict_video['badWeather'][0] + "/" + "groundtruth" + "/"

grd_truth = cv2.imread(path_gt + 'gt000510.png',cv2.IMREAD_GRAYSCALE)
img_merge = np.zeros((img.shape[0], img.shape[1]), dtype='int8')

TP = 0
FP = 0
TN = 0
FN = 0

for index,_ in np.ndenumerate(img_merge):
    if img[index] > 0:
        if grd_truth[index] > 170:
            img_merge[index] = 255
            TP += 1
        else:
            FP += 1
    else:
        if grd_truth[index] > 170:
            FN += 1
        else:
            TN += 1
tot = TP+FP+TN+FN
array = np.array(
        [[TP, int(TP * 100 /tot)],
         [FP, int(FP * 100 /tot)],
         [TN, int(TN * 100 /tot)],
         [FN, int(FN * 100 /tot)]])
prt = pd.DataFrame(
    array,
    index=['True positif', 'False positif', 'True negative', 'False negatif'],
    columns=['numbers', 'purcentage'])

print(prt)

fig = plt.figure(figsize=(16,16))
processed = fig.add_subplot(2,2,1)
processed_plot = plt.imshow(img, cmap = plt.get_cmap('gray'))
processed.set_title('Processed')

gt = fig.add_subplot(2,2,3)
gt_plot = plt.imshow(grd_truth, cmap = plt.get_cmap('gray'))
gt.set_title('Groundtruth')

merged = fig.add_subplot(2,2,4)
merged_plot = plt.imshow(img_merge, cmap = plt.get_cmap('gray'))
merged.set_title('Merged')

plt.show()
