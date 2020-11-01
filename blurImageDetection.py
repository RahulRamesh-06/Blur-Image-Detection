import numpy as np 

import os

from sklearn import preprocessing, svm
from scipy.ndimage import variance
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler

#Declared the list and iteratable indexs 
i=0
j=0
sharp_laplaces=[]
blurry_laplaces=[]

def getVarianceAndMaxi(img_type) :
  obtained_laplaces=[]
  if(img_type=='blur') :
    path='/media/blur'
  elif (img_type=='sharp') :
    path='/media/sharp'

  for image_path in os.listdir(path):
   print(f"img_path: {image_path}")
   img = io.imread(path+'/'+image_path)

# Resizing the image.
   img = resize(img, (400, 600))

# Convert the image to greyscale
   img = rgb2gray(img)

# Detecting the Edge of the image
   edge_laplace = laplace(img, ksize=3)

   print(f"Variance: {variance(edge_laplace)}")
   print(f"Maximum : {np.amax(edge_laplace)}")

   # Adding the variance and maximum to a list of sets
   obtained_laplaces.insert(i+1,((variance(edge_laplace),(np.amax(edge_laplace)))))
   if(img_type=='blur') :
    print(f"blur_obtained_laplaces : {obtained_laplaces}")
   elif (img_type=='sharp') :  
    print(f"sharp_laplaces : {obtained_laplaces}")
  return obtained_laplaces 
 
sharp_laplaces=getVarianceAndMaxi('sharp')  
blurry_laplaces=getVarianceAndMaxi('blur')

# set class labels (non-blurry / blurry) and prepare features
y = np.concatenate((np.ones((76, )), np.zeros((76, ))), axis=0)
laplaces = np.concatenate((np.array(sharp_laplaces), np.array(blurry_laplaces)), axis=0)

# scale features
laplaces = preprocessing.scale(laplaces)

# train the classifier (support vector machine)
clf = svm.SVC(kernel='linear', C=100000)
clf.fit(laplaces, y)

scaler = StandardScaler()
print(scaler.fit(sharp_laplaces))
print(f'Mean Value: {scaler.mean_}')
print(f'Scaler Value: {scaler.scale_}')
print(f'Weights: {clf.coef_[0]}')
print(f'Intercept: {clf.intercept_}')


scaled_value_1 = 0.00898692
scaled_value_2 = 0.34110443

Scalar_mean_1 = 0.00848896
Scalar_mean_2 = 1.11494341

weighted_avg_1=22.28791672
weighted_avg_2=2.80257642

intercept=9.61884965
variance= 0.0007378721487632876
max=1.0196819469802882

#  // standardize based on the variance and maximum of the image to be classified with the values obtained from the trained dataset
standarized_value_1 = (variance - Scalar_mean_1) / scaled_value_1;
standarized_value_2 = (max - Scalar_mean_2) / scaled_value_2;
print(f'standarized_value_1:{standarized_value_1}')
print(f'standarized_value_2:{standarized_value_2}')
 # // predict the image based on the weighted avg and std
predicted_value = (weighted_avg_1 * standarized_value_1 + weighted_avg_2 * standarized_value_2) + intercept;

# Printing the value obtained.
print(f'value obtained:{predicted_value}')

if (predicted_value < 0) :
  print(' The given image is blurred') 
elif (predicted_value < 1) :
  print('The given image is likely blurred')
else :
  print('The given image is sharp') 
  


# print parameters
# print(f'Weights: {clf.coef_[0]}')
# print(f'Intercept: {clf.intercept_}')

# make sample predictions

# clf.predict([[0.0001949989738491614, 0.11942989103266866]])  # result: 0 (blurred)
# clf.predict([[0.008667345877673704, 0.975144409150345]])  # result: 1 (sharp)

# print(f'blur: {clf.predict([[ 0.0017708252585035996,  0.6858889134117716]])}')
# print(f'sharp: {clf.predict([[0.006140374621412874, 1.1548249444706242]])}')




