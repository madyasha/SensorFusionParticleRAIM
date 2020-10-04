import cv2 
import pandas as pd
import numpy as np

def blurring (image):
					#add a suitable blurring to image
		# Gaussian Blurring 
		# Again, you can change the kernel size 
		gausBlur = cv2.GaussianBlur(img, (21,21),0) 
		# Median blurring 
		# medBlur = cv2.medianBlur(img,5) 

		  
		# # Bilateral Filtering 
		# bilFilter = cv2.bilateralFilter(img,9,75,75) 

		return gausBlur
		  
def occlude(image,seed):
	h, w, _ = image.shape
	out_image = None
	if seed % 2== 0:
		out_image = cv2.rectangle(image, (0, 0), (w, h//2), (0, 0, 0), cv2.FILLED)
	elif seed % 3 == 0:
		out_image = cv2.rectangle(image, (0, 0), (w//2,h), (0, 0, 0), cv2.FILLED)
	elif seed % 5 ==0:
		out_image = cv2.rectangle(image, (0, 0), (w//4, h//4), (0, 0, 0), cv2.FILLED)
	else:
		out_image = cv2.rectangle(image, (0, 0), (w//3, h//3), (0, 0, 0), cv2.FILLED)
	
	return out_image

mode = 'blur'
dest_path = 'C:/Users/JNM/Downloads/ParticleRAIM/REAR IMAGES/BLURRED IMAGES'

file_path = 'C:/Users/JNM/Downloads/ParticleRAIM/REAR IMAGES/REAR IMAGES'
#excel_file= 'camera_test.xlsx'
excel_file= 'camera_train.xlsx'
df = pd.read_excel(excel_file, usecols="A:D")
names= df['Name']

for i in range(len(names)):
	fname = names[i]
	print (file_path + "/" + fname)
	img = cv2.imread(file_path + "/" + fname)

	if i % 3 == 0:
		if mode == 'blur':
			out_img = blurring (img)
		elif mode =='occlude':
			seed = np.random.randint(2, high=8, size=1, dtype='l')
			out_img = occlude(img,seed)
		# cv2.imshow('modified_image',out_img)
		# cv2.waitKey()		
		cv2.imwrite(dest_path + "/" + fname,out_img)

	else:
		cv2.imwrite(dest_path + "/" + fname,img)











 




