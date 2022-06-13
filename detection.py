################################################################################
#Normal and Abnormal detection
################################################################################
print("\n Normal and Abnormla Detection")

#Load the model
from tensorflow.keras.models import load_model

classifier_model= load_model(r'E:\Study\Project\Suspicious2/trianed_model.h5')

#Image Prediction 
import tensorflow
import cv2
import os
from tensorflow.keras.preprocessing import image
#import matplotlib.pyplot as plt
import numpy as np
import os.path as op
from playsound import playsound
import ctypes
import winsound
def sound():
    playsound(r"C:\Users\karth\OneDrive\Desktop\m\vadivel.mp3")
img_path = r'E:\Study\Project\Suspicious2\frames'
img_new_path= r'E:\Study\Project\Suspicious2\abnormal'
current_frame = 0
count =0
while(True):
    if(not op.exists(img_path + '\\'+ str(current_frame) +'.jpg')):
        break
    input_image= image.load_img(img_path + '\\'+ str(current_frame) +'.jpg', target_size=(64, 64))
    img_1= image.img_to_array(input_image)
    #print(img_1)
    img_1 = img_1/255
    #print(img_1)

    img_1 = np.expand_dims(img_1, axis=0)
    prediction = classifier_model.predict(img_1, batch_size=None,steps=1) 
    print("Value",prediction)
    if(prediction[:,:]>0.5):
        output ='Normal :%1.2f'%(prediction[0,0])
        #plt.text(20, 62,output,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
        
    else:
        if not os.path.exists(r'E:\Study\Project\Suspicious2\abnormal'):
            os.makedirs(r'E:\Study\Project\Suspicious2\abnormal')
        image1=cv2.imread(img_path + '\\'+ str(current_frame) +'.jpg')    
        #cv2.imshow("image",image)
        cv2.imwrite(img_new_path + '\\'+ str(current_frame) +'.jpg',image1)
        count+=1
        output ='Abnormal :%1.2f'%(prediction[0,0])
        if(count == 5):
            #ctypes.windll.user32.MessageBoxW(0, "PLEASE ALERT", "ABNORMAL", 16)
            #sound()
            #winsound.Beep(2500,3000)
            count=0
        output ='Abnormal :%1.2f'%(prediction[0,0])
        #plt.text(20, 62,output,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
    print('\tFrame' + str(current_frame) + ' = ' + output)
    current_frame += 1
    
