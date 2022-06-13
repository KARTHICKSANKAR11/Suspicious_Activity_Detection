import cv2 
import os 

#path = text
# Read the video from specified path 
#cam = cv2.VideoCapture(path) 
cam = cv2.VideoCapture(r"E:\Study\Project\Suspicious2\Real Life Violence Dataset\Violence\V_1.mp4") 
try: 
	
	# creating a folder named data 
	if not os.path.exists(r'E:\Study\Project\Suspicious2\frames'): 
		os.makedirs(r'E:\Study\Project\Suspicious2\frames') 

# if not created then raise error 
except OSError: 
	print ('Error: Creating directory of data') 

# frame 
currentframe = 0

while(True): 
	
	# reading from frame 
	ret,frame = cam.read()

	if ret: 
		# if video is still left continue creating images 
		name = 'E:\Study\Project\Suspicious2/frames/' + str(currentframe) + '.jpg'
        #new_img= name.resize((256,256))
		print('Creating...' + name) 
		# writing the extracted images 
		cv2.imwrite(name, frame) 

		# increasing counter so that it will 
		# show how many frames are created 
		currentframe += 1
	else: 
		break

# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 
