import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
from keras.models import load_model
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
threshold=0.90
cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font=cv2.FONT_HERSHEY_COMPLEX
model = load_model('MyTrainingModel.h5')

def preprocessing(img):
    img=img.astype("uint8")
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img = img/255
    return img


def get_className(classNo):
	if classNo==0:
		return "Mask"
	elif classNo==1:
		return "No Mask"


while True:
	sucess, imgOrignal=cap.read()
	faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
	for x,y,w,h in faces:
		# cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
		# cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
		crop_img=imgOrignal[y:y+h,x:x+h]
		img=cv2.resize(crop_img, (32,32))
		img=preprocessing(img)
		img=img.reshape(1, 32, 32, 1)
		# cv2.putText(imgOrignal, "Class" , (20,35), font, 0.75, (0,0,255),2, cv2.LINE_AA)
		# cv2.putText(imgOrignal, "Probability" , (20,75), font, 0.75, (255,0,255),2, cv2.LINE_AA)
		prediction=model.predict(img)
		classIndex=model.predict_classes(img)
		probabilityValue=np.amax(prediction)
		if probabilityValue>threshold:
			if classIndex==0:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
				cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			elif classIndex==1:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
				cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)

			# cv2.putText(imgOrignal,str(round(probabilityValue*100, 2))+"%" ,(180, 75), font, 0.75, (255,0,0),2, cv2.LINE_AA)
	cv2.imshow("Result",imgOrignal)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()














