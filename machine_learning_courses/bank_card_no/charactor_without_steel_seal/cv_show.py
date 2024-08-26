import cv2
def img_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()