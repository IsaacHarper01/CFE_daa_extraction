import cv2
from pyzbar.pyzbar import decode
import numpy as np

img = cv2.imread('dataset2/IMG_20230418_115707.jpg')
img =cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
img = cv2.resize(img,[1156,520]) 
data = decode(img)
print(data)
for barcode in data:
    points = np.array([barcode.polygon],np.int32)
    points = points.reshape((-1,1,2))
    print(barcode.polygon[0][0])
    print(barcode.polygon[0][1])
    cv2.circle(img,(barcode.polygon[2][0],barcode.polygon[2][1]),5,(255,0,255),-1)
cv2.imshow("IMAGE",img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
