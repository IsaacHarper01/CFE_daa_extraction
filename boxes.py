import torch
import cv2
import numpy as np
import pandas as pd

model = torch.hub.load('/home/isaac/Isaac/Xira/CFE_data_extraction/yolov5', 'custom', path='/home/isaac/Isaac/Xira/CFE_data_extraction/yolov5/weights/yolov5s_telmex.pt',source='local')
aspectR_telmex_horizontal = 1.7 #this gets dividing the telmex logo width and the hight and the image is horizontal
aspectR_telmex_vertical = 0.5625 #this gets dividing the telmex logo width and the hight and the image is vertical
aspectR_cfe_horizontal = 2.0 
aspectR_cfe_vertical = 0.5
# open the image
#image = cv2.imread('dataset2/IMG_20230418_115625.jpg')
image = cv2.imread('dataset2/IMG_20230418_115625.jpg')
image = cv2.resize(image,[1156,520])
image = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY) 

#pass the image to the model
detect = model(image)
info = detect.pandas().xyxy[0]  # im1 predictions
#info = pd.DataFrame(info)
df = info.loc[info['name']!='barras']
idx = df['confidence'].idxmax()
file = df.loc[idx]
point1 = (int(df['xmin'][1]),int(df['ymin'][1]))
point2 = (int(df['xmax'][1]),int(df['ymax'][1]))
print(info)
cv2.circle(image, point1, 7, (0, 255, 0), -1)
cv2.circle(image, point2, 7, (0, 255, 0), -1)
file_name = df.loc[idx,'name'] 
def orientation(df):
    asptec_ratio = (df['xmax']-df['xmin'])/(df['ymax']-df['ymin'])
    if df['name'] == 'cfe':
        if 2.3 > asptec_ratio > 1.7:
            position = 'horizontal'
        if 0.571 > asptec_ratio > 0.425:
            position = 'vertical'
    else:
        if 1.955 > asptec_ratio > 1.445:
            position='horizontal'
        if 0.646 > asptec_ratio > 0.477:
            position='vertical'
    print(df['name'],asptec_ratio)
    return position
    

print(orientation(file))
cv2.imshow('image',image)
#cv2.imshow('Detector de recibos', np.squeeze(detect.render()))
cv2.waitKey(15000)
cv2.destroyAllWindows()
