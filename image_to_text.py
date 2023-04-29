import torch
import cv2
import numpy as np
import pandas
from pyzbar.pyzbar import decode


class text_detector():
    def __init__(self):
        self.aspectR_telmex_horizontal = 2.125 #this gets dividing the telmex logo width and the hight and the image is horizontal
        self.aspectR_telmex_vertical = 0.4705 #this gets dividing the telmex logo width and the hight and the image is vertical
        self.aspectR_cfe_horizontal = 2.0 
        self.aspectR_cfe_vertical = 0.5
        #load the yoloV5 model to detect key points in the image
        self.model = torch.hub.load('/home/isaac/Isaac/Xira/CFE_data_extraction/yolov5', 'custom', path='/home/isaac/Isaac/Xira/CFE_data_extraction/yolov5/weights/yolov5s_telmex.pt',source='local')
        #python dictionary with information about the cfe or telmex logo detected
        self.logo = None

    def detect_elements(self,image_path):
        #function who call the model yoloV5 and indentify the code bars, telmex and cfe logos
        # and returns a pandas data frame with all the information  
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
        #image = cv2.resize(image,(1150,520))
        data = decode(image) 
        detect = self.model(image)
        info = detect.pandas().xyxy[0]  # im1 predictions
        for bar in data:
            new_row = {'xmin': bar.polygon[0][0],'ymin': bar.polygon[0][1],'xmax': bar.polygon[2][0],'ymax': bar.polygon[2][1],'confidence':bar.quality,'class':3,'name': bar.type}
            info.loc[len(info)] = new_row
        return info

    def identify_file(self,df):
        #this function indentify if the file is telmex or cfe document and return a tuple
        #of a pandas data frame with all the information about the object detected with the yolo network 
        #also calculate the orientation of the logo

        if df is not None:
            #in this line left only the information about cfe or telmex logo and
            #extract the code bars info
            df2 = df.loc[df['name']!='barras']
            #if the model detects more than one logo we keep the logo with the bigest confidence
            idx = df2['confidence'].idxmax()
            file = df2.loc[idx,'name']
            df3 = df2.loc[idx]
            #calculate the aspect ratio of the logo to know if the image is horizontal or vertical
            asptec_ratio = (df3['xmax']-df3['xmin'])/(df3['ymax']-df3['ymin'])
            if file == 'cfe':
                #we define a 15% error for the aspect ratio
                if 2.3 > asptec_ratio > 1.7:
                    orientation = 'horizontal'
                if 0.571 > asptec_ratio > 0.425:
                    orientation = 'vertical'
                else:
                    return "detection not confidence"
            else:
                if 1.955 > asptec_ratio > 1.445:
                    orientation='horizontal'
                if 0.646 > asptec_ratio > 0.477:
                    orientation='vertical'
                else:
                    return "detection not confidence"
            self.logo = {"data":df3,"orientation":orientation,"file":file}
            return self.logo
        else:
            print('No se ha detectado ningun recibo')
    
    def cut_cfe_vertical(self,df):
        dfCFE = df.loc[df['name']== 'cfe']
        xc,yc = int(dfCFE['xmax'][0]),int(dfCFE['ymax'][0])
        if df.isin(['QRCODE']).any().any():
            dfQR = df.loc[df.isin(['QRCODE']).any(axis=1)]
            #print(dfQR)
            if len(dfQR)==1:
                #print('1 QR code')
                xq, yq = int(dfQR['xmax'][1]),int(dfQR['ymax'][1])
        print(xc,yc)
        print(xq,yq)
            
 




path = 'dataset2/IMG_20230418_115707.jpg'
#path = 'dataset2/IMG_20230418_115625.jpg'
detector = text_detector()
detector.cut_cfe_vertical(detector.detect_elements(path))
#print(detector.logo["file"])