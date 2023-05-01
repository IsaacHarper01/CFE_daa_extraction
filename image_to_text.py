import torch
import cv2
import numpy as np
import pandas
from pyzbar.pyzbar import decode
from math import tan
import pytesseract
 

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
        self.image=None

    def detect_elements(self,image_path):
        #function who call the model yoloV5 and indentify the code bars, telmex and cfe logos
        # and returns a pandas data frame with all the information  
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
        #image = cv2.resize(image,(1640,738))
        self.image = image
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
        #get the cut points for the first image that contains the name and addres
        
        dfCFE = df.loc[df['name']== 'cfe']
        xclr,yclr = int(dfCFE.iloc[0,2]),int(dfCFE.iloc[0,3])
        xcul,ycul = int(dfCFE.iloc[0,0]),int(dfCFE.iloc[0,1])
        xcll,ycll = xcul, yclr
        xcur,ycur = xclr, ycul
        #width and heigth of the CFE logo
        cfe_higth = yclr-ycul
        cfe_width = xcur-xcul
        #edge points 
        xcpul,ycpul = int(xcll-(cfe_width*0.5)),int(ycll+(cfe_higth*0.5))
        xcplr,ycplr = int(xclr+cfe_width), int(ycpul+(1.7*cfe_higth))
        xcpur,ycpur = xcplr,ycpul
        xcpll,ycpll = xcpul, ycplr
        
        #crop the image using the points obtained before
        points = np.array([(xcpul, ycpul), (xcpur, ycpur), (xcplr, ycplr), (xcpll, ycpll)], np.int32)
        mask = np.zeros_like(self.image)
        cv2.fillPoly(mask, [points], (255, 255, 255))
        result = cv2.bitwise_and(self.image, mask)
        x, y, w, h = cv2.boundingRect(points)
        #croped image
        result = result[y:y+h, x:x+w]
        #result = cv2.resize(result, (w, h))
        # cv2.imshow("image",result)
        # cv2.waitKey(10000)
        # cv2.destroyAllWindows()
        return result

        if df.isin(['QRCODE']).any().any():
            dfQR = df.loc[df.isin(['QRCODE']).any(axis=1)]
            #if qr in the tallest position
            xqlr, yqlr = int(dfQR.iloc[0,2]),int(dfQR.iloc[0,3])#xmax,ymax
            xqul, yqul = int(dfQR.iloc[0,0]),int(dfQR.iloc[0,1])#xmin,ymin
            xqll, yqll = xqul, yqlr
            xqur, yqur = xqlr , yqul
            QR_hight = yqlr-yqul
            print(QR_hight/cfe_higth)
        
    
    def find_point(self,alpha,beta,gama,point1,point2):
        #xc=point2[0],yc=point2[1],xq=point1[0],yq=point1[1]
        ma =  (point1[1]-point2[1])/(point1[0]-point2[0])
        slope = lambda angle,m1: (tan(angle)+m1)/(1-(tan(angle)*m1))
        mc = slope(beta,ma)
        mb = slope(180-gama,mc)
        print(ma,mb,mc)
        # xi = int(((-mb*point2[0])+point2[1]-point1[1]+(mc*point1[0]))/(mc-mb))
        # yi = int((mb*xi)+(-mb*point2[0])+point2[1])
        # #print(xi,yi)
        # return xi,-yi

    def turn_image(self,image):
        transposed = cv2.transpose(image)
        vertical = cv2.flip(transposed,0)#0= turn the image 90°, 1 turn the image -90°
        cv2.imshow('image',vertical)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

    def cut_telmex_vertical(self,df):
        if df.isin(['telmex']).any().any():
            dftel = df.loc[df['name']== 'telmex']
            if df.isin(['barras']).any().any():
                #detected points in the code bar
                dfbars = df.loc[df.isin(['barras']).any(axis=1)]
                xlr,ylr = int(dfbars.iloc[0,2]), int(dfbars.iloc[0,3])#xmax,ymax
                cpxll,cpyll = int(dfbars.iloc[0,0]), int(dfbars.iloc[0,1])#xmin,ymin:superior left point in the code bar deteced
                cpxlr,cpylr = xlr,cpyll #superior right point in the bar code detected
                
                #detected points in the telmex logo
                xtlr,ytlr = int(dftel.iloc[0,2]), int(dftel.iloc[0,3])#xmax,ymax
                xtul,ytul = int(dftel.iloc[0,0]), int(dftel.iloc[0,1])#xmin,ymin
                cpxul,cpyul = xtul,ytlr #cut point uper left
                cpxur,cpyur = cpxlr, ytlr #cut point uper right
                disty = int(1.5*(ytlr-ytul))
            
                #crop the image using the points obtained before
                points = np.array([(cpxul, cpyul+disty), (cpxur, cpyur+disty), (cpxlr, cpylr), (cpxll, cpyll)], np.int32)
                mask = np.zeros_like(self.image)
                cv2.fillPoly(mask, [points], (255, 255, 255))
                result = cv2.bitwise_and(self.image, mask)
                x, y, w, h = cv2.boundingRect(points)
                #croped image
                result = result[y:y+h, x:x+w]
                result = cv2.resize(result, (w, h))
                return result
            
        if df.isin(['barras']).any().any():
            dfbars = df.loc[df.isin(['barras']).any(axis=1)]
            xlr,ylr = int(dfbars.iloc[0,2]), int(dfbars.iloc[0,3])
            cpxll,cpyll = int(dfbars.iloc[0,0]), int(dfbars.iloc[0,1])#superior left point in the code bar deteced
            cpxlr,cpylr = xlr,cpyll #superior right point in the bar code detected
            disty = int(2*(ylr-cpyll))
            cpxul,cpyul = cpxll, cpyll-disty
            cpxur, cpyur = cpxlr, cpylr-disty
            #crop the image using the points obtained before

            points = np.array([(cpxul, cpyul), (cpxur, cpyur), (cpxlr, cpylr), (cpxll, cpyll)], np.int32)
            mask = np.zeros_like(self.image)
            cv2.fillPoly(mask, [points], (255, 255, 255))
            result = cv2.bitwise_and(self.image, mask)
            x, y, w, h = cv2.boundingRect(points)
            #croped image
            result = result[y:y+h, x:x+w]
            result = cv2.resize(result, (w, h))
            return result

    def cut_telmex_horizontal(self,df):
         if df.isin(['telmex']).any().any():
            dftel = df.loc[df['name']== 'telmex']
            if df.isin(['barras']).any().any():
                #detected points in the code bar
                dfbars = df.loc[df.isin(['barras']).any(axis=1)]
                cpxll,cpyll = int(dfbars.iloc[0,2]), int(dfbars.iloc[0,3])#xmax,ymax
                xul,yul = int(dfbars.iloc[0,0]), int(dfbars.iloc[0,1])#xmin,ymin:superior left point in the code bar deteced
                cpxul,cpyul = cpxll,yul #superior left point in the bar code detected
                #detected points in the telmex logo
                xtur,ytur = int(dftel.iloc[0,2]), int(dftel.iloc[0,3])#xmax,ymax
                cpxur,cpyur = int(dftel.iloc[0,0]), int(dftel.iloc[0,1])#xmin,ymin
                cpxlr, cpylr = cpxur, cpyll
                distx = cpxul-xul
            
                #cv2.circle(self.image,(cpxll,cpyll),7,(0,255,0),5)
                # cv2.circle(self.image,(cpxul,cpyul),7,(0,255,0),5)
                # cv2.circle(self.image,(cpxur,cpyur),7,(0,255,0),5)
                # cv2.circle(self.image,(cpxlr,cpylr),7,(0,255,0),5)

                #crop the image using the points obtained before
                ajust = int(self.image.shape[0]*0.02)
                points = np.array([(cpxul-ajust, cpyul), (cpxur-distx, cpyur), (cpxlr-distx, cpylr), (cpxll-ajust, cpyll)], np.int32)
                mask = np.zeros_like(self.image)
                cv2.fillPoly(mask, [points], (255, 255, 255))
                result = cv2.bitwise_and(self.image, mask)
                x, y, w, h = cv2.boundingRect(points)
                #croped image
                result = result[y:y+h, x:x+w]
                #result = cv2.resize(result, (w, h))
                return result

    def image_to_text(self,image):
        # # Perform thresholding to create a binary image
        cv2.imshow("image",image)
        # # Perform OCR using Tesseract
        text = pytesseract.image_to_string(image,lang='spa')
        print(text)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

#path = 'dataset2/IMG_20230418_115707.jpg'
#path = 'dataset2/IMG_20230418_115625.jpg'
path = 'dataset2/30564823_0.jpg'
detector = text_detector()
#img = detector.cut_telmex_horizontal(detector.detect_elements(path))
#detector.image_to_text(img)
img = detector.cut_cfe_vertical(detector.detect_elements(path))
detector.image_to_text(img)
# detector.detect_elements(path)
# detector.turn_image(detector.image)
