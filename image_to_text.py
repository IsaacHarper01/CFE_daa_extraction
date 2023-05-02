import torch
import cv2
import numpy as np
import pandas
from pyzbar.pyzbar import decode
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
        self.file_info = None
        self.image=None

    def detect_elements(self,image_path):
        #function who call the model yoloV5 and indentify the code bars, telmex and cfe logos
        # and returns a pandas data frame with all the information  
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
        #image = cv2.resize(image,(390,520))
        self.image = image
        data = decode(image) 
        detect = self.model(image)
        info = detect.pandas().xyxy[0]  # im1 predictions
        for bar in data:
            new_row = {'xmin': bar.polygon[0][0],'ymin': bar.polygon[0][1],'xmax': bar.polygon[2][0],'ymax': bar.polygon[2][1],'confidence':bar.quality,'class':3,'name': bar.type}
            info.loc[len(info)] = new_row
        print("Elementos detectados en la imagen")
        return info

    def identify_file(self,df):
        #this function indentify if the file is telmex or cfe document and return a tuple
        #of a pandas data frame with all the information about the object detected with the yolo network 
        #also calculate the orientation of the logo
        
        if df is not None:
            #in this line left only the information about cfe or telmex logo and
            #extract the code bars, QR and CODE128 info
            print("identificando archivo...")
            df2 = df.loc[df['name'].isin(['cfe', 'telmex'])]
            df_elements = df.loc[df['name'].isin(['barras', 'QRCODE','CODE128'])]
            #if the model detects more than one logo we keep the logo with the bigest confidence
            id_elem = df_elements['confidence'].idxmax()
            df_elements = df.iloc[id_elem]
            idx = df2['confidence'].idxmax()
            file = df2.loc[idx,'name']
            df3 = df2.loc[idx]
            #calculate the aspect ratio of the logo to know if the image is horizontal or vertical
            xdist, ydist = (df3.iloc[2]-df3.iloc[0]),(df3.iloc[3]-df3.iloc[1])#(xmax-xmin/ymax-ymin)
            
            if file == 'cfe':
                #we define a 25% error for the aspect ratio
                #cfe orientation cases
                if xdist > ydist:
                    logo = 'horizontal'
                    if df_elements is not None:
                        yqlr = int(df_elements.iloc[3])#ymax is the lower rigth point in the qr code
                        yc = int(df3.iloc[3])#ymax is the lower rigth point in the cfe logo
                        dist = yqlr-yc
                        # print(yqlr,yc)
                        # print(dist)
                        if dist > 0:
                            orientation = 'vertical_up'
                        else: 
                            orientation = 'vertical_down'
                        self.file_info = {"data":df3,"orientation":orientation,"file":file}
                        return self.file_info
                    else:
                        orientation='vertical_up'
                        self.file_info = {"data":df3,"orientation":orientation,"file":file}
                        return self.file_info
                #the logo is vertical
                if xdist < ydist:
                    logo = 'vertical'
                    if df_elements is not None:
                        xqlr = int(df_elements.iloc[2])#xmax is the lower rigth point in the qr code
                        xc = int(df3.iloc[2])#xmax is the lower rigth point in the cfe logo
                        dist = xc-xqlr
                        #print(dist)
                        if dist > 0:
                            orientation = 'horizontal_rigth'
                        else: 
                            orientation = 'horizontal_left'
                        self.file_info = {"data":df3,"orientation":orientation,"file":file}
                        return self.file_info
                    else:
                        orientation='horizontal_rigth'
                        self.file_info = {"data":df3,"orientation":orientation,"file":file}
                        return self.file_info
            else: #if it is not cfe is telmex
                if xdist>ydist:
                    logo='horizontal'
                    if df_elements is not None:
                        yqlr = int(df_elements.iloc[3])#ymax is the lower rigth point in the qr code
                        yc = int(df3.iloc[3])#ymax is the lower rigth point in the cfe logo
                        dist = yqlr-yc
                        if dist > 0:
                            orientation = 'vertical_up'
                        else: 
                            orientation = 'vertical_down'
                        self.file_info = {"data":df3,"orientation":orientation,"file":file}
                        return self.file_info
                    else:
                        orientation='vertical_up'
                        self.file_info = {"data":df3,"orientation":orientation,"file":file}
                        return self.file_info
                if xdist<ydist:
                    logo='vertical'
                    if df_elements is not None:
                        xqlr = int(df_elements.iloc[2])#xmax is the lower rigth point in the qr code
                        xc = int(df3.iloc[2])#xmax is the lower rigth point in the cfe logo
                        dist = xc-xqlr
                        if dist > 0:
                            orientation = 'horizontal_rigth'
                        else: 
                            orientation = 'horizontal_left'
                        self.file_info = {"data":df3,"orientation":orientation,"file":file}
                        return self.file_info
                    else:
                        orientation='horizontal_rigth'
                        self.file_info = {"data":df3,"orientation":orientation,"file":file}
                        return self.file_info
                          
        else:
            print('No se ha detectado ningun recibo')
    
    def cut_cfe_vertical(self,df):
        #get the cut points for the first image that contains the name and addres
        
        dfCFE = df.loc[df['name']== 'cfe']
        xclr,yclr = int(dfCFE.iloc[0,2]),int(dfCFE.iloc[0,3])#xmax, ymax
        xcul,ycul = int(dfCFE.iloc[0,0]),int(dfCFE.iloc[0,1])#xmin, ymin
        xcll,ycll = xcul, yclr
        xcur,ycur = xclr, ycul
        #width and heigth of the CFE logo
        cfe_higth = yclr-ycul
        cfe_width = xcur-xcul
        #edge points 
        xcpul,ycpul = int(xcll-(cfe_width*0.5)),int(ycll+(cfe_higth*0.5))
        xcplr,ycplr = int(xclr+(cfe_width*1.5)), int(ycpul+(1.9*cfe_higth))
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
        return result
         
    def turn_image(self,image,orientation):
        if orientation=='horizontal_rigth':
            transposed = cv2.transpose(image)
            vertical = cv2.flip(transposed,0)#0= turn the image 90°, 1 turn the image -90°
            detect = self.model(vertical)
            new_df = detect.pandas().xyxy[0]
            self.image = vertical 
            return vertical,new_df
         
        if orientation=='horizontal_left':
            transposed = cv2.transpose(image)
            vertical = cv2.flip(transposed,1)#0= turn the image 90°, 1 turn the image -90°
            detect = self.model(vertical)
            new_df = detect.pandas().xyxy[0]
            self.image = vertical 
            return vertical,new_df
        
        if orientation=='vertical_down':
            transposed = cv2.transpose(image)
            vertical1 = cv2.flip(transposed,0)#0= turn the image 90°, 1 turn the image -90°
            transposed2= cv2.transpose(vertical1,0)
            vertical2 = cv2.flip(transposed2,0)
            detect = self.model(vertical2)
            new_df = detect.pandas().xyxy[0] 
            self.image = vertical2
            return vertical2,new_df

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

    def image_to_text(self,image):
        # # Perform thresholding to create a binary image
        cv2.imshow("image",image)
        # # Perform OCR using Tesseract
        text = pytesseract.image_to_string(image,lang='spa')
        print(text)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

#path = 'dataset2/35868554_0.jpg'
path = 'dataset2/IMG_20230418_115640.jpg'
#path = 'dataset2/1693.jpg'
detector = text_detector()
df = detector.detect_elements(path)
detector.identify_file(df)
print(detector.file_info["orientation"])
if detector.file_info['file']=='telmex':
    if detector.file_info['orientation']=='vertical_up':
        image = detector.cut_telmex_vertical(df)
    else:
        img, n_df = detector.turn_image(detector.image,detector.file_info['orientation'])
        image = detector.cut_telmex_vertical(n_df)
if detector.file_info['file']=='cfe':
     if detector.file_info['orientation']=='vertical_up':
        image = detector.cut_cfe_vertical(df)
     else:
        img,n_df = detector.turn_image(detector.image,detector.file_info['orientation'])
        image = detector.cut_cfe_vertical(n_df)
detector.image_to_text(image)
