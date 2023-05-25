import torch
import cv2
import numpy as np
import pandas
from pyzbar.pyzbar import decode
from math import atan
import pytesseract
from glob import glob
import string
import re


class text_detector():
    def __init__(self):
        self.hight_cuted_telmex = 0.5
        self.width_cuted_telmex = 2
        self.hight_cuted_cfe = 4
        self.width_cuted_cfe = 2
        #load the yoloV5 model to detect key points in the images
        self.model = torch.hub.load('/home/isaac/Isaac/Xira/CFE_data_extraction/yolov5', 'custom', path='/home/isaac/Isaac/Xira/CFE_data_extraction/yolov5/weights/yolov5s_telmex.pt',source='local')
        #python dictionary with information about the cfe or telmex logo detected
        self.file_info = None
        #the main image
        self.image=None
        self.image_color=None
        self.path = None
        #load spacy model for natural language processing
        #self.model_ner = spacy.load('./spacy_data/output_spacy/model-best/') 

    def detect_elements(self,image_path):
        self.path = image_path
        #function who call the model yoloV5 and indentify the code bars, telmex and cfe logos
        # and returns a pandas data frame with all the information  
        image = cv2.imread(image_path)
        #image = cv2.resize(image,(int(image.shape[1]*0.25),int(image.shape[0]*0.25)))
        image = cv2.fastNlMeansDenoising(image,None,10,9,21)
        self.image_color = image
        image = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
        #image = cv2.resize(image,(390,520))
        self.image = image
        data = decode(image) 
        detect = self.model(image)
        info = detect.pandas().xyxy[0]  # im1 predictions
        #print(info)
        if data is not None:
            for bar in data:
                if bar.type == 'QRCODE':
                    new_row = {'xmin': bar.polygon[0][0],'ymin': bar.polygon[0][1],'xmax': bar.polygon[2][0],'ymax': bar.polygon[2][1],'confidence':bar.quality,'class':3,'name': bar.type}
                    info.loc[len(info)] = new_row
                else:
                    continue
            print("Se detectaron elementos en la imagen")
             
        #     xmin, ymin=dfmax.iloc[0],dfmax.iloc[1]
        #     xmax, ymax=dfmax.iloc[2],dfmax.iloc[3]
        #     print(xmax,xmin,ymax,ymin)
        #     if len(dfbarras)>1:
        #         for i in range(1,len(dfbarras)):
        #             if xmin-dfbarras.iloc[i,0]<100:

        return info

    def identify_file(self,df):
        
        #this function indentify if the file is telmex or cfe document and return a tuple
        #of a pandas data frame with all the information about the object detected with the yolo network 
        #also calculate the orientation of the logo
        if not df.empty:
            print("identificando archivo...")

            #in this line left only the information about cfe or telmex logo and
            #extract the code bars, QR and CODE128 info
            df2 = df.loc[df['name'].isin(['cfe', 'telmex'])]
           
            if not df2.empty:
               
                idx = df2['confidence'].idxmax()
                file = df2.loc[idx,'name']
                df3 = df2.loc[idx]
               
            else:
                ('No se pudo detectar ningun recibo')
            df_elements = df.loc[df['name'].isin(['barras', 'QRCODE','CODE128'])]
            #if the model detects more than one logo we keep the logo with the bigest confidence
            if df_elements.empty:
                print('no se puede determinar la orientacion del archivo')
                print('se asignara la orientacion mas probable')
            else:    
                id_elem = df_elements['confidence'].idxmax()
                df_elements = df.iloc[id_elem]
            
            #calculate the width and heigth of the logo to compare them and decide
            #if the image is horizontal or vertical
            xdist, ydist = (df3.iloc[2]-df3.iloc[0]),(df3.iloc[3]-df3.iloc[1])#(xmax-xmin/ymax-ymin)
            
            if file == 'cfe':
                #in the case that xdist is biger than ydist it means the logo is horizontal 
                #and the image is vertical
                if xdist > ydist:
                    logo = 'horizontal'
                    if not df_elements.empty:
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
                #the logo is vertical
                if xdist < ydist:
                    logo = 'vertical'
                    if not df_elements.empty:
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
            else: #if it is not cfe is telmex
                if xdist>ydist:
                    logo='horizontal'
                    if not df_elements.empty:
                        yqlr = int(df_elements.iloc[3])#ymax is the lower rigth point in the bar code
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
                    if not df_elements.empty:
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
    
    def turn_image(self,image,orientation):

        if orientation=='horizontal_rigth':
            transposed = cv2.transpose(image)
            vertical = cv2.flip(transposed,0)#0= turn the image 90°, 1 turn the image -90°
            detect = self.model(vertical)
            new_df = detect.pandas().xyxy[0]
            self.image_color = vertical 
            return vertical,new_df
         
        if orientation=='horizontal_left':
            transposed = cv2.transpose(image)
            vertical = cv2.flip(transposed,1)#0= turn the image 90°, 1 turn the image -90°
            detect = self.model(vertical)
            new_df = detect.pandas().xyxy[0]
            self.image_color = vertical 
            return vertical,new_df
        
        if orientation=='vertical_down':
            transposed = cv2.transpose(image)
            vertical1 = cv2.flip(transposed,0)#0= turn the image 90°, 1 turn the image -90°
            transposed2= cv2.transpose(vertical1,0)
            vertical2 = cv2.flip(transposed2,0)
            detect = self.model(vertical2)
            new_df = detect.pandas().xyxy[0] 
            self.image_color = vertical2
            return vertical2,new_df
    
    def get_rotation_angle(self,image):
        #newImage = cv2.resize(image,(int(image.shape[1]*0.3),int(image.shape[0]*0.3)))
        blur = cv2.GaussianBlur(image, (9, 9), 0)
        edges = cv2.Canny(blur,150,150)
        sensibility_line_detection = 150
        lines = cv2.HoughLines(edges, 1, np.pi/180, sensibility_line_detection)
        slopes=[]

        for i in range(4):
            if lines is not None:
                break
            else:
                sensibility_line_detection = sensibility_line_detection-30
                print(sensibility_line_detection)
                lines = cv2.HoughLines(edges, 1, np.pi/180, sensibility_line_detection)

        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            slope = (y2-y1)/(x2-x1) if (x2-x1)!=0 else 0
            if slope < 1 and slope >-1:
                slopes.append(slope)
            cv2.line(self.image_color, (x1, y1), (x2, y2), (0, 0, 255), 4)
        average = sum(slopes)/len(slopes)
        rad_angle = atan(average)
        rotation_angle = (rad_angle*360)/(2*np.pi)
        cv2.imshow("lines detected", self.image_color)
        return rotation_angle
        
    def correct_image_angle(self,image,rotation_angle,df):
        elements=[]
        for index,row  in df.iterrows():
            elements.append(np.array([[df.iloc[index,0],df.iloc[index,1]],[df.iloc[index,2],df.iloc[index,3]]],dtype=np.float32))
       
        center = (int(self.file_info['data']['xmin']),int(self.file_info['data']['ymin']))
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        image_rotated = cv2.warpAffine(image, M, center, flags=cv2.INTER_CUBIC)
        rotated_points=[]
        for i in range(len(elements)):
            rotated_points.append(cv2.transform(elements[i].reshape(-1, 1, 2), M).squeeze())
        for index, row in df.iterrows():
            df.iloc[index,0]= rotated_points[index][0][0]
            df.iloc[index,1]= rotated_points[index][0][1]
            df.iloc[index,2]= rotated_points[index][1][0]
            df.iloc[index,3]= rotated_points[index][1][1]

        return image_rotated, df

    def cut_cfe_vertical(self,image,df):
        #get the cut points for the image that contains the information nedeed
        
        dfCFE = df.loc[df['name']== 'cfe']
        #points of the CFE logo
        xclr,yclr = int(dfCFE.iloc[0,2]),int(dfCFE.iloc[0,3])#xmax, ymax
        xcul,ycul = int(dfCFE.iloc[0,0]),int(dfCFE.iloc[0,1])#xmin, ymin

        if  xcul <0:
            xcul = 0

        xcll,ycll = xcul, yclr
        xcur,ycur = xclr, ycul
        #width and heigth of the CFE logo
        cfe_higth = yclr-ycul
        cfe_width = xcur-xcul
        #edge points from image that contains the info
        xcpul,ycpul = int(xcll-(cfe_width*0.5)),int(ycll+(cfe_higth*0.5))
        xcplr,ycplr = int(xclr+(cfe_width*self.width_cuted_cfe)), int(ycpul+(self.hight_cuted_cfe*cfe_higth))
        
        if xcpul < 0:
            xcpul = 0

        xcpur,ycpur = xcplr,ycpul
        xcpll,ycpll = xcpul, ycplr

        #crop the image using the points obtained before
        points = np.array([(xcpul, ycpul), (xcpur, ycpur), (xcplr, ycplr), (xcpll, ycpll)], np.int32)
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [points], (255, 255, 255))
        result1 = cv2.bitwise_and(image, mask)
        x, y, w, h = cv2.boundingRect(points)
        #croped image
        result1 = result1[y:y+h, x:x+w]

        return result1

    def cut_telmex_vertical(self,image,df):
        if df.isin(['telmex']).any().any():
            dftel = df.loc[df['name']== 'telmex']
            if df.isin(['barras']).any().any():
                #detected points in the code bar
                dfbars = df.loc[df.isin(['barras']).any(axis=1)]
                xlr,ylr = int(dfbars.iloc[0,2]), int(dfbars.iloc[0,3])#xmax,ymax
                cpxll,cpyll = int(dfbars.iloc[0,0]), int(dfbars.iloc[0,1])#xmin,ymin:superior left point in the code bar deteced
                cpxlr,cpylr = xlr,cpyll #superior right point in the bar code detected
                
                #distx is a pixels paramater to do a little adjust in the image crop
                distx = 15
                
                #detected points in the telmex logo
                xtlr,ytlr = int(dftel.iloc[0,2]), int(dftel.iloc[0,3])#xmax,ymax
                xtul,ytul = int(dftel.iloc[0,0])-distx, int(dftel.iloc[0,1])#xmin,ymin

                if xtul <0:
                    xtul = 0

                cpxul,cpyul = xtul,ytlr #cut point uper left
                cpxur,cpyur = cpxlr, ytlr #cut point uper right
                logo_heigth = ytlr-ytul
                logo_width = xtlr-xtul
                disty = int(self.hight_cuted_telmex*logo_heigth) # this modifies the top line of the image croped

                #crop the image using the points obtained before

                                    #uper left point      #uper rigth point     #lower rigth point    #lower left point
                points = np.array([(cpxul, cpyul+disty), (cpxlr, cpyur+disty), (cpxlr, cpylr+distx), (cpxul, cpylr+distx)], np.int32)
                mask = np.zeros_like(image)
                cv2.fillPoly(mask, [points], (255, 255, 255))
                result1 = cv2.bitwise_and(image, mask)
                x, y, w, h = cv2.boundingRect(points)
                #croped image
                result1 = result1[y:y+h, x:x+w]
                #result1 = cv2.resize(result1, (w, h))

                return result1
            
        if df.isin(['barras']).any().any():
            dfbars = df.loc[df.isin(['barras']).any(axis=1)]
            xlr,ylr = int(dfbars.iloc[0,2]), int(dfbars.iloc[0,3])
            cpxll,cpyll = int(dfbars.iloc[0,0]), int(dfbars.iloc[0,1])#superior left point in the code bar deteced
            
            if cpxll<0:
                cpxll=0

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
        
    def clean_bar_codes(self,df):
            print(self.file_info['data']['xmin'])
            if df.isin(['barras']).any().any():
                dfbarras = df.loc[df['name']=='barras']
                if len(dfbarras)>1:
                    idmax = dfbarras['confidence'].idxmax()
                    dfmax = dfbarras.loc[idmax]
                    heigth_logo = self.file_info['data']['ymax']-self.file_info['data']['ymin']
                    print(dfmax['ymax'])
                    #if dfmax['ymax']>self.file_info['data']['ymax']+heigth_logo*10:
            #  for code in dfbarras:
            #      if 
        
    def image_to_text(self,image):
        # # Perform OCR using Tesseract
    
        text = pytesseract.image_to_string(image, lang='spa')
        text_list = list(text.split('\n'))
        for line in text_list:
                if line == '' or line == '\x0c':
                    text_list.remove(line)
        datos={}
        # #identify the date, name and addres in the cfe file
        if self.file_info['file']=='cfe':
            eliminate_words = ["RMU","XAXX","CFE","SERVICIO","CORTE","PARTIR"]
            text_clean = [line for line in text_list if all(word not in line for word in eliminate_words)]
            date_exist = any("PAGO" in line for line in text_clean)
            text_clean2=[]
            if date_exist:
                for line in text_clean:
                    if 'PAGO' in line:
                        text_clean2.append(line)
                        break
                    else:
                        text_clean2.append(line)
            for line in text_clean2:
                if len(line)<2:
                    text_clean2.remove(line)
            words_last_line = list(text_clean2[len(text_clean2)-1].split())
            date=[]
            number = False
            for word in words_last_line:
                if word[0].isdigit():
                    number = True
                if number:
                    date.append(word)
            if len(date)>=3:
                text_clean2[len(text_clean2)-1] = "-".join(date[:3])
            else:
                text_clean2[len(text_clean2)-1] = "-".join(date)
            nombre=text_clean2.pop(0)
            fecha=text_clean2.pop(len(text_clean2)-1)
            direccion= " ".join(text_clean2)
            datos = {'nombre:':nombre,'direccion':direccion,'fecha de pago':fecha,'archivo':self.file_info['file']} 
            
        # #indentify the name, addres and date in telmex file
        else: 
            #confirm if exist the word KT6 or DV to identify the line where it is the date
            date_exist = any(("KT6" in line or "DV" in line) for line in text_list)
            rfc_kt6_dv_exist = False
            text_clean1=[]
            if date_exist:
                #eliminate all the lines before the date
                for line in text_list:
                    if "KT6" in line or "DV" in line:
                        rfc_kt6_dv_exist = True
                    if rfc_kt6_dv_exist:
                        text_clean1.append(line)
                #take the first line that contain the date a delete all the not necesary information
                words_line1 = text_clean1[0].split()
                words_line1 = [word.replace("O","0") for word in words_line1]
                date = [word for word in words_line1 if not word[0].isalpha()]
                for word in date:
                    if len(word)<2:
                        date.remove(word)
                text_clean1[0] = date[0]
            else:
                text_clean1=text_list
            #eliminate all dots in the text
            text_clean1 = [line.replace(".","") for line in text_clean1]
            CR_exist = any(("CR" in line or "CP" in line) for line in text_clean1)
            text_clean2=[]
            if CR_exist:
                for line in text_clean1:
                    if 'CR' in line:
                        text_clean2.append(line)
                        break
                    else:
                        text_clean2.append(line)
            else:
                text_clean2 = text_clean1
            for line in text_clean2:
                if len(line)<2:
                    text_clean2.remove(line)
            fecha = text_clean2.pop(0)
            nombre = text_clean2.pop(0)
            direccion = " ".join(text_clean2)
            datos={'nombre :':nombre,'direccion :':direccion,'fecha de emision:':fecha}
        
        return datos
        
        
