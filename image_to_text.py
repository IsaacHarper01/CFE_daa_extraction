#librarys nedeed
from image_to_text_utils import text_detector
#from glob import glob
import cv2
import numpy as np
import pandas
#//////////////////////////////////////////////

#path = 'val2/5652.jpg'
#path = 'val2/1.png'
#path= 'val2/30565033_0.jpg'
path = 'val2/IMG_20230418_115712.jpg'

#detector = text_detector()
#df = detector.detect_elements(path)
# elements=[]
# array =[]
# for index,row  in df.iterrows():
#     elements.append(np.array([[df.iloc[index,0],df.iloc[index,1]],[df.iloc[index,2],df.iloc[index,3]]],dtype=np.float32))
# print(elements)
# center = (0,0)
#angle = detector.get_rotation_angle(detector.image)
# M = cv2.getRotationMatrix2D(center,angle,1.0)
# print(df)
# rotated_points=[]
# for i in range(len(elements)):
#     rotated_points.append(cv2.transform(elements[i].reshape(-1, 1, 2), M).squeeze())
# print(rotated_points)
# for index, row in df.iterrows():
#     df.iloc[index,0]= rotated_points[index][0][0]
#     df.iloc[index,1]= rotated_points[index][0][1]
#     df.iloc[index,2]= rotated_points[index][1][0]
#     df.iloc[index,3]= rotated_points[index][1][1]
# print(df)
#elements_points = np.array[[df.iloc[0,0]],]


#cv2.imshow('image',image)

detector = text_detector()
df = detector.detect_elements(path)
detector.identify_file(df)
print(detector.file_info['orientation'])
print("Se detectó un recibo : ",detector.file_info['file'])
if detector.file_info['file']=='telmex':
    if detector.file_info['orientation']=='vertical_up':
        angle = detector.get_rotation_angle(detector.image)
        print('angulo de la imagen: ',angle)
        if angle<2 and angle >-2:
            crop_image = detector.cut_telmex_vertical(detector.image,df)
        else:
            rot_image, n_df = detector.correct_image_angle(detector.image,angle,df)
            crop_image = detector.cut_telmex_vertical(rot_image,n_df)
    else:
        img, n_df = detector.turn_image(detector.image,detector.file_info['orientation'])
        angle = detector.get_rotation_angle(img)
        if angle <2 and angle >-2:
            crop_image = detector.cut_telmex_vertical(img,n_df)
        else:
            rot_image, n_df = detector.correct_image_angle(detector.image,angle,n_df)
            crop_image = detector.cut_telmex_vertical(img,n_df)
if detector.file_info['file']=='cfe':
    if detector.file_info['orientation']=='vertical_up':
        angle = detector.get_rotation_angle(detector.image)
        print(angle)
        if angle <2 and angle >-2:
            crop_image = detector.cut_cfe_vertical(detector.image,df)
        else:
            rot_image, n_df = detector.correct_image_angle(detector.image,angle,df)
            crop_image = detector.cut_cfe_vertical(rot_image,n_df)
    else:
        img,n_df = detector.turn_image(detector.image,detector.file_info['orientation'])
        angle = detector.get_rotation_angle(img)
        if angle <2 and angle >-2:
            crop_image = detector.cut_cfe_vertical(img,n_df)
        else:
            rot_image, n_df = detector.correct_image_angle(img,angle,n_df)
            crop_image = detector.cut_cfe_vertical(rot_image,n_df)

#print(detector.image_to_text(image))
cv2.imshow("rotated image", crop_image)
cv2.waitKey(0)
cv2.destroyAllWindows()








