import cv2
import numpy as np

# Load the image
path = 'dataset2/images.jpg'
image = cv2.imread(path,0)

# Define the center point of rotation
rows, columns = image.shape
point = np.array([[190],[60]])
print(point[0])
center = (columns // 2, rows // 2)

# Define the angle of rotation in degrees
angle = 90

# Calculate the rotation matrix using cv2.getRotationMatrix2D()
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
rot_point = cv2.warpAffine(point,rotation_matrix,(3,3))
print(rot_point)
# Apply the rotation to the image using cv2.warpAffine()
rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
cv2.circle(image,(190,60),5,(0,255,0),5)
# Display the original and rotated images side by side
cv2.imshow("Original Image", image)
cv2.imshow("Rotated Image", rotated_image)
cv2.waitKey(10000)
cv2.destroyAllWindows()
