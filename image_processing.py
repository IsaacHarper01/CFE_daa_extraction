import cv2
import numpy as np
import pytesseract


image = cv2.imread('35859466_0.png',cv2.IMREAD_COLOR)

_,thress= cv2.threshold(image, 115, 255, cv2.THRESH_BINARY)

lab = cv2.cvtColor(thress, cv2.COLOR_BGR2LAB)
# Split the LAB image channels
l, a, b = cv2.split(lab)
# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l = clahe.apply(l)
# Merge the LAB channels back to form the contrast-enhanced image
lab = cv2.merge((l, a, b))
# Convert the LAB image back to BGR color space
contrast_enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
filtered_image = cv2.filter2D(contrast_enhanced_image, -1, kernel)

ksize = (5, 5)  # kernel size for GaussianBlur
sigma = 0       # standard deviation for GaussianBlur
image_blur = cv2.GaussianBlur(filtered_image, ksize, sigma)

#_, binary = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
text = pytesseract.image_to_string(thress,lang='spa')
print(text)

# Display the original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', contrast_enhanced_image)
cv2.waitKey(10000)
cv2.destroyAllWindows()

