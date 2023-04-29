import cv2
import pytesseract

image = cv2.imread('35859466_0.png')

# Mostrar la imagen original y la imagen enfocada

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Perform thresholding to create a binary image
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("image",gray)

# # Perform OCR using Tesseract
text = pytesseract.image_to_string(gray,lang='spa')
print(text)
cv2.waitKey(15000)
cv2.destroyAllWindows()
# # Print the extracted text

