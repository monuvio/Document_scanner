import cv2
import imutils
from PIL import Image
import pytesseract
from utils import four_point_transform

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\admin\AppData\Local\Tesseract-OCR\tesseract'

image = cv2.imread('instruction.jpg')
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)


# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    
    
# loop over the contours
max_contour = cnts[0]
for c in cnts:
    peri = cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)
    screenCnt = approx

    if len(approx) == 4:
        screenCnt = approx
        break
       
warped = four_point_transform(orig, screenCnt.reshape(max(list(screenCnt.shape)), 2) * ratio)
 
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


ret3,th3 = cv2.threshold(warped,140,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imwrite('for_tess.jpg', th3)

image = Image.open('for_tess.jpg')
text = pytesseract.image_to_string(th3, lang="rus")

with open('text.txt', 'w') as text_file:
    text_file.write(text)
    
