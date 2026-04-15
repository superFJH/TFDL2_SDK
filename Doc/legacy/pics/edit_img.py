import cv2

img = cv2.imread("matmul.png", cv2.IMREAD_UNCHANGED)
print(img.shape)
height = img.shape[0]
width = img.shape[1]
img = cv2.resize(img, (int(width / 1.5), int(height / 1.5)), interpolation=cv2.INTER_LINEAR)
print(img.shape)
cv2.imwrite("matmul.png", img)