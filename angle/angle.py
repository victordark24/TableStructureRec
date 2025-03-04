import cv2


from wired_table_rec.utils import ImageOrientationCorrector

img_path = f'qingxie.jpg'
img_orientation_corrector = ImageOrientationCorrector()
img = cv2.imread(img_path)
img = img_orientation_corrector(img)
cv2.imwrite(f'img_rotated.jpg', img)