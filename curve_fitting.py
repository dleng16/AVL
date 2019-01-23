import numpy as np
import cv2
from matplotlib import pyplot as plt



img = cv2.imread('curve.png',0)
ret, img = cv2.threshold(img, 127, 255, 0)
print(img)

print(len(img))
print(len(img[0]))
x = []
y = []
counter = 0

#black image
canvas = np.zeros((len(img),len(img[0]),3), np.uint8)

for i in range(len(img)):
    for j in range(len(img[0])):
        if(img[i,j] == 0):
            x.append(i)
            y.append(j)
            img[i,j] = 255
        else:
            img[i,j] = 0

new_img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# print(contours[0])
# print(hierarchy)
# epsilon = 0.001*cv2.arcLength(contours[0],True)
# approx = cv2.approxPolyDP(contours[0],epsilon,False)
# print(epsilon)
# print(approx)

new_img = cv2.drawContours(canvas, contours, -1, (0, 255, 0), 3)
canvas[600,600] = (0, 255, 0)
canvas[601,600] = (0, 255, 0)
canvas[602,600] = (0, 255, 0)
canvas[603,600] = (0, 255, 0)

# for i in range(len(img)):
#     for j in range(len(img[0])):
#         if(img[i,j] in contours[0]):
#             y.append(i)
#             x.append(j)
#             img[i,j] = 255
#         else:
#             img[i,j] = 0

plt.plot(x, y)


#@draws contour points@
# for point in contours[0]:
#     print(point)
#     print(point[0])
#     print(point[0][0])
#     #print(canvas(point[0]))
#     canvas[point[0][1],point[0][0]] = (0,255,0)

coeffsx = [1, 2, 3, 4, 5]
coeffsy = [1, 2, 3, 4, 5]









p = np.polyfit(x, y,3)

print(p)

j = np.poly1d(p)

print(j)

def PolyCoefficients(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    y = 0
    for i in range(o):
        y += coeffs[i]*x**(o-1-i)
    return y

x = np.linspace(0, 599, 600)
plt.plot(x, PolyCoefficients(x, p))
plt.show()


# y = 0
# o = len(p)
# counter = 0
# for i in range(len(img)):
#     y = 0
#     for j in range(o):
#         y += p[j]*i**j
#     print(y)
#     if(y > len(img[0])):
#         y = 100
#         counter = counter + 1
#     if(y < 1):
#         y = 100
#         counter = counter + 1
#     img[i,int(y)] = 180

# print(counter)
# print(y)


	

img[100,600] = 200
img[103,600] = 200
img[102,600] = 200
img[101,600] = 200


cv2.imshow('frame2', img)
cv2.imshow('frame2', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()



		
		



