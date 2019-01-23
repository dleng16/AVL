import numpy as np
import cv2

cap = cv2.VideoCapture(0)

#trackbar_name = 'Alpha x %d' % alpha_slider_max
#cv.createTrackbar('cool', 'wow' , 0, alpha_slider_max, on_trackbar)
def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

#thick red lines 
def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    """workflow:
    1) examine each individual line returned by hough & determine if it's in left or right lane by its slope
    because we are working "upside down" with the array, the left lane will have a negative slope and right positive
    2) track extrema
    3) compute averages
    4) solve for b intercept 
    5) use extrema to solve for points
    6) smooth frames and cache
    """
    global cache
    global first_frame
    y_global_min = img.shape[0] #min will be the "highest" y value, or point down the road away from car
    y_max = img.shape[0]
    l_slope, r_slope = [],[]
    l_lane,r_lane = [],[]
    det_slope = 0.4
    alpha =.2 
    #i got this alpha value off of the forums for the weighting between frames.
    #i understand what it does, but i dont understand where it comes from
    #much like some of the parameters in the hough function
    
    for line in lines:
        #1
        for x1,y1,x2,y2 in line:
            slope = get_slope(x1,y1,x2,y2)
            if slope > det_slope:
                r_slope.append(slope)
                r_lane.append(line)
            elif slope < -det_slope:
                l_slope.append(slope)
                l_lane.append(line)
        #2
        y_global_min = min(y1,y2,y_global_min)
    
    # to prevent errors in challenge video from dividing by zero
    if((len(l_lane) == 0) or (len(r_lane) == 0)):
        print ('no lane detected')
        return 1
        
    #3
    l_slope_mean = np.mean(l_slope,axis =0)
    r_slope_mean = np.mean(r_slope,axis =0)
    l_mean = np.mean(np.array(l_lane),axis=0)
    r_mean = np.mean(np.array(r_lane),axis=0)
    
    if ((r_slope_mean == 0) or (l_slope_mean == 0 )):
        print('dividing by zero')
        return 1
   
    
    #4, y=mx+b -> b = y -mx
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])
    
    #5, using y-extrema (#2), b intercept (#4), and slope (#3) solve for x using y=mx+b
    # x = (y-b)/m
    # these 4 points are our two lines that we will pass to the draw function
    l_x1 = int((y_global_min - l_b)/l_slope_mean) 
    l_x2 = int((y_max - l_b)/l_slope_mean)   
    r_x1 = int((y_global_min - r_b)/r_slope_mean)
    r_x2 = int((y_max - r_b)/r_slope_mean)
    
    #6
    if l_x1 > r_x1:
        l_x1 = int((l_x1+r_x1)/2)
        r_x1 = l_x1
        l_y1 = int((l_slope_mean * l_x1 ) + l_b)
        r_y1 = int((r_slope_mean * r_x1 ) + r_b)
        l_y2 = int((l_slope_mean * l_x2 ) + l_b)
        r_y2 = int((r_slope_mean * r_x2 ) + r_b)
    else:
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max
      
    current_frame = np.array([l_x1,l_y1,l_x2,l_y2,r_x1,r_y1,r_x2,r_y2],dtype ="float32")
    
    if first_frame == 1:
        next_frame = current_frame        
        first_frame = 0        
    else :
        prev_frame = cache
        next_frame = (1-alpha)*prev_frame+alpha*current_frame
             
    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]),int(next_frame[3])), color, thickness)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]),int(next_frame[7])), color, thickness)
    
    cache = next_frame

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray,9,75,75)
    edges = cv2.Canny(gray,100,200)

    # imshape = blur.shape
    # print(imshape[0]) 
    # print(imshape[1])
    # mask = np.zeros_like(blur)  
    # ignore_mask_color = 255
    # lower_left = [imshape[1]/9,imshape[0]]
    # lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
    # top_left = [imshape[1]/2-imshape[1]/6, imshape[0]/2 +imshape[0]/10]
    # top_right = [imshape[1]/2+imshape[1]/6, imshape[0]/2 + imshape[0]/10]
    # vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    # roi_image = cv2.fillPoly(mask, vertices, ignore_mask_color)
    # masked_image = cv2.bitwise_and(edges, roi_image)

    # rho = 4
    # theta = np.pi/180
    # threshold = 30
    # min_line_theta = 100
    # max_line_theta = 180
    # lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), min_line_theta, max_line_theta)
    # line_img = np.zeros_like(frame)
    # #line_img = np.zeros((blur.shape[0], blur.shape[1], 3), dtype=np.uint8)
    # first_frame = 1
    # draw_lines(line_img,lines)
    # result = cv2.addWeighted(frame, 0.8, line_img, 1, 0)

    #cv2.GaussianBlur(img,(5,5),0)

    # Display the resulting frame
    #cv2.imshow('frame', edges)
    cv2.imshow('frame2', edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()