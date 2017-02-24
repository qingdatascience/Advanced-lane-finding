import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from tracker import tracker


dist_pickle = pickle.load( open( "./camera_cal/calibration.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image and grayscale it
image = mpimg.imread('./test_images/test3.jpg')

def imgselect(img, sthresh=(0, 255), vthresh=(2,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_out_s = np.zeros_like(s_channel)
    binary_out_s[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    binary_out_v = np.zeros_like(v_channel)
    binary_out_v[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

    # define range of yellow and white color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 25, 255])

    # Threshold the HSV image to get yellow and white colors
    mask_w = cv2.inRange(hsv, lower_white, upper_white)
    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)

    output = np.zeros_like(s_channel)
    output[(binary_out_v==1)&(binary_out_s==1)] = 1

    return output, mask_w, mask_y


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1,ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

    # draw windows 
def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2)\
        ,img_ref.shape[1])] = 1
    return output


def warped(img):
     # Grab the image shape
    img_size = (img.shape[1], img.shape[0])

    offset = img_size[0]*.25 # offset for dst points
    bot_width = .69
    mid_width = .148
    height_pct = .67
    bottom_trim = .99
            # For source points, grabbing a trapizoid region 
    src = np.float32([[img.shape[1]*(.5-mid_width/2), img.shape[0]*height_pct],[img.shape[1]*(.5+mid_width/2)\
        ,img.shape[0]*height_pct], [img.shape[1]*(.5+bot_width/2),img.shape[0]*bottom_trim]\
        ,[img.shape[1]*(.5-bot_width/2),img.shape[0]*bottom_trim]])

    dst = np.float32([[offset, 0], [img_size[0]-offset, 0]\
        ,[img_size[0]-offset, img_size[1]],[offset, img_size[1]]])
        # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
        # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size,flags=cv2.INTER_NEAREST)

    # Return the resulting image and matrix
    return warped, img_size, Minv
  
    
# Run the function
ksize = 3

image_ = cv2.undistort(image, mtx, dist, None, mtx)

# select pixels and make binary

gradx = abs_sobel_thresh(image_, orient='x', sobel_kernel=ksize, thresh=(12, 255))
grady = abs_sobel_thresh(image_, orient='y', sobel_kernel=ksize, thresh=(25, 255))

#select s and v channels
c_binary, mask_w, mask_y = imgselect(image_, sthresh=(90, 255),vthresh=(50,255))

processed_img = np.zeros_like(gradx)
processed_img[((gradx == 1) & (grady == 1)|(c_binary == 1))] = 1
processed_img *= mask_w
processed_img += mask_y


#perspective transformation
warped_img, img_size, Minv = warped(processed_img)

# locate line pixels by sliding door convolution 
window_width = 25 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 40 # How much to slide left and right for searching


curve_centers = tracker(window_width, window_height,margin,ym = 10/728, xm = 4/667, smooth_factor = 15)

window_centroids = curve_centers.find_window_centroids(warped_img)

    # Points used to draw all the left and right windows
l_points = np.zeros_like(warped_img)
r_points = np.zeros_like(warped_img)

leftx = []
rightx = []

# Go through each level and draw the windows    
for level in range(0,len(window_centroids)):
    # Window_mask is a function to draw window areas
    leftx.append(window_centroids[level][0])
    rightx.append(window_centroids[level][1])
    l_mask = window_mask(window_width,window_height,warped_img,window_centroids[level][0],level)
    r_mask = window_mask(window_width,window_height,warped_img,window_centroids[level][1],level)
    # Add graphic points from window mask here to total pixels found 
    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

# Draw the results
template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
zero_channel = np.zeros_like(template) # create a zero color channle 
template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
warpage = np.array(cv2.merge((warped_img,warped_img,warped_img)),np.uint8) # making the original road pixels 3 color channels
output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
 

# fit the line 
yvals = range(0,warped_img.shape[0])

res_yvals = np.arange(warped_img.shape[0]-(window_height/2),0,-window_height)

left_fit = np.polyfit(res_yvals, leftx,2)
left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
left_fitx = np.array(left_fitx,np.int32)

right_fit = np.polyfit(res_yvals, rightx,2)
right_fitx = right_fit[0]*yvals*yvals+ right_fit[1]*yvals + right_fit[2]
right_fitx = np.array(right_fitx,np.int32)

left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
middle_marker = np.array(list(zip(np.concatenate((left_fitx+window_width/2,right_fitx[::-1]-window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

road = np.zeros_like(image)
road_bkg = np.zeros_like(image)
cv2.fillPoly(road,[left_lane],color=[255,0,0])
cv2.fillPoly(road,[right_lane],color=[0,0,255])
cv2.fillPoly(road,[middle_marker],color=[0,200,0])
cv2.fillPoly(road_bkg,[left_lane],color=[255,255,255])
cv2.fillPoly(road_bkg,[right_lane],color=[255,255,255])

road_warped = cv2.warpPerspective(road, Minv, img_size,flags=cv2.INTER_NEAREST)
road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size,flags=cv2.INTER_NEAREST)

road_base = cv2.addWeighted(image, 1, road_warped_bkg, -1.0, 0.0)
road_combined = cv2.addWeighted(road_base, 1, road_warped, 0.7, 0.0)

ym_per_pix = curve_centers.ym_per_pix
xm_per_pix = curve_centers.xm_per_pix

#calculate the offset of the center on the road
camera_center = (left_fitx[-1] + right_fitx[-1])/2
center_diff = (camera_center - warped_img.shape[1]/2)*xm_per_pix
side_pos = 'left'
if center_diff <= 0:
    side_pos = 'right'

left_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix, np.array(leftx,np.float32)*xm_per_pix, 2)
right_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix, np.array(rightx,np.float32)*xm_per_pix, 2)

# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*yvals[-1]*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*yvals[-1]*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])


#display the info 
cv2.putText(road_combined,'Radius of Curvature = '+str(round((left_curverad+right_curverad)/2,3))+'(m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
cv2.putText(road_combined,'Vehicle is '+str(abs(round(center_diff,3)))+'(m) '+side_pos+' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

# save the result

cv2.imwrite('./test_images/test3_withLine.jpg',cv2.cvtColor(road_combined, cv2.COLOR_BGR2RGB))


