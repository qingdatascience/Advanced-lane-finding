# Advanced Lane Finding
## The goals / steps of this project are the following:
Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
Apply a distortion correction to raw images.
Use color transforms, gradients, etc., to create a thresholded binary image.
Apply a perspective transform to rectify binary image ("birds-eye view").
Detect lane pixels and fit to find the lane boundary.
Determine the curvature of the lane and vehicle position with respect to center.
Warp the detected lane boundaries back onto the original image.
Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Camera Calibration
__1. Have the camera matrix and distortion coefficients been computed correctly and
checked on one of the calibration images as a test?__

The code for this step is in lines 8 through 36 of the file called undistorted.py.I first use glob() function to make a file_list for all the chessboard images in the camera_cal folder. I then start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, objp is just a replicated array of coordinates, and objpoints will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. imgpoints will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort() function and obtained this result:

<img src="https://cloud.githubusercontent.com/assets/19335028/23294924/b9a5d07a-fa22-11e6-9a3b-d3ce4e841026.jpg" width="45%"></img> <img src="https://cloud.githubusercontent.com/assets/19335028/23294929/bdb80228-fa22-11e6-90a1-24a74525e1d3.jpg" width="45%"></img> 

##Pipeline (single images)
__1. Has the distortion correction been correctly applied to each image?__

I saved the ‘mtx’ and ‘dist’ values as pickle file, I then read these values before I do the distortion correction, which is the first step in my pipeline. To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

__Before ------------------------------------------------------------------------                                After__

<img src="https://cloud.githubusercontent.com/assets/19335028/23294977/0f2b9ade-fa23-11e6-9179-266ea27625b7.jpg" width="45%"></img> <img src="https://cloud.githubusercontent.com/assets/19335028/23294981/121f3e08-fa23-11e6-8265-7146335d04e5.jpg" width="45%"></img> 

__2. Has a binary image been created using color transforms, gradients or other methods?__

I applied sobel operator on x and y axis on gray scale image. I also selected the s and v channels of hls and hsv color space. I combined them into a binary image. Then I selected the yellow and white color in hsv color space using the following boundaries: 

      lower_yellow = np.array([20, 100, 100])
      upper_yellow = np.array([30, 255, 255])

      lower_white = np.array([0, 0, 150])
      upper_white = np.array([180, 25, 255])


Below shows the processed image, which is the same image showed in step one.
<img src="https://cloud.githubusercontent.com/assets/19335028/23295019/3e713218-fa23-11e6-975c-2663eda95901.jpg" width="45%"></img> 

__3. Has a perspective transform been applied to rectify the image?__

The code for my perspective transform is includes a function called warper(), which appears in lines 71 through 94 in the file image_process.py and video_process.py. The warper() function takes as inputs an image. I designated a a trapizoid region in the center bottom part of the image as source points and a rectangle region as destination points. 

  
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

Please be careful, the image.shape has been switched in img_size and source points, otherwise, it won’t work properly. I display the perspective transformed image here:

<img src="https://cloud.githubusercontent.com/assets/19335028/23295049/5d5d0c38-fa23-11e6-9361-61ce18a2f724.jpg" width="45%"></img> 

__4. Have lane line pixels been identified in the rectified image and fit with a polynomial?__

Then I setup a tracker class in tracker.py to find the lines based on the pixels that I have extracted in the warped_image. I actually can write up a function for processing images, but class is very useful for processing videos, which I will explain later. Please bear with me. 
The algorithm used to find lines is called “Sliding Window”, which uses windows sliding horizontally to search the peak values, which usually means the lines. Numpy convolve() function can serve this purpose conveniently, only two things I need to do: squeezing all the pixels in the layer of one window level into 1D, and defining a search boundary. Here is central piece of code in tracker.py (line 21-47):  
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:],
        axis=0)
        conv_signal = np.convolve(window, image_layer)

After I found all the line points, I fit the line using numpy.polyfit() function. I drew the left and right lines on the original image like this:

<img src="https://cloud.githubusercontent.com/assets/19335028/23295092/9092ccb4-fa23-11e6-94ac-3f7a78ad7fd0.jpg" width="45%"></img> 

__5. Having identified the lane lines, has the radius of curvature of the road been estimated? And the position of the vehicle with respect to center in the lane?__

In order to calculate the radius, I have to do another polyfit, this time with the real world meter values, which were transformed by the ratio reflected by ym_per_pix (meter per pixel in y axis) and xm_per_pix (meter per pixel in x axis). Codes are in line 186-206 in image_process.py. Figure is showed in step 4. 

##Pipeline (video)
__1. Does the pipeline established with the test images work to process the video?__

It sure does! 
* [project video](https://youtu.be/UMuOYm5b3Tg)  
* [challenge video](https://youtu.be/_PeFK5wWzLc)
 
##Discussion
I used Convolute Sliding window approach to find all the line centers. It works pretty well when all the noise pixels have been filtered out and both lines are present. I used cv2.nRange() function to select yellow and white color and used them to rectify the binary images. This approach improved the performance of the code on challenge video. In order to smooth out the lines between video frames, I used the tracker class in tracker.py to store and average the lines. I also checked whether the two lines are parallel before accepted them into self.recent_centers. When I did the average, I put more weight on more recent lines. Here is my code:

        end = window_centroids[-1][1] - window_centroids[-1][0]
        if 0.75 <= float(start)/end <= 1.25:
            self.recent_centers.append(window_centroids)

        l = [1/(2*x) for x in range(1,self.smooth_factor+1)]
        
        if len(self.recent_centers)< self.smooth_factor:
            return np.average(self.recent_centers[-self.smooth_factor:], axis = 0)
            
        # add weight to the average. Recent centers have bigger weight 
        else:
            return np.average(self.recent_centers[-self.smooth_factor:], axis = 0,weights=l[::-1])

I initiated the tracker class before I call “video_clip = clip1.fl_image(process_clip)” in video_process.py, and the instance “curve_centers” became a global variable which can be passed on between clips. My code works pretty well on project video, which does not have many confusing lines on the road and curves are mild. My code works relatively well on challenge video, which has many interfering lines besides the real lines are very faint, but the curves are still mild. My code failed on harder challenge video, which has many sharp curves, sometimes one side of the line disappears all together. It is impossible to pass my sanity check, and all the road information are thrown out. The sun flash make the whole image white, which literally stuns my algorithm too. :)  
To improve the robustness of my code, I think I can fine tune the image process methods to filter out background and enforce the real line signals. Using more sophisticated algorithm such as SVM, random forest or even neural network to do the sanity check would help to pick up more line information rather than simply filtering them out.   
