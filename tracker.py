import numpy as np

class tracker():
    def __init__(self,win_width, win_height,margin,ym = 1, xm = 1, smooth_factor = 15):
        # store past center values
        self.recent_centers = [] 

        # window pixel width of the center values
        self.window_width = win_width 
        #window pixel height of the center values
        self.window_height = win_height   
        #pixel distence in both direction to slide
        self.margin = margin  
        #meters per pixel in vertical axis
        self.ym_per_pix = ym 
        #meters per pixel in horizontal axis
        self.xm_per_pix = xm

        self.smooth_factor = smooth_factor

    def locat_centroids(self, l_center,r_center, warped):
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin
        offset = window_width/2        
        window_centroids_ = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions

        for level in range(1,(int)(warped.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)

            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window

         
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
            if sum(conv_signal[l_min_index:l_max_index]) >= 1000:
                l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
            
            if sum(conv_signal[r_min_index:r_max_index]) >= 1000:
                r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
             
              
            # Add what we found for that layer
            window_centroids_.append((l_center,r_center))
        return window_centroids_

    def find_window_centroids(self, warped):

        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin
        offset = window_width/2        
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions
        
        # if it is not the first frame, use the centers from pervious frame to start 

        if len(self.recent_centers) >0:
            l_center = self.recent_centers[-1][0][0]
            r_center = self.recent_centers[-1][0][1]
            start = r_center - l_center
            window_centroids.append((l_center,r_center))
            centers_above=self.locat_centroids(l_center,r_center,warped)
            for i in range(len(centers_above)):
                window_centroids.append(centers_above[i])
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        else:

            l_sum = np.sum(warped[int(2*warped.shape[0]/3):,:int(warped.shape[1]/2)], axis=0)
            l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
            r_sum = np.sum(warped[int(2*warped.shape[0]/3):,int(warped.shape[1]/2):], axis=0)
            r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
            start = r_center - l_center
            # Add what we found for the first layer
            window_centroids.append((l_center,r_center)) 

            centers_above=self.locat_centroids(l_center,r_center,warped)
            for i in range(len(centers_above)):
                window_centroids.append(centers_above[i])
            
        # check to see whether the left and right centers are roughly parallel

        end = window_centroids[-1][1] - window_centroids[-1][0]
        if 0.75 <= float(start)/end <= 1.25:
            self.recent_centers.append(window_centroids)

        l = [1/(2*x) for x in range(1,self.smooth_factor+1)]
        
        if len(self.recent_centers)< self.smooth_factor:
            return np.average(self.recent_centers[-self.smooth_factor:], axis = 0)
            
        # add weight to the average. Recent centers have bigger weight 
        else:
            return np.average(self.recent_centers[-self.smooth_factor:], axis = 0,weights=l[::-1])
