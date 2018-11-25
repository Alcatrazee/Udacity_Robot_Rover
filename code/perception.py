import numpy as np
import matplotlib.pyplot as plt
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img

    # this block is step 1: update vision_image
    rock_min = np.array([15, 128, 46])
    rock_max = np.array([34, 255, 255])
    # color space convertion
    gray = cv2.cvtColor(Rover.img,cv2.COLOR_RGB2GRAY)
    HSV = cv2.cvtColor(Rover.img,cv2.COLOR_RGB2HSV)
    # get paramerters ready to make affin transformation
    pts1 = np.float32([[119,96],[201,96],[16,140],[301,140]])
    pts2 = np.float32([[155,150],[165,150],[155,160],[165,160]])
    # get sample rock sigment
    rock = cv2.inRange(HSV,rock_min,rock_max)
    rock_transformed = perspect_transform(rock,pts1,pts2)

    transformed = perspect_transform(gray,pts1,pts2)

    ret,obstacle = cv2.threshold(transformed,1,1000,cv2.THRESH_BINARY)
    ret,threshed = cv2.threshold(transformed,150,255,cv2.THRESH_BINARY)

    obstacle-=threshed

    Rover.vision_image[:,:,0] = obstacle
    Rover.vision_image[:,:,1] = rock_transformed
    Rover.vision_image[:,:,2] = threshed

    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image 

    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    
    xpix,ypix = rover_coords(threshed)
    xpix_world,ypix_world = pix_to_world(xpix,ypix,Rover.pos[0],Rover.pos[1],Rover.yaw,200,10)
        
    xpix_obstacle,ypix_obstacle = rover_coords(obstacle)
    xpix_obstacle,ypix_obstacle = pix_to_world(xpix_obstacle,ypix_obstacle,Rover.pos[0],Rover.pos[1],Rover.yaw,200,10)

    xpix_rock,ypix_rock = rover_coords(rock_transformed)
    xpix_rock,ypix_rock = pix_to_world(xpix_rock,ypix_rock,Rover.pos[0],Rover.pos[1],Rover.yaw,200,10)
    
    if (Rover.roll <= 1 or Rover.roll >= 359) and (Rover.pitch <= 1 or Rover.pitch >= 359):
        Rover.worldmap[ypix_obstacle, xpix_obstacle, 0] += 1
        Rover.worldmap[ypix_rock, xpix_rock, 1] += 1
        Rover.worldmap[ypix_world, xpix_world, 2] += 1

    
    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles

    dist, angles = to_polar_coords(xpix, ypix)
    mean_dir = np.mean(angles)

    dist_rock,angles_rock = to_polar_coords(xpix_rock,ypix_rock)

    Rover.nav_dists = dist
    Rover.nav_angles = angles
    # testing code
    if len(angles)!=0:
        Transposed_angles = angles.T
        Rover.line_angles = np.array([])
        index_array = np.where(dist<=170)
        for i in range(0,len(index_array)):
            Rover.line_angles = np.hstack((Rover.line_angles,Transposed_angles[i]))
    #print(np.mean(Rover.line_angles))
    return Rover
