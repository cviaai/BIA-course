import numpy as np
import math
import cv2
import os, sys, random
import gym
from gym import spaces
from pathlib import Path
import matplotlib.pyplot as plt
    
    
def load_image_mask(path, img_path, msk_path):
    """ Load image and mask (in final prototype will be received from previous step in pipeline)

    Parameters:
        path: relative path to folder with image and mask files
        img_path: image file name (with extension)
        msk_path: mask file name (with extension)

    Returns:
        image: image loaded
        mask: mask loaded
    """
    #print(os.path.join(path, img_path))
    image = cv2.imread(os.path.join(path, img_path))
    mask = cv2.cvtColor(cv2.imread(os.path.join(path, msk_path)), cv2.COLOR_BGR2GRAY)

    thrshd = 100  ### to delete artifacts
    mask[mask > thrshd] = 255
    mask[mask <= thrshd] = 0
    
     
    return image, mask    
    
            
def mask_contour(mask):
    """ Given mask return an outline mask (rectangle mask contour with width: bord)

    Parameters:
        mask:     image (int n*m matrix).  vein mask 
    Returns:
        contour:  image (int n*m matrix). grayscale contour of vein mask
    """

    contour = mask.copy()
    vett = np.array(np.nonzero(contour))
    bord=50
    rows_up=[max(vett[0].min()-bord,0), vett[0].min()]
    rows_down=[vett[0].max()+1, min(vett[0].max()+bord+1,contour.shape[0])]
    cols_left=[max(vett[1].min()-bord,0), vett[1].min()]
    cols_right=[vett[1].max()+1, min(vett[1].max()+bord+1,contour.shape[1])]
    
    roi = [rows_up, rows_down, cols_left, cols_right]
    
    contour.fill(0)
    contour[rows_up[0]:rows_up[1], cols_left[0]:cols_right[1]] = 255
    contour[rows_down[0]:rows_down[1], cols_left[0]:cols_right[1]] = 255
    contour[rows_up[0]:rows_down[1], cols_left[0]:cols_left[1]] = 255
    contour[rows_up[0]:rows_down[1], cols_right[0]:cols_right[1]] = 255

    return contour

def gen_mask_rectangle(mask):
    """ Given vein mask return a rectangle mask of the initial area

    Parameters:
        mask:     image (int n*m matrix). grayscale vein mask 
    Returns:
        rectangle:  image (int n*m matrix). grayscale rectangle mask
    """
    rectangle = mask.copy()
    rectangle.fill(0)
    vett_mask = np.array(np.nonzero(mask))
    roi_mask = [vett_mask[0].min(),vett_mask[0].max()+1, vett_mask[1].min(),vett_mask[1].max()+1]
    rectangle[roi_mask[0]:roi_mask[1], roi_mask[2]:roi_mask[3]] = 255 
    return rectangle    
    
def mask_crop_colour(mask):
    """ Given mask return a projectable version of it cropping out the empty area outside the 
    minimum enclosing rectangle of the vein mask, switching veins to red & the empty area to white

    Parameters:
        mask:       image (int n*m matrix). grayscale vein mask 
    Returns:
        proj_mask:  image (int n*m*c matrix). vein mask suitable for projection
    """
    
    temp = mask.copy()
    vett = np.array(np.nonzero(temp))
    roi = [vett[0].min(),vett[0].max()+1, vett[1].min(),vett[1].max()+1]
    cut  = temp[roi[0]:roi[1], roi[2]:roi[3]]
    colored = np.zeros((cut.shape[0], cut.shape[1], 3), dtype=np.uint8)
    colored.fill(255)

    if(len(cut[:, :].shape)<3 or cut[:, :].shape[2]==1):
        colored[...,0] = 255-cut[:, :]
        colored[...,1] = 255-cut[:, :]
    else:
        colored[...,0] = 255-cut[:, :, 2]
        colored[...,1] = 255-cut[:, :, 2]
    
    return colored
    
    
    
def rotate_scale(mask, angle=0, scale=1):
    """ Given colored mask, an angle and a scaling factor return the mask
    after the rotation and scaling, the new empty areas are set to black

    Parameters:
        mask:  image (int n*m*c matrix). colored vein mask suitable for projection
        angle: int. degrees of rotation (counterclockwise orientation)
        scale: float. scaling factor (e.g. 1.01 scale up 1%; 0.90 scale down to 90% of original size)
    Returns:
        rs_mask:  image (int n*m*c matrix). vein mask rotated and scaled
    """
    # grab the dimensions of the image and then determine the
    # center
    
    img = mask.copy()
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix then grab the sine and cosine
    M = cv2.getRotationMatrix2D((cX, cY), angle, scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    rs_mask = cv2.warpAffine(img, M, (nW, nH), borderValue=(0, 0, 0))
    
    #rs_mask[rs_mask > 0] = 255
    
    return  rs_mask 
    
    
def sim_projection(image, colored_mask, tr_rows, tr_cols, angle=0, scale=1.0):
    """ Given an image, a colored mask and transformation parameters merge the mask on the image 
    to simulate the physical projector. Return the merged image.

    Parameters:
        image:  image (int n*m*c matrix). image from camera 
        colored_mask: (int n*m*c matrix). colored vein mask suitable for projection
        tr_rows: int. vertical mask translation WRT the image   (start from top)
        tr_cols: int. horizontal mask translation WRT the image (start from left)
        angle: int. degrees of rotation (counterclockwise orientation)
        scale: float. scaling factor (e.g. 1.01 scale up 1%; 0.90 scale down to 90% of original size)
    Returns:
        merged:  image (int n*m*c matrix). vein mask rotated and scaled
    """
    #make copies of image and mask
    merged = image.copy()
    projection = colored_mask.copy()

    # sanity check for integers
    tr_rows = int(tr_rows)
    tr_cols = int(tr_cols)
    angle = int(angle)
    
    # apply rotation and scaling of mask
    rotated = rotate_scale(projection, angle, scale)
    
    # if in the previous step the mask swas rotated the size overall size changed
    # need to recalculate the translation offsets WRT the new dimensions
    # delta_r is how much the height changed after rotate_scale() function. 
    # since it affect both the upper and lower part of the mask it's divided by 2
    # same for delta_c for witdh
    delta_r = int((rotated.shape[0]-colored_mask.shape[0])/2)
    delta_c = int((rotated.shape[1]-colored_mask.shape[1])/2)

    # sanity check in order to write over the original image 
    # avoid underflow (try to write on negative indexes)
    delta_r_min = np.abs(min(tr_rows - delta_r, 0))
    delta_c_min = np.abs(min(tr_cols - delta_c, 0))
    
    # sanity check in order to write over the original image 
    # avoid overflow (try to write on indexes bigger than the image size)    
    delta_r_max = np.abs(max(tr_rows -delta_r + rotated.shape[0] - merged.shape[0], 0))
    delta_c_max = np.abs(max(tr_cols -delta_c + rotated.shape[1] - merged.shape[1], 0))

    # 
    prj_crds=[tr_rows -delta_r + delta_r_min, tr_rows -delta_r + rotated.shape[0] - delta_r_max, tr_cols -delta_c + delta_c_min, tr_cols -delta_c + rotated.shape[1]-delta_c_max]

    img_overlap = cv2.addWeighted(merged[prj_crds[0]:prj_crds[1], prj_crds[2]:prj_crds[3]], 0.9, rotated[delta_r_min:rotated.shape[0]-delta_r_max, delta_c_min:rotated.shape[1]-delta_c_max], 0.7, 0)

    merged[prj_crds[0]:prj_crds[1], prj_crds[2]:prj_crds[3]] = img_overlap
    
    return merged    
    
    
def analyze_image_mask(image, mask, init_roi):
    """ Given image, mask and initial intensity mask values calc the average current intensity of the mask area

    Parameters:
        image:     image (int n*m[*c] matrix). [grayscale]
        mask:      image (int n*m matrix). grayscale
        init_roi:  array (int), the initial intensity of the pixels in the mask area
    Returns:
        align:    int 0 to 1000. metric mapping the number of pixels in image (inside the mask area) that 
                   have an increased intensity value (WRT their own value before switching on the projector).  
    """
    img = image
    # if the image is 3 channel take only the RED (opencv order is BGR)
    if len(image.shape)>2 and image.shape[2]==3:
        img = image[:,:,2]
       
    roi=img[mask>0]
    
    """ explanation of the next line: 
    - count the element of the mask that have an increased intensity value 
    (greater than their value when the projector was off).
    - add the pixels that eventually where already maxed before the projection 
    (their intensity was already 255 so there can not be an increment)
    - multiply by 1000 and divide by the number of pixels present in the mask, (this way the metric goes from 0 to 1000)
    """
    return int(np.round(np.divide((np.greater(roi, init_roi).sum()+len(init_roi[init_roi==255]))*1000, len(roi))))


   
def analyze_image(image, mask, contour, val_mask, val_contour):
    """ Given image, mask, contour, val_mask and val_contour calc the "projection alignment" 
    as mean intensity in our ROI minus the increased intensity in the surrounding (contour) area 

    Parameters:
        image:         image (int n*m[*c] matrix). [grayscale] image from camera 
        mask:          image (int n*m matrix). grayscale vein mask 
        mask_contour:  image (int n*m matrix). grayscale contour of vein mask
        val_mask:      array (int). initial intensity value of the mask area
        val_outl:      array (int). initial intensity value of the mask_outline area
    Returns:
        align:      int. metric for alignment
    """
    
    img = image.copy()
    # if the image is 3 channel take only the RED (opencv order is BGR)
    if len(image.shape)>2 and image.shape[2]==3:
        img = image[:,:,2]

    roi_mask = img[mask>0] 
    roi_contour = img[contour>0]
    
    """ explanation of the next line: 
    - count the element of the mask that have an increased intensity value 
    (greater than their value when the projector was off).
    - add the pixels that eventually where already maxed before the projection 
    (their intensity was already 255 so there can not be an increment)
    - multiply by 1000 and divide by the number of pixels present in the mask, (this way the metric goes from 0 to 1000)
    """
    metr_mask = int(np.round(np.divide((np.greater(roi_mask, val_mask).sum()+len(val_mask[val_mask==255]))*1000, len(roi_mask))))
    
    """
    same procedure but for the outline area around the mask
    the only difference is that we do not add elements initially equal to 255, since there is
    no way to affect those with the mask projection
    """
    metr_contour = int(np.round(np.divide(np.greater(roi_contour, val_contour).sum()*1000, len(roi_contour))))
    
    # return metr_mask minus metr_outline (we want the projection over the mask and not over the surrounding area)
    return metr_mask-metr_contour

    
    
 




#########################################################    
#########################################################    
######################################################### 

class Mask_Proj_Env(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is an env where the agent must learn to just follow the reward function. 
    """
    # Define constants for clearer code
    STILL = 0       # no change
    UP    = 1       # up
    DOWN  = 2       # down
    LEFT  = 3       # left
    RIGHT = 4       # right
    CLOCKWISE = 5   # one degree clockwise rotation
    COUNTER = 6     # one degree counterclockwise rotation
    INCREASE = 7    # one percent increase
    DECREASE = 8    # one percent decrease
    nA = 9          # total number of possible actions
    
    
    action_names =  {0:"STILL", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 
                     5: "CLOCKWISE", 6: "COUNTER", 7: "INCREASE", 8: "DECREASE"}
    
    LIMIT=20
        
        
        
    def __init__(self, image, mask):
        super(Mask_Proj_Env, self).__init__()
      
        self.merge=None
        self.debug = False
        
        #image, vein_mask 
        self.image, self.vein_mask = image.copy(), mask.copy()
        # rectangle mask
        self.rectangle_mask = gen_mask_rectangle(self.vein_mask)
        
        # colored mask for projection
        self.colored = mask_crop_colour(self.vein_mask)      

        # contour area (where to look for negative contribution)
        self.contour = mask_contour(mask)   

        self.reward = None
        self.prev_true_obs = None
        
        
        self.image_rows = self.image.shape[0]
        self.image_cols = self.image.shape[1]
        
        self.mask_rows = self.colored.shape[0]
        self.mask_cols = self.colored.shape[1]
        
        self.nrows = self.image_rows - self.mask_rows
        self.ncols = self.image_cols - self.mask_cols   
    
        
        self.max_rot = 10
        self.nrot = self.max_rot*2 +1        
        
        self.max_scale = 0.2
        self.list_scale = np.round(np.arange(1-self.max_scale, 1+self.max_scale, 0.01), 2)
        self.nscale = len(self.list_scale)        
                
        
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(self.nA)
        # The observation will be the coordinate of the agent
        self.low_state=np.array([0], dtype=np.int16)
        self.high_state=np.array([1000], dtype=np.int16)
        self.observation_space = spaces.Box(low= self.low_state, high=self.high_state, dtype=np.int16)  


        coord = np.array(np.nonzero(self.vein_mask))

        self.target_row = coord[0].min()
        self.target_col = coord[1].min()
        self.target_rot = 0
        self.target_scale = int(self.nscale/2)        
        
        
        
        self.manhattan = 0
 
        # agent relative positions start
        # self.start_row = int(self.target_row)    
        # self.start_col = int(self.target_col)    
        # self.start_rot = self.target_rot        
        # self.start_scale =  self.target_scale 

        
        
    def reset(self):

        self.merge = None
        self.done  = False
        self.count = 0
        
        self.action = 0
        self.prev_true_obs = None
        
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        
        
        # this is necessary for simulating an initial displacement of max 1/4 in horizontal and vertical
        max_delta_rows = self.colored.shape[0]/4
        max_delta_cols = self.colored.shape[1]/4 
        
        
        coord = np.array(np.nonzero(self.vein_mask))
        #print(coord[0].min())
        #print(coord[1].min())
       
        
        self.target_row = int(coord[0].min())
        self.target_col = int(coord[1].min())
        self.target_rot = 0
        self.target_scale = int(self.nscale/2)

        
        # agent relative positions start
        self.agent_row = np.random.randint(self.target_row-max_delta_rows, self.target_row+max_delta_rows)
        self.agent_col = np.random.randint(self.target_col-max_delta_cols, self.target_col+max_delta_cols) 
        self.agent_rot = random.randint(-self.max_rot, self.max_rot) 
        self.agent_scale = random.randint(0, self.nscale-1)

        self.manhattan = np.maximum(np.abs(self.target_row - self.agent_row) + np.abs(self.target_col - self.agent_col) +
                                    + np.abs(self.target_rot - self.agent_rot) + np.abs(self.target_scale - self.agent_scale), 1)

#        self.agent_row = int(self.start_row +20)# int(self.start_row +  max_delta_rows) #random.choice([-1, +1]) *
#        self.agent_col = int(self.start_col -20)#random.choice([-1, +1]) *
#        self.agent_rot = self.start_rot -5
#        self.agent_scale = self.start_scale -5

        # vector with initial intensity values for the mask and contour area
        if len(self.image.shape)>2 and self.image.shape[2]==3:
            img = self.image[:,:,2]
            self.mask_roi = img[self.rectangle_mask>0]
            self.contour_roi = img[self.contour>0]
        else:
            self.mask_roi = self.image[self.rectangle_mask>0]
            self.contour_roi = self.image[self.contour>0]
                    
        if self.debug:
            print("Agent start position: {}, {}, {}, {}".format(self.agent_row, 
            self.agent_col, self.agent_rot, self.list_scale[self.agent_scale]))
        
        self.observation = self.calc_metric()
        #self.observation, self.reward = self.process_observation(self.observation)
        return self.observation, 0 #self.reward
    
    
    
    def calc_metric(self):
        # merge background image and mask
        self.merge = sim_projection(self.image, self.colored, self.agent_row, self.agent_col, self.agent_rot, self.list_scale[self.agent_scale])          

        # calculate the metric
        metric = analyze_image(self.merge, self.rectangle_mask, self.contour, self.mask_roi, self.contour_roi)

        return metric    

    def set_start(self, row, col, rot, scale):

        self.agent_row = self.target_row-row
        self.agent_col = self.target_col-col
        self.agent_rot = 0 - rot
        self.agent_scale = self.target_scale -scale
        self.observation = self.calc_metric()
        #print(self.agent_row)
        #print(self.agent_col)
        return self.observation #self.reward
               
    def step(self, action, debug=False):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
                    
        done=False
        self.merge = None
        
        if action == self.UP:
            if self.agent_row > 0:
                self.agent_row -= 1
        elif action == self.DOWN:
            if self.agent_row < self.image_rows-1:
                self.agent_row += 1
        elif action == self.LEFT:
            if self.agent_col > 0:
                self.agent_col -= 1               
        elif action == self.RIGHT:
            if self.agent_col < self.image_cols-1:
                self.agent_col += 1   
        elif action == self.CLOCKWISE:
            if  self.agent_rot > -self.max_rot:
                 self.agent_rot -= 1               
        elif action == self.COUNTER:
            if  self.agent_rot < self.max_rot:
                 self.agent_rot += 1   
        elif action == self.INCREASE:
            if self.agent_scale < len(self.list_scale)-1:
                 self.agent_scale += 1
        elif action == self.DECREASE:
            if self.agent_scale > 0:
                 self.agent_scale -= 1        
        elif action != self.STILL:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        self.action = action
             

        # simulate projection over video
        self.observation = self.calc_metric()
        
        if debug:
            print("env metric: " + str(self.observation))
        
        # if the metric is equal to 1000 the overlap is perfect
        # to make the env kind of sequential the agent should remain
        # perfectly aligned for self.LIMIT times
        if self.observation==1000 and self.prev_true_obs==1000:
            self.count+=1
            if self.count>self.LIMIT:
                done = True
        
        # Optionally we can pass additional info
        info = {'Row_Agent':self.agent_row , 'Column_Agent': self.agent_col, 'Rotation_Agent': self.agent_rot, 'Scale_Agent': self.list_scale[self.agent_scale]}        
        #info["Manhattan"]= np.maximum(np.abs(self.target_row - self.start_row) + np.abs(self.target_col - self.start_col) +
        #                            + np.abs(self.target_rot - self.start_rot) + np.abs(self.target_scale - self.start_scale), 1)
        #self.observation, self.reward = self.process_observation(self.observation)
        self.prev_true_obs = self.observation
        return self.observation, self.reward, done, info
   
   
    def render(self, debug=False):
        #image = sim_projection(self.image, self.colored, self.agent_row, self.agent_col, self.agent_rot, self.list_scale[self.agent_scale])
        if debug:
            cv2.putText(self.merge, text= 'row: ' + str(self.agent_row), org=(10,170),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,255,0),
                    thickness=2, lineType=cv2.LINE_AA)

            cv2.putText(self.merge, text= "col: "  + str(self.agent_col), org=(10,200),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,255,0),
                    thickness=2, lineType=cv2.LINE_AA)
            
            cv2.putText(self.merge, text= "rot: " + str(self.agent_rot), org=(10,230),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,255,0),
                    thickness=2, lineType=cv2.LINE_AA)
            
            cv2.putText(self.merge, text= "scale: "  + str(self.list_scale[self.agent_scale]), org=(10,260),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,255,0),
                    thickness=2, lineType=cv2.LINE_AA)    
            
            cv2.putText(self.merge, text= "observation: "  + str(self.observation), org=(10,290),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,255,0),
                    thickness=2, lineType=cv2.LINE_AA)
            
            cv2.putText(self.merge, text= "action: "  + self.action_names[self.action], org=(10,320),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,255,0),
                    thickness=2, lineType=cv2.LINE_AA)                    
                      
        return self.merge
      