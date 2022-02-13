###############################
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
###############################

class Tracker(object):
    def __init__(self, name=None, data_dir=None, startIndex=1, endIndex=np.inf, form='square', n_particles=100, dt=0.1, firstMask=False, plot=True, video="", step=1, **pf_args):
        # Data frame settings
        self.name = name
        self.path = os.path.join(data_dir, name)
        self.startIndex = startIndex
        self.index = startIndex
        self.endIndex = endIndex
        self.step = step # ***recent***

        # Particule fiter settings
        self.pf = None
        self.form = form
        self.n_particles = n_particles
        self.dt = dt
        self.firstMask = firstMask # Use the first mask to compute the first hist # ***recent***
        self.pf_args = pf_args

        # Plot/video
        self.video = video
        self.out = None
        self.plot = plot

        # Some display consts
        self.BLUE = (0,0,255)
        self.GREEN = (0,255,0)
        self.RED = (255,0,0)
        self.alpha = 0.5
    
    def reset(self, name=False, startIndex=False, endIndex=False, form=False, n_particles=False, dt=False):
        if name: self.name = name
        if startIndex: self.index = startIndex
        if startIndex: self.endIndex = endIndex
        if startIndex: self.form = form
        if startIndex: self.n_particles = n_particles
        if startIndex: self.dt = dt

        self.index = self.startIndex
        self.pf = None
        self.out = None
        self.scores = []

    # Main function
    def track(self, n=-1):
        # Set the stop index
        if n > 0: stopIndex = self.index + n
        else: stopIndex = self.endIndex

        # Set the first frame and initialize the particle filter
        if not(self.pf): self.setFirstFrame()

        # Track the object through the frames
        self.index += self.step
        while os.path.isfile(self.path+'-%0*d.bmp'%(3,self.index)) and self.index <= min(self.endIndex, stopIndex):
            
            # Read next frame
            frame = cv2.imread(self.path+'-%0*d.bmp'%(3,self.index)) 
            orig = np.array(frame)
            self.preprocess(frame)
            
            # Get next state of the particle filter
            x, y, shape_param, distrib, distrib_control = self.pf.next_state(frame)
            
            # Get groundtruth centroid
            mask = cv2.imread(self.path+'-%0*d.png'%(3,self.index))
            x_list, y_list, _ = (mask == 255).nonzero()
            x_gtcentroid, y_gtcentroid = x_list.mean(), y_list.mean()

            # Add score
            score = self.centroidScore(x, y, x_gtcentroid, y_gtcentroid)
            self.scores.append(score)
            
            if self.plot:
                print("\n{}: {}".format(self.name, self.index))
                # Add particles on frame
                self.addDistribPoints(frame, distrib_control, self.RED, 7) # Before resampling
                self.addDistribPoints(frame, distrib, self.BLUE, 3) # After resampling
                #self.addDistribForms(frame, distrib_control, shape_param, self.RED, 7) # Before resampling
                #self.addDistribForms(frame, distrib, shape_param, self.BLUE, 3) # After resampling
                
                # Add the center and the associated bounding box
                cv2.circle(frame, (y, x), 1, self.GREEN, thickness=2) 
                self.addForm(frame, x, y, shape_param, self.GREEN)

                # Add groundtruth centroid
                cv2.circle(frame, (y_centroid, x_gtcentroid), 1, (122,122,255), thickness=2)

                # Set the image to display
                cv2.addWeighted(orig, self.alpha, frame, 1 - self.alpha, 0, frame)   
                self.create_legend(frame, (40,40), (40,20))

                # Plot frame
                plt.figure(self.index)
                plt.imshow(frame)
                plt.show()

            # Save frame to video file
            if self.video: self.out.write(frame)

            # Increment image index
            self.index += self.step

            if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.waitKey(20) & 0xFF == 27):
                break
        
        if self.video: self.out.release()
        cv2.destroyAllWindows()

    def setFirstFrame(self):
        # Read initial image/mask and define bounding box
        mask = cv2.imread(self.path+'-%0*d.png'%(3,self.index))
        frame = cv2.imread(self.path+'-%0*d.bmp'%(3,self.index))
        orig = np.array(frame)
        
        # Preprocess frame
        self.preprocess(frame)

        # Define initial parameters
        x, y, shape_param = self.initParam(mask)

        # True centroid
        x_list, y_list, _ = (mask == 255).nonzero()
        x_gtcentroid, y_gtcentroid = x_list.mean(), y_list.mean()
        self.scores = [self.centroidScore(x, y, x_gtcentroid, y_gtcentroid)]

        # Initialize the particle filter
        if self.firstMask:
            self.pf = ParticleFilter(x, y, frame, mask, self.n_particles, self.dt, shape_param, self.getShapeMask, self.form, **self.pf_args)
        else:
            self.pf = ParticleFilter(x, y, frame, None, self.n_particles, self.dt, shape_param, self.getShapeMask, self.form, **self.pf_args)
    
        if self.plot:
            # Set the image to display
            self.addForm(frame, x, y, shape_param, self.GREEN)
            cv2.addWeighted(orig, self.alpha, frame, 1 - self.alpha, 0, frame)

            # Display infos
            print("Frame shape:", frame.shape)
            print("x, y = {}, {}".format(x,y))
            print("Shape param = {}".format(shape_param))

            # Plot first frame and mask
            plt.figure(1)
            plt.subplot(121)
            plt.imshow(frame)
            plt.subplot(122)
            plt.imshow(mask)
            plt.show()

        # Set a video writer to save frames  in a video file
        if self.video:
            fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
            self.out = cv2.VideoWriter(self.video+'.avi',fourcc, 20.0,(frame.shape[1],frame.shape[0]))
            self.out.write(frame)

    def preprocess(self, frame):
        frame = cv2.blur(frame,(5,5))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def initParam(self, mask):
        """Initialize parameters depending on the form"""
        x_list, y_list, _ = (mask == 255).nonzero()
        x, y = (x_list.max() + x_list.min())/2, (y_list.max() + y_list.min())/2

        if self.form == 'circle':
            shapeParam = [max(y_list.max() - y_list.min(), x_list.max() - x_list.min())//2]

        elif self.form == 'ellipse':
            x, y, shapeParam = self.getEllipseParam(mask, x_list, y_list)

        elif self.form == 'square':
            shapeParam = [max(y_list.max() - y_list.min(), x_list.max() - x_list.min())]

        elif self.form == 'rectangle':
            shapeParam = [x_list.max() - x_list.min(), y_list.max() - y_list.min()]

        return x, y, shapeParam
    
    def centroidScore(self, x1, y1, x2, y2):
        """Compare the history with another one"""
        return ((x1-x2)**2 + (y1-y2)**2)**0.5

    def getShapeMask(self, img, x, y, shape_param):
        """ Return a mask of the input paramametrized form"""
        mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        self.addForm(mask, x, y, shape_param, color=1, thickness=-1)
        return mask

    def addForm(self, img, x, y, shape_param, color, thickness=3):
        """ Add the form in the input image"""
        center = (int(y), int(x))

        if self.form == 'circle':
            radius = int(shape_param[0])
            cv2.circle(img, center, radius, color, thickness)

        elif self.form == 'ellipse':
            r1, r2, angle = tuple(shape_param)
            cv2.ellipse(img, center, (int(r1),int(r2)), angle, 0, 360, color, thickness)

        elif self.form == 'square':
            side = int(shape_param[0])
            start_point = (int(y-side/2), int(x-side/2))
            end_point = (int(y+side/2), int(x+side/2))
            cv2.rectangle(img, start_point, end_point, color, thickness)

        elif self.form == 'rectangle':
            side_x, side_y = int(shape_param[0]), int(shape_param[1])
            start_point = (int(y-side_y/2), int(x-side_x/2))
            end_point = (int(y+side_y/2), int(x+side_x/2))
            cv2.rectangle(img, start_point, end_point, color, thickness)

    def addDistribForms(self, img, distrib, param, color, size=5): 
        n = param.shape[0] + 2
        for d in distrib:
            x, y = tuple(d[:2])
            self.addForm(img, x, y, d[2:n], color, 2)

    def addDistribPoints(self, img, distrib, color, size=5): 
        for d in distrib:
            x, y = tuple(d[:2])
            cv2.circle(img, (int(y), int(x)), 1, color, thickness=size) 

    def create_legend(self, img, pt1, pt2):
        cv2.putText(img, "Before resampling", pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.RED)
        cv2.putText(img, "After resampling", pt2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.BLUE)
 
    # Functions just to initialize an ellipse
    def getEllipseParam(self, mask, x_list, y_list):
        """ Return center coordinates and parameters for an ellipse fitting the mask"""
        # Inverse axis
        x_list, y_list = y_list, x_list

        # Get barycenter to stand as center
        x, y = np.mean(x_list), np.mean(y_list)

        # Set the barycenter as the origin
        x_list, y_list = x_list - x, y_list - y

        # Get distances to the barycenter
        distances = np.sqrt(x_list**2 + y_list**2)

        # Index and radius for the furthest point
        idx = np.argmax(distances)
        radius1_1 = distances[idx]

        # Compute angle between barycenter and the furthest point
        vec1 = np.array([x_list[idx], y_list[idx]])
        angle = self.get_angle(vec1)
        
        # Get radius associated with the opposite point
        radius1_2, vec2 = self.getRadiusAndPoint(angle, x_list, y_list, 180) # Look 180° further

        # Compute the radius of the first ax
        radius1 = (radius1_1 + radius1_2)/2

        # Define a new center in the middle of the first ax of the ellipse
        center = (vec1 + vec2)/2

        # Deplace the origin at the new center
        x_list, y_list = x_list , y_list - center[1]

        # Compute the second radius (can be improved maybe by taking the mean of radius from -90 and +90)
        radius2, _ = self.getRadiusAndPoint(angle, x_list, y_list, 90) # Look 90° further

        # Set return values 
        y_center, x_center = x+center[0], y+center[1] # inverse x and y
        shape_param = [radius1 , radius2, angle]

        return x_center, y_center, shape_param 

    def getRadiusAndPoint(self, angle, x_list, y_list, deg=180):
        radius, vec = 0, np.zeros(2)

        for v in np.dstack((x_list, y_list))[0]: # For each point/vector of the mask
            a = self.get_angle(v)

            if round(a-angle)%360 == abs(round(deg)):
                r = np.linalg.norm(v)
                if r > radius: radius, vec = r, v

        return radius, vec

    def get_angle(self, v):
        """
        return angle between [0,360] with x_axis, trigo orientation
        """
        if np.linalg.norm(v) == 0: return False

        X_axis = np.array([1,0])
        Y_axis = np.array([0,1])

        v_u = v / np.linalg.norm(v)
        angle = np.arccos(np.dot(X_axis, v_u)) * 180/np.pi

        if np.dot(Y_axis, v_u) < 0: # Projection on y axis
            angle = 360 - angle

        return angle

    # Setters and getters
    def setPF(self, pf):
        self.pf = pf
    
    def getPF(self):
        return self.pf

    def setStartIndex(self, startIndex):
        self.startIndex = startIndex
    
    def getStartIndex(self):
        return self.index

    def setIndex(self, index):
        self.index = index
    
    def getIndex(self):
        return self.index

    def setEndIndex(self, endIndex):
        self.endIndex = endIndex
    
    def getEndIndex(self):
        return self.endIndex

    def setForm(self, form):
        self.form = form
    
    def getForm(self):
        return self.form

    def setNParticles(self, n_particles):
        self.n_particles = n_particles
    
    def getNParticles(self):
        return self.n_particles
    
    def setDt(self, dt):
        self.dt = dt
    
    def getDt(self):
        return self.dt

    def setPlot(self, b):
        self.plot=b
    
    def isPlot(self):
        return self.plot

    def setVideo(self, title):
        self.video = title
    
    def getVideo(self):
        return self.video

#############################################################################################################################

class ParticleFilter(object):
    def __init__(self, x, y, first_frame, mask=None, n_particles=100, dt=0.04, shape_param=[20], getShapeFunc=None, form='circle', alpha=0.7, lbd=10, Nh=10, Ns=10, Nv=10, thresh_sat=0.1, thresh_val=0.2, background=False):
        self.n_iter = 0 # Number of iterations
        self.form = form # Shape of the window (string)
        self.n_particles = n_particles # Number of particles
        self.state = np.array([x,y] + shape_param) # Current state
        self.n_state = self.state.shape[0] # Number of state parameters
        self.getShapeFunc = getShapeFunc # Function returning mask from the x, y and shape parameters 

        # Set particles
        self.particles = np.array([self.state]*n_particles)
        self.last_particles = np.array(self.particles) # Keep in memory

        # Define the second-order auto-regressive dynamics
        self.initial, self.A, self.B, self.std_state = self.init_dynamics(dt, len(shape_param))
                                        
        # Define histogram
        self.background_hist = None # Hist of the background, init in calc_hist
        self.background = background
        self.alpha = alpha # Weight to update hist
        self.lbd = lbd # Factor in calculation of the distance of hists
        self.Nh = Nh # nb Hue bins
        self.Ns = Ns # nb Saturation bins
        self.Nv = Nv # nb Value bins
        self.thresh_sat = thresh_sat # Saturation threshold
        self.thresh_val = thresh_val # Value threshold
        self.ref_hist = self.calc_hist(first_frame, x, y, shape_param, mask)

        # Define parameters to limit size
        self.window_size = (first_frame.shape[0],first_frame.shape[1])
        self.max_square = self.window_size[0]*0.7 
        self.min_square = self.window_size[0]*0.1

    def next_state(self,frame):       
        
        # Prediction from second-order AR dynamics (candidate particles)
        control_prediction = self.transition()
        control_prediction = self.filter_borders(control_prediction)

        # Compute candidate histograms
        hists = self.candidate_histograms(control_prediction, frame)

        # Compute weights
        weights = self.compare_histograms(hists, self.ref_hist)
        
        # Resample candidate particles
        self.last_particles = np.array(self.particles)
        self.particles = self.resample(control_prediction, weights)

        # Compute current state
        self.state = np.mean(self.particles, axis=0)
        
        # Update the reference histogram (weighted between current and previous hists)
        current_state_hist = self.calc_hist(frame, self.state[0], self.state[1], self.state[2:])
        new_hist = self.ref_hist*self.alpha + current_state_hist*(1-self.alpha)
        
        # Plot histogram (ref and current)
        # plt.figure()
        # plt.plot(self.ref_hist, label='ref')
        # plt.plot(self.calc_hist(frame, self.state[0], self.state[1], self.state[2:]), label='mean', alpha=0.5)
        # plt.legend()
        # plt.show()

        self.ref_hist = new_hist

        # Increment iteration
        self.n_iter += 1

        # Set return values into integers
        x = int(self.state[0])
        y = int(self.state[1])
        shape_param = self.state[2:].astype(np.uint16)

        return x, y, shape_param, self.particles, control_prediction

    def init_dynamics(self, dt, n_param):
        n = 2 + n_param # Center (x,y) and shape parameters

        init_noise_factor = 2

        A = np.identity(n)
        A[:2] *= 1+dt
        A[2:] *= 1+dt/4 # Factor for shape parameters

        B = np.identity(n)*(-dt)
        B[2:] *= 1/4 # Factor for shape parameters

        std = np.ones(n)
        std[:2] *= 15 # Factor for x and y

        if self.form == 'ellipse':
            std[-1] *= 20 # More amplitude for the angle parameter

        return init_noise_factor, A, B, std
                   
    def transition(self):
        if self.n_iter  == 0:
            noises = self.std_state * np.random.randn(self.n_particles, self.n_state) * self.initial
            particles = self.particles + noises
        else:
            noises = self.std_state * np.random.randn(self.n_particles, self.n_state)
            particles = self.particles@self.A + self.last_particles@self.B + noises
        return particles

    def calc_hist(self, frame, x, y, shape_param, mask=None):
        """
        Computes the color histogram of an image 
        return: One dimensional Numpy array
        """
        
        # Get pixels from the area define by the shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Mask from the first frame
        if mask is None: 
            mask = self.getShapeFunc(hsv, x, y, shape_param)
        elif len(mask.shape) == 3:
            mask = mask[:,:,0]  

        # Initiatlize background histogram
        if self.background and self.background_hist is None:
           self.background_hist = True # Not None anymore
           inverse_mask = 1 - mask
           self.background_hist = self.calc_hist(frame, False, False, False, inverse_mask)

        # Apply threshold creating to images
        image_hs = np.array(hsv, np.uint16)
        image_v = np.array(hsv, np.uint16)
        
        image_v[(hsv[:,:,1] > self.thresh_sat*180) & (hsv[:,:,2] > self.thresh_val*255), 0] = -1 # (not to be taken into account in the hist)
        image_hs[image_v[:,:,0] != -1, 2] = -1 # same...
        
        # Compute H/S and V histograms 
        Nbins = self.Nh*self.Ns + self.Nv # Total number of bins
        hist_hs = cv2.calcHist([image_hs], [0, 1], mask, [self.Nh, self.Ns], [0, 181, 0, 256]) # Hue/Saturation histogram
        hist_v = cv2.calcHist([image_v], [2], mask, [self.Nv], [0, 256]) # Value histogram
        
        # Normalize histograms
        cv2.normalize(hist_hs, hist_hs, 0, 1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_v, hist_v, 0, 1, norm_type=cv2.NORM_MINMAX)
            
        # Concatenate both histograms (weighted)
        hist = np.concatenate((hist_hs.flatten()*self.Nh*self.Ns/Nbins, hist_v.flatten()*self.Nv/Nbins))
        return hist

    def candidate_histograms(self, predictions, image):
        "Compute histograms for all candidates"
        hists = [] 
        for state in predictions:
            hists.append(self.calc_hist(image, state[0], state[1], state[2:]))
        return hists
        
    def compare_histograms(self, hists, last_hist):
        "Compare histogram of current (last) histogram and all candidates"
        weights = np.array(list(map(lambda x: self.distanceHist(x, last_hist), hists)))
        weights /= np.sum(weights)
        return weights

    def distanceHist(self, hist1, hist2):
        """
        Compares two histograms together NOT using the article's metric
        Replace Battacharrya by L2 norm
        hist1,hist2: One dimensional numpy arrays
        return: A number
        """
        if self.background:
            return np.exp(-self.lbd*(np.linalg.norm(hist2-hist1)-np.linalg.norm(self.background_hist-hist1))) # Background taken into account
        else:
            return np.exp(-self.lbd*np.linalg.norm(hist2-hist1))
        #return np.exp(lbd*np.sum(np.sqrt(hist1*hist2)))
     
    def resample(self, predictions, weights):
        "Scatter new particles according to the weights of the predictions"
        indexes = np.arange(weights.shape[0])
        new_indexes = np.random.choice(indexes, self.n_particles, p=weights)
        return predictions[new_indexes]

    def filter_borders(self, predictions):  
        "Remove candidates that will not have the correct sizes."

        if self.form in ['square', 'circle']: 
            side_x, side_y = self.state[2]//2, self.state[2]//2
            np.clip(predictions[:,2], self.min_square, self.max_square, predictions[:,2]) # scale 

        elif self.form == 'rectangle':
            side_x, side_y = self.state[2]//2, self.state[3]//2
            np.clip(predictions[:,2], self.min_square, self.max_square, predictions[:,2]) # scale 
            np.clip(predictions[:,3], self.min_square, self.max_square, predictions[:,3]) # scale 

        elif self.form == 'ellipse': # Not correct
            side = (self.state[2] + self.state[3])//4
            side_x, side_y = side, side
            np.clip(predictions[:,2], self.min_square//2, self.window_size[0], predictions[:,2]) # scale 
            np.clip(predictions[:,3], self.min_square//2, self.window_size[0], predictions[:,3]) # scale 

        np.clip(predictions[:,0], side_x+1, self.window_size[0]-(1+side_x), predictions[:,0]) # x
        np.clip(predictions[:,1], side_y+1, self.window_size[1]-(1+side_y), predictions[:,1]) # y
        

        return predictions

