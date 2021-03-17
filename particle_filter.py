import numpy as np
import cv2


def init_particles(state,n):
    particles = np.array([state,]*n)
    return particles
    

def get_view(image,x,y,sq_size):
    """
    Get a smaller image, centered at (x,y) with size (sq_size x sq_size)
    """
    
    # with numpy arrays this is an O(1) operation
    view = image[int(x-sq_size/2):int(x+sq_size/2),
                 int(y-sq_size/2):int(y+sq_size/2),:]
    return view
    
def calc_hist(image, mask=None, Nh=10, Ns=10, Nv=10):
    """
    Computes the color histogram of an image (or from a region of an image).
    
    image: 3D Numpy array (X,Y,RGB)

    return: One dimensional Numpy array
    """
    mask_hs = cv2.inRange(image, np.array((25., 50.,0)), np.array((180.,255.,255.)))
    mask_v = cv2.bitwise_not(mask_hs)
    if mask is not None:
        mask_hs = cv2.bitwise_and(mask, mask_hs)
        mask_v = cv2.bitwise_and(mask, mask_v)
    hist_hs = cv2.calcHist([image],[0,1],mask_hs,[Nh,Ns],[0,180,0,256])
    hist_v = cv2.calcHist([image], [2], mask_v, [Nv], [0,256])
    hist = np.concatenate([hist_hs.flatten(), hist_v.flatten()])
    cv2.normalize(hist,hist,0,1,norm_type=cv2.NORM_MINMAX)
    return hist
    

class ParticleFilter(object):
    def __init__(self,x,y,first_frame,mask=None,n_particles=1000,dt=0.04,
                    window_size=(480,640),square_size=20):
        self.n_particles = n_particles
        self.n_iter = 0
        self.state = np.array([x,y,square_size]) 
        # state =[X[t],Y[t],S[t],X[t-1],Y[t-1],S[t-1]]
        self.std_state = np.array([15,15,1])

        self.window_size = window_size
        
        self.max_square = window_size[0]*0.5
        self.min_square = window_size[0]*0.1
        self.lbd = 20

        self.A = np.array([[1+dt,0,0],
                           [0,1+dt,0],
                           [0,0,1+dt/4]])


        self.B = np.array([[-dt,0,0],
                           [0,-dt,0],
                           [0,0,-dt/4]])


        self.particles = init_particles(self.state,n_particles)
        self.last_particles = np.array(self.particles)
        mask = (mask*255).astype('uint8')
        mask = mask[...,np.newaxis]
        self.hist = calc_hist(get_view(first_frame,x,y,square_size),
                            mask=get_view(mask,x,y,square_size)[:,:,0])
        
     
    def next_state(self,frame):       
      
        control_prediction = self.transition()
        control_prediction = self.filter_borders(control_prediction)
       
        hists = self.candidate_histograms(control_prediction,frame)

        weights = self.compare_histograms(hists,self.hist)
        self.last_particles = np.array(self.particles)
        self.particles = self.resample(control_prediction,weights)
        self.state = np.mean(self.particles,axis=0)


        self.last_frame = np.array(frame)
        self.n_iter += 1
        #self.hist = calc_hist(get_view(frame,self.state[0],self.state[1],self.state[2]))
        

        
        return int(self.state[0]),int(self.state[1]),int(self.state[2]),self.particles,control_prediction
        
        
    def transition(self):

        n_state = self.state.shape[0]
        n_particles = self.particles.shape[0]   
        noises = self.std_state*np.random.randn(n_particles,n_state)
        particles = np.dot(self.particles,self.A) + np.dot(self.last_particles,self.B) + noises
        return particles

    def candidate_histograms(self,predictions,image):
        "Compute histograms for all candidates"
        hists = [] 

        for x in predictions:
            v = get_view(image,x[0],x[1],x[2])
            hists.append(calc_hist(v))
        return hists
    
    def comp_hist(self,hist1,hist2):
        """
        Compares two histograms together using the article's metric

        hist1,hist2: One dimensional numpy arrays
        return: A number
        """
        return np.exp(self.lbd*np.sum((hist1*hist2)**0.5))
        
    def compare_histograms(self,hists,last_hist):
        "Compare histogram of current (last) histogram and all candidates"
        weights = np.array(list(map(lambda x: self.comp_hist(x,last_hist),hists)))
        return weights/np.sum(weights)

    def resample(self,predictions,weights):
        "Scatter new particles according to the weights of the predictions"
        indexes = np.arange(weights.shape[0])
        inds = np.random.choice(indexes,self.n_particles,p=weights)
        return predictions[inds]
    def filter_borders(self,predictions):  
        "Remove candidates that will not have the correct square size."
        np.clip(predictions[:,2],self.min_square,self.max_square,predictions[:,2])
        np.clip(predictions[:,0],predictions[:,2]+1,self.window_size[0]-(1+predictions[:,2]),predictions[:,0])        
        np.clip(predictions[:,1],predictions[:,2]+1,self.window_size[1]-(1+predictions[:,2]),predictions[:,1])
        
        return predictions