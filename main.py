import os
import cv2
from particle_filter import ParticleFilter
import numpy as np

def create_legend(img,pt1,pt2):
    text1 = "Before resampling"
    cv2.putText(img,text1, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
    text2 = "After resampling"
    cv2.putText(img,text2, pt2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
    
def main():
    name = "book"
    im_begin, im_end = 1, 26
    wdir = "../../data/sequences-train/"
    frame = cv2.imread(wdir+name+'-%0*d.bmp'%(3,im_begin))
    img = frame
    mask = cv2.imread(wdir+name+'-%0*d.png'%(3,im_begin))
    mask = (mask == 255)
    x_list, y_list, _ = mask.nonzero()
    y, x = (y_list.max()+y_list.min())/2, (x_list.max()+x_list.min())/2
    sq_size = min(y_list.max()-y_list.min(), x_list.max()-x_list.min()) * 0.5
    print(frame.shape)
    print("x,y=", x, y)
    print("square size=", sq_size)

    height , width , layers =  frame.shape
    video = cv2.VideoWriter(name+'.mp4',cv2.VideoWriter_fourcc(*"mp4v"),20,(width,height))

    print("MAIN:",frame.shape)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pf = ParticleFilter(x,y,frame,mask=mask[:,:,0], n_particles=1000,square_size=sq_size,
                        window_size=(frame.shape[0],frame.shape[1]), dt=0.2)
    alpha = 0.5

    orig = np.array(img)
    p1 = (int(y-sq_size),int(x-sq_size))
    p2 = (int(y+sq_size),int(x+sq_size))
    cv2.rectangle(img,p1,p2,(0,0,255),thickness=5)
    cv2.addWeighted(orig, alpha, img, 1 - alpha,0, img)
    cv2.imshow('frame',img)
    video.write(img)
    im = im_begin + 1
    while os.path.exists(wdir+name+'-%0*d.bmp'%(3,im)):     
        #cv2.waitKey(1000)   
        frame = cv2.imread(wdir+name+'-%0*d.bmp'%(3,im))
        orig = np.array(frame)
        img = frame
        #cv2.circle(img, (int(y),int(x)), 1, (0,255,0),thickness=100) 
        #norm_factor = 255.0/np.sum(frame,axis=2)[:,:,np.newaxis]
        #frame = frame*norm_factor
        frame = cv2.convertScaleAbs(frame)
        frame = cv2.blur(frame,(5,5))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        x,y,sq_size,distrib,distrib_control = pf.next_state(frame)
        p1 = (int(y-sq_size),int(x-sq_size))
        p2 = (int(y+sq_size),int(x+sq_size))
        
        # before resampling
        for (x2,y2,scale2) in distrib_control:
            x2 = int(x2)
            y2 = int(y2)
            cv2.circle(img, (y2,x2), 1, (255,0,0),thickness=10) 
        # after resampling
        for (x1,y1,scale1) in distrib:
        	x1 = int(x1)
        	y1 = int(y1)
        	cv2.circle(img, (y1,x1), 1, (0,0,255),thickness=10) 
        	
        cv2.circle(img, (y,x), 1, (0,255,0),thickness=10) 
        cv2.rectangle(img,p1,p2,(0,255,0),thickness=5)

        cv2.addWeighted(orig, alpha, img, 1 - alpha,0, img)   
        create_legend(img,(40,40),(40,20))

        cv2.imshow('frame',img)
        video.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break      
        im += 1    
    video.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
