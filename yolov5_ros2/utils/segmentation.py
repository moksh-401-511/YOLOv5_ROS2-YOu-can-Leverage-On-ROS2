import numpy as np
import cv2

class_color = {'person':(100,50,50),'cell phone':(200,50,50),'car':(50,50,150),'chair':(80,70,100),'sofa':(200,70,200),'refrigrator':(20,70,20),'tv':(150,70,150),'table':(80,80,150),'bed':(150,70,20)}

def image_segmentation(img,xyxy,classes):
    '''img: 'Depth' image matrix (greyscale image)
       xyxy: list of starting and end cordinates of rectangle (x,y,x,y)
       classes: list of classes corresponding to xyxy rectangles'''
    #xyxy = np.array(xyxy.cpu())
    #classes = np.array(classes)
    img = img*255
    
    # sorting based on size of rectangle 
    idx_xyxy = [idx for _,idx in sorted(zip(xyxy,range(len(xyxy))), key=lambda x: (int(x[0][0])-int(x[0][2]))*(int(x[0][1])-int(x[0][3])), reverse=True)]
    sorted_xyxy = [xyxy[idx] for idx in idx_xyxy]
    classes = [classes[idx] for idx in idx_xyxy]
    
    # RGB final image
    x,y,z = img.shape[0], img.shape[1], 3
    final_img = np.zeros((x,y,z))
    
    for xy_xy,cls_conf in zip(sorted_xyxy,classes):
        cls = cls_conf.split()[0]
        try:
            confidence = float(cls_conf.split()[1])
        except:
            confidence = 0
        
        #print(xy_xy)
        # extracting image from rectangle of detected object 
        croped_img = img[int(xy_xy[1]):int(xy_xy[3]),int(xy_xy[0]):int(xy_xy[2])]
        #show_image(croped_img)
        per10, per20, per30, per35, per40, per45, per50, per75, per80, per85, per90, per100 = np.percentile(croped_img,[10,20,30,35,40,45,50,75,80,85,90,100])
        #print("percentiles:",per10,per40,per80,per100,np.min(croped_img),np.max(croped_img))
        if (abs(per90-per10)) > 0:
            #print("doing segmentation")
            # object segmentation
            seg_img = transform(croped_img,per45,per80)
            #show_image(seg_img,"seg img")
            #print("seg_img shape",seg_img.shape,"&",final_img[xy_xy[1]:xy_xy[3],xy_xy[0]:xy_xy[2]].shape)
            if cls in class_color.keys():
                color = class_color[cls]
            else:
                color = (200,200,200)
            if confidence > 0.50:
                final_img[int(xy_xy[1]):int(xy_xy[3]),int(xy_xy[0]):int(xy_xy[2]),:][seg_img>5] =  color
        
    #print(final_img.shape)
    cv2.imwrite('segmented image yolov5.jpg', final_img)
    #show_image(final_img,"segmented image")
    return final_img/255
    
def obj_seg(img):
    per10, per20, per30, per35, per40, per45, per50, per75, per80, per85, per90, per100 = np.percentile(img,[10,20,30,35,40,45,50,75,80,85,90,100])
    #output = convolve2D(img, kernel3, 1, 1, per40, per80)
    output = transform(img,per40,per80)
    return output
    
def transform(x, lb, ub):
    x = np.array(x,dtype=np.float128)
    lower_bound = 1/(1 + np.exp(-(x-lb)))
    upper_bound = 1/(1 + np.exp(x-ub))
    return np.array(x*upper_bound*lower_bound,dtype=np.float64)
    
def show_image(img,name="test image"):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
