import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal as spsig, optimize as spopt

class ImageStacker():
    def __init__(self):
        self.data_dir = ""
        self.save_dir = ""

        self.ref_image = 0        
        self.interp = 1
        self.mode = "RGB"

        self.raw_images = []

    def open_images(self):
        valid_file_extentions = ".png", ".tif" 
        self.data_files = os.listdir(self.data_dir) 

        # Loop through files in directory and add images
        images = []
        for data_file in self.data_files:
            
            file_extention = data_file[data_file.rfind((".")):]
            if file_extention not in valid_file_extentions:
                continue
            
            data_path = os.path.join(self.data_dir, data_file)
            img = Image.open(data_path)
            img = img.convert( self.mode)
            
            array = np.array(img)
            images += [array]

        # Convert to np array, resize images
        images = np.array(images)
        images = [cv2.resize(image, None, fx=self.interp, fy=self.interp, interpolation = cv2.INTER_CUBIC)
                  for n, image
                  in enumerate(images)]
        images = np.array(images)

        # Extract image properties
        self.num_images, self.height, self.length, self.num_channels = images.shape
        self.raw_images = images

        print("Images opened!")

    def peaks(self):
        self.peaks = [np.unravel_index(np.argmax(image), image.shape) for image in self.raw_images[:,:,:,0]]

    def crosscor(self):
        self.peaks = []
        for image in self.raw_images[:,:,:,0]:
            correlation = spsig.correlate(image.astype(float), self.raw_images[self.ref_image,:,:,0].astype(float), mode="full", method= "fft")
            self.peaks += [np.unravel_index(np.argmax(correlation), correlation.shape)]

    def circle(self):
        self.peaks = np.ndarray(shape= (0,2), dtype= int)
        for image in self.raw_images[:,:,:,0]:
            
            # CONSIDER BLURRING THE IMAGE IF IT IS LOOKING AT THE WRONG FEATURE
            threshold, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU) #image, threshold, max, type (int)
            contours, hierarchy = cv2.findContours(mask, 1, 2) #binary image, mode, method
            contour = contours[-1] # I think the last one refers to the biggest shape.
            
            (x,y), cov = spopt.leastsq(self.radius_residuals, x0= [np.mean(contour[:,:,0]), np.mean(contour[:,:,1])],kwargs=contour)
            
            self.peaks = np.concatenate((self.peaks, [[round(y),round(x)]]), axis= 0)

    def crescent(self):
        self.peaks = np.ndarray(shape= (0,2), dtype= int)
        for image in self.raw_images[:,:,:,0]:
            
            # CONSIDER BLURRING THE IMAGE IF IT IS LOOKING AT THE WRONG FEATURE
            threshold, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU) #image, threshold, max, type (int)
            contours, hierarchy = cv2.findContours(mask, 1, 2) #binary image, mode, method
            contour = contours[-1] # I think the last one refers to the biggest shape.
            
            (x,y),radius = cv2.minEnclosingCircle(contour)
            
            self.peaks = np.concatenate((self.peaks, [[round( y),round(x)]]), axis= 0)

    def radius_residuals(args, contour):
        x0, y0 = args
        r = np.sqrt((contour[:,:,0]-x0)**2 + (contour[:,:,1]-y0)**2)
        return r.flatten() - r.mean()
    
    def shift_and_add(self):
        y_enlargement, x_enlargement = np.max(self.peaks, axis= 0) -np.min(self.peaks, axis= 0)

        shifts = np.max(self.peaks, axis= 0) - self.peaks

        # Loop through images, shift em, stack em
        shift_and_added = np.zeros((self.height +y_enlargement, self.length +x_enlargement, self.num_channels))
        weights = np.zeros((self.height +y_enlargement, self.length +x_enlargement))
        for image, shift in zip(self.raw_images, shifts):
            shift_slice = np.s_[shift[0]:self.height +shift[0], shift[1]:self.length +shift[1]]
            weights[shift_slice] += np.ones_like(image[:,:,0])
            shift_and_added[shift_slice] += image
        shift_and_added /= weights[:,:,None]
        shift_and_added[np.isnan(shift_and_added)] = 0.0

        processed = np.copy(shift_and_added)
        if processed.min() <= 0.0: processed -= processed.min()
        processed *= (2**8 -1) / processed.max()

        processed = processed.astype(np.uint8)

        self.final_image = Image.fromarray(processed, mode = self.mode)

        return self.final_image

