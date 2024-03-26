'''
MIT License

Copyright (c) 2024 Abhishek Sriram

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
from os.path import exists


class CannyOperator:
    def __init__(self, img_fs: str) -> None:
        '''
        Load image file and create a numpy array
        The image data is then loaded into the numpy array
        '''
        self.__img_fs = img_fs
        file_exists = exists(img_fs)  # check if filepath exists
        if (file_exists):
            img = Image.open(img_fs)  # load image using Pillow
            self.__img_np = np.array(img)  # load image into a numpy array
        else:
            print("Error: '{}' File does not exist".format(img_fs))
            sys.exit(-1)
            
    # grayscale
    #! TODO: use formula
    def grayscale(self):
        '''
        Convert image (RGB) color to grayscale
        Assuming that the original image has 3 channels: Red, Green and Blue
        Take average of the RGB channel. The numpy has a shape of 3, the third one has 3 channels: colors
        These 3 channels can be used from the 2nd axis in the numpy array
        
        Alternate:
        For 3 channels, use the formula: I = 0.2989 * R + 0.5870 * G + 0.1140 * B
        '''
        self.__grayscale_np = np.mean(self.__img_np, axis=2, dtype=np.uint8)
        # self.__grayscale_np = np.dot(self.__img_np[..., :3], [0.1140, 0.5870, 0.2989]) # for 3 channels, convert into 1 channel -> intensity
        # grayscale_img = Image.fromarray(self.__grayscale_np)
        # grayscale_img.show()
        return self
    
    def gaussian_kernel(self, size, sigma):
        '''
        Generate a Gaussian kernel of a given size and standard deviation
        Return the normalized gaussian kernel
        '''
        kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-size//2)**2 + (y-size//2)**2) / (2*sigma**2)), (size, size))
        return kernel / np.sum(kernel) # normalizing
    
    def gaussian_blur(self, size, sigma=2.5):
        '''
        Reduce noise in the image by applying Gaussian blur
        Smoothens the output image
        '''
        kernel = self.gaussian_kernel(size, sigma)
        self.__blurred_np = convolve2d(self.__grayscale_np, kernel, mode='same', boundary='symm', fillvalue=0)
        # blurred_img = Image.fromarray(self.__blurred_np)
        # blurred_img.show()
        return self
        
    def gradient_approximation(self):
        '''
        Applying two convolution filter to the grayscale image and approximate the magnitude of the gradient
        Horizontal filter (Hf) := [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Vertical filter (Vf) := transpose(Hf)
        '''
        horizontal_filter = np.array([[-1, 0, 1],
                                      [-2, 0, 2],
                                      [-1, 0, 1]]) # sobel kernel x-axis
        vertical_filter = np.transpose(horizontal_filter) # sobel kernel y-axis
        horizontal_gradient = convolve2d(
            self.__blurred_np, horizontal_filter, mode='same', boundary='symm', fillvalue=0)
        vertical_gradient = convolve2d(
            self.__blurred_np, vertical_filter, mode='same', boundary='symm', fillvalue=0)
        self.__gradient_magnitude = np.sqrt(
            horizontal_gradient ** 2 + vertical_gradient ** 2)  # magnitude of the gradient
        self.__gradient_direction = np.arctan2(vertical_gradient, horizontal_gradient)  # magnitude of the gradient
        # magnitude_img = Image.fromarray(self.__gradient_magnitude)
        # magnitude_img.show()
        return self
    
    def non_max_supression(self):
        '''
        Check for local maximums in the direction of gradient (in the neighboring pixels) and only preserve them
        Else, we supress the pixel value
        '''
        self.__suppressed_img = np.zeros_like(self.__gradient_magnitude)
        for x in range(self.__gradient_magnitude.shape[0]):
            for y in range(self.__gradient_magnitude.shape[1]):
                angle = self.__gradient_direction[x, y]
                # convert angle to radians if it's in degrees
                if angle > np.pi:
                    angle = angle * np.pi / 180.0
                
                # define neighboring pixel indices based on gradient direction
                if (0 <= angle < np.pi / 8) or (15 * np.pi / 8 <= angle <= 2 * np.pi):
                    neighbor1_i, neighbor1_j = x, y + 1
                    neighbor2_i, neighbor2_j = x, y - 1
                elif (np.pi / 8 <= angle < 3 * np.pi / 8):
                    neighbor1_i, neighbor1_j = x - 1, y + 1
                    neighbor2_i, neighbor2_j = x + 1, y - 1
                elif (3 * np.pi / 8 <= angle < 5 * np.pi / 8):
                    neighbor1_i, neighbor1_j = x - 1, y
                    neighbor2_i, neighbor2_j = x + 1, y
                elif (5 * np.pi / 8 <= angle < 7 * np.pi / 8):
                    neighbor1_i, neighbor1_j = x - 1, y - 1
                    neighbor2_i, neighbor2_j = x + 1, y + 1
                else:
                    neighbor1_i, neighbor1_j = x - 1, y
                    neighbor2_i, neighbor2_j = x + 1, y
                    
                # check if neighbor indices are within bounds
                neighbor1_i = max(0, min(neighbor1_i, self.__gradient_magnitude.shape[0] - 1))
                neighbor1_j = max(0, min(neighbor1_j, self.__gradient_magnitude.shape[1] - 1))
                neighbor2_i = max(0, min(neighbor2_i, self.__gradient_magnitude.shape[0] - 1))
                neighbor2_j = max(0, min(neighbor2_j, self.__gradient_magnitude.shape[1] - 1))
                
                # compare current pixel magnitude with its neighbors along gradient direction
                current_mag = self.__gradient_magnitude[x, y]
                neighbor1_mag = self.__gradient_magnitude[neighbor1_i, neighbor1_j]
                neighbor2_mag = self.__gradient_magnitude[neighbor2_i, neighbor2_j]

                # perform supression
                if (current_mag >= neighbor1_mag) and (current_mag >= neighbor2_mag):
                    self.__suppressed_img[x, y] = current_mag
                else:
                    self.__suppressed_img[x, y] = 0
        # suppressed_img = Image.fromarray(self.__suppressed_img)
        # suppressed_img.show()
        return self
    
    def double_thresholding(self, low_threshold_ratio=0.1, high_threshold_ratio=0.3):
        '''
        Categorize pixels into: Strong, weak, and non-edges
        Apply threshold and preserve the strong and weak edges
        Reject the values that are below the weak threshold (non-edges)
        '''
        h_threshold = np.max(self.__gradient_magnitude) * high_threshold_ratio
        l_threshold = h_threshold * low_threshold_ratio
        
        # store edge status (strong, weak, or non-edge)
        strong_edges = (self.__gradient_magnitude >= h_threshold)
        weak_edges = (self.__gradient_magnitude >= l_threshold) & (self.__gradient_magnitude < h_threshold)
        
        # connect weak edges to strong edges
        self.__connected_edges = np.zeros_like(self.__gradient_magnitude)
        self.__connected_edges[strong_edges] = 255

        # apply edge connectivity to weak edges
        for x in range(self.__gradient_magnitude.shape[0]):
            for y in range(self.__gradient_magnitude.shape[1]):
                if weak_edges[x, y]:
                    # check if any of the 8-connected neighbors are strong edges
                    if (strong_edges[x - 1:x + 2, y - 1:y + 2].any()):
                        self.__connected_edges[x, y] = 255
        # connnected_edges_img = Image.fromarray(self.__connected_edges)
        # connnected_edges_img.show()
        return self
    
    def hysteresis(self, weak_pixel_intensity=50, strong_pixel_intensity=255):
        '''
        Connect weak edges to strong edges and reject isolated weak edges
        Use the intensity values to reject the weak edges
        '''
        self.__canny_img = self.__connected_edges.copy()
        
        # coordinates of wek edges in the image
        weak_edges_x, weak_edges_y = np.where(self.__canny_img == weak_pixel_intensity)
        for x, y in zip(weak_edges_x, weak_edges_y):
            if np.any(self.__connected_edges[x - 1:x + 2, y - 1:y + 2] == strong_pixel_intensity):
                self.__canny_img[x, y] = strong_pixel_intensity # connnect to strong edges
            else:
                self.__canny_img[x, y] = 0
        # final_img = Image.fromarray(self.__canny_img)
        # final_img.show()
        return self
    
    def canny_cv2(self, tmin=50, tmax=255):
        '''
        Canny edge detection implemented using OpenCV
        '''
        grayscale = cv2.imread(self.__img_fs, cv2.IMREAD_GRAYSCALE)
        self.__canny_cv2 = cv2.Canny(grayscale, threshold1=tmin, threshold2=tmax)
        return self
    
    def comparisons(self):
        '''
        Implemented:
        1) Grayscale vs Gradieent Approx. vs Supressed vs Double Threshold
        2) Canny edge detector vs Cv2's Canny edge detector
        '''
        # image = plt.imread(self.__img_fs)
        grayscale_image = Image.fromarray(self.__grayscale_np)
        gradient_approx_image = Image.fromarray(self.__gradient_magnitude)
        suppressed_img = Image.fromarray(self.__suppressed_img)
        connnected_edges_img = Image.fromarray(self.__connected_edges)
        final_image = Image.fromarray(self.__canny_img)
        canny_image = Image.fromarray(self.__canny_cv2)

        # Display original and blurred images
        _, axarr = plt.subplots(2, 2)

        axarr[0, 0].title.set_text('Grayscale Image')
        axarr[0, 0].imshow(grayscale_image, cmap='gray',
                           vmin=0, vmax=255)  # quantize to grayscale
        axarr[0, 0].axis('on')

        axarr[0, 1].title.set_text('Gradient Approx.')
        axarr[0, 1].imshow(gradient_approx_image)
        axarr[0, 1].axis('on')

        axarr[1, 0].title.set_text('Suppressed Image')
        axarr[1, 0].imshow(suppressed_img, cmap='gray', vmin=0, vmax=255)
        axarr[1, 0].axis('on')
        
        axarr[1, 1].title.set_text('Double thresholded Image')
        axarr[1, 1].imshow(connnected_edges_img, cmap='gray', vmin=0, vmax=255)
        axarr[1, 1].axis('on')

        plt.show()
        plt.close()

        plt.subplot(1, 2, 1)
        plt.title('Final Image')
        plt.imshow(final_image, cmap='gray', vmin=0, vmax=255)
        axarr[0, 0].axis('on')

        plt.subplot(1, 2, 2)
        plt.title('Canny Cv2 Image')
        plt.imshow(canny_image, cmap='gray', vmin=0, vmax=255)
        plt.axis('on')

        plt.show()
        plt.close()
        final_image.convert('RGB').save("Canny_impl.png") # final image
        canny_image.convert('RGB').save("Canny_cv2.png") # cv2's canny image

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        for i in range(1, len(sys.argv)):
            obj = CannyOperator(sys.argv[i])
            obj.grayscale()\
                .gaussian_blur(size=3, sigma=10)\
                .gradient_approximation()\
                .non_max_supression()\
                .double_thresholding(low_threshold_ratio=0.1, high_threshold_ratio=0.3)\
                .hysteresis(weak_pixel_intensity=50)\
                .canny_cv2(tmin=50)\
                .comparisons()
    else:
        print("Error: image file directory not mentioned")