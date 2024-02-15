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


class SobelOperator:
    def __init__(self, img_fs: str):
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

    def grayscale(self):
        '''
        Convert image (RGB) color to grayscale
        Assuming that the original image has 3 channels: Red, Green and Blue
        Take average of the RGB channel. The numpy has a shape of 3, the third one has 3 channels: colors
        These 3 channels can be used from the 2nd axis in the numpy array
        '''
        self.__grayscale_np = np.mean(
            self.__img_np, axis=2, dtype=np.uint8)  # covert into grayscale using mean
        # self.grayscale_img = Image.fromarray(self.__grayscale_np)
        # self.grayscale_img.show()
        return self

    def gradient_approximation(self):
        '''
        Applying two convolution filter to the grayscale image and approximate the magnitude of the gradient
        Horizontal filter (Hf) := [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Vertical filter (Vf) := transpose(Hf)
        '''
        horizontal_filter = np.array([[-1, 0, 1],
                                      [-2, 0, 2],
                                      [-1, 0, 1]])
        vertical_filter = np.transpose(horizontal_filter)
        horizontal_gradient = convolve2d(
            self.__grayscale_np, horizontal_filter, mode='same', boundary='symm', fillvalue=0)
        vertical_gradient = convolve2d(
            self.__grayscale_np, vertical_filter, mode='same', boundary='symm', fillvalue=0)
        self.__gradient_magnitude = np.sqrt(
            horizontal_gradient ** 2 + vertical_gradient ** 2)  # magnitude of the gradient
        return self

    def thresholding(self, threshold: int = 100):
        '''
        A final procedure follows applying a threshold
        Pixels with gradient magnitude above the threshold are considered part of an edge, while those below are not
        '''
        self.__binary_edge_np = (self.__gradient_magnitude >
                                 threshold).astype(np.uint8) * 255
        return self

    def sobel_cv2(self):
        '''
        Sobel Operator with OpenCv2
        Compare against the implemented sobel operator
        '''
        img = cv2.imread(self.__img_fs)
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.__sobel_cv2 = cv2.Sobel(grayscale, cv2.CV_8U, 1, 0, ksize=3)
        return self

    def comparisons(self):
        '''
        Implemented Sobel operator vs Sobel Cv2
        '''
        image = plt.imread(self.__img_fs)
        grayscale_image = Image.fromarray(self.__grayscale_np)
        gradient_approx_image = Image.fromarray(self.__gradient_magnitude)
        final_image = Image.fromarray(self.__binary_edge_np)
        sobel_image = Image.fromarray(self.__sobel_cv2)

        # Display original and blurred images
        _, axarr = plt.subplots(2, 2)
        axarr[0, 0].title.set_text('Original Image')
        axarr[0, 0].imshow(image)
        axarr[0, 0].axis('on')

        axarr[0, 1].title.set_text('Grayscale Image')
        axarr[0, 1].imshow(grayscale_image, cmap='gray',
                           vmin=0, vmax=255)  # quantize to grayscale
        axarr[0, 1].axis('on')

        axarr[1, 0].title.set_text('Gradient Approx.')
        axarr[1, 0].imshow(gradient_approx_image)
        axarr[1, 0].axis('on')

        axarr[1, 1].title.set_text('Final Image')
        axarr[1, 1].imshow(final_image, cmap='gray', vmin=0, vmax=255)
        axarr[1, 1].axis('on')

        plt.show()
        plt.close()

        plt.subplot(1, 2, 1)
        plt.title('Final Image')
        plt.imshow(final_image, cmap='gray', vmin=0, vmax=255)
        axarr[0, 0].axis('on')

        plt.subplot(1, 2, 2)
        plt.title('Sobel Cv2 Image')
        plt.imshow(sobel_image, cmap='gray', vmin=0, vmax=255)
        plt.axis('on')

        plt.show()
        plt.close()
        final_image.save("Sobel_impl.png")
        sobel_image.save("Sobel_cv2.png")


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        for i in range(1, len(sys.argv)):
            obj = SobelOperator(sys.argv[i])
            obj.grayscale()\
                .gradient_approximation()\
                .thresholding(100)\
                .sobel_cv2()\
                .comparisons()
    else:
        print("Error: image file directory not mentioned")
