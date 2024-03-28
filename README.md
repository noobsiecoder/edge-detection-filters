# Edge Detection Filters

This Python project focuses on implementing edge detection algorithms without using OpenCV library. It includes the Sobel and Canny edge detection technique.

## Links

1. [Canny Edge Detection: Explained and Compared with OpenCV in Python](https://medium.com/@abhisheksriram845/canny-edge-detection-explained-and-compared-with-opencv-in-python-57a161b4bd19)
2. [Seeing the World in Edges: An Insider’s Look at Sobel Detection](https://medium.com/@abhisheksriram845/seeing-the-world-in-edges-an-insiders-look-at-sobel-detection-fc118e3c5ea8)

## Implementation

1. Sobel Filter

- The Sobel filter is a gradient-based edge detection algorithm. It computes the gradient magnitude of an image, which highlights edges where the intensity changes rapidly.
- The Sobel operator applies two 3x3 convolution kernels to the image, one for detecting horizontal changes and the other for vertical changes.
- The algorithm to perform Sobel Filter is shown below

```code
    1. Convert the color image to grayscale by taking its mean from the values of the RGB channel in the image (assuming it is a color image)
    2. Apply a 3x3 convolution filter horizontally and vertically to the grayscale intensity values
    3. Calculate the gradient magnitude := sqrt( (hrz ** 2) + (vrt ** 2))
    4. Choose an appropriate value for thresholding, higher the value, lowers the chances of finding an edge (dull)
```

2. Canny Edge Detection

- The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images. 
- It was developed by John F. Canny in 1986. Canny also produced a computational theory of edge detection explaining why the technique works. 
- The process of applying Canny edge detection on an image:

```code
    1. Convert the color image to grayscale
    2. Apply Gaussian filter (kernel) to smooth the image in order to remove the noise
    3. Find the intensity gradients (magnitude and orientation) of the image
    4. Perform non-max supression by using the gradient magnitude in the gradient direction (check if the pixel is a local maxima)
    5. Find potential edges by categorizing edges: weak, strong and no-edges
    6. Track edges by hysteresis and connect the weak edges to strong edges in the image
```

## Result

1. Sobel Edge Detector
   ![comparison b/w sobel and custom impl](https://github.com/noobsiecoder/edge-detection-filters/blob/main/results/sobel/1/Figure_2.png?raw=true)

2. Canny Edge Detector
    ![comparison b/w canny and custom impl](https://github.com/noobsiecoder/edge-detection-filters/blob/main/results/canny/1/Figure_2.png?raw=true)

## Requirements

- Python 3.x
- Numpy
- Matplotlib
- SciPy

## Installation

- Ensure you have Python installed on your system. If not, download and install it from [here](https://www.python.org/downloads/).
- Install the required dependencies:

```bash
    pip3 install -r requirements.txt
```

## Usage/Examples

1. Clone or download this repository to your local machine.
2. Run the following command to execute the edge detection script:

```bash
    # edge_detection_file<str> := [sobel | canny]
    # img_file(s)<str| list> := [img_01, .., img_03]
    python3 src/<edge_detection_file>.py samples/<img_file(s)>.png
```

## Contributing

Contributions are always welcome! Feel free to submit pull requests or open issues for any improvements or additional features you'd like to see.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/noobsiecoder/edge-detection-filters/tree/main?tab=MIT-1-ov-file) file for details.

## References

- [Edge Detection](https://en.wikipedia.org/wiki/Edge_detection)
- [Sobel edge detection](https://en.wikipedia.org/wiki/Sobel_operator)
- [How to find Edges in Images: Sobel Operators & Full Implementation](https://www.youtube.com/watch?v=VL8PuOPjVjY)
- [Canny Edge Detection](https://en.wikipedia.org/wiki/Canny_edge_detector)
- [Canny Edge Detection Slides by Prof. Robert B. Fisher, School of Informatics, University of Edinburgh](https://homepages.inf.ed.ac.uk/rbf/AVINVERTED/STEREO/av5_edgesf.pdf)
- [Edges: The Canny Edge Detector](https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MARBLE/low/edges/canny.htm)
