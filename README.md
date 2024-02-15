# Edge Detection Filters

This Python project focuses on implementing edge detection algorithms without using OpenCV library. It includes the Sobel edge detection technique.

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

## Result

1. Sobel Edge Detector
   ![comparison b/w sobel and custom impl](https://github.com/noobsiecoder/edge-detection-filters/blob/main/results/sobel/1/Figure_2.png?raw=true)

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
    # edge_detection_file<str> := [sobel]
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
