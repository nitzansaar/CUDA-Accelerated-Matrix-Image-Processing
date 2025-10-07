#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    // Read image (change "image.jpg" to your file)
    Mat img = imread("image.jpg", IMREAD_COLOR); // or IMREAD_GRAYSCALE

    // Check if image loaded successfully
    if (img.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    cout << "Image size: " << img.rows << " x " << img.cols << endl;

}