#include "opencv2/opencv.hpp"
#include <DarkHelp.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// int main()
// {
//     const std::string config_file = "/media/dan/UBUNTU1/LAB/deep_sort_yolo/cfg/small_yolov4.cfg";
//     const std::string weights_file = "/media/dan/UBUNTU1/LAB/deep_sort_yolo/weights/yolov4.weights";
//     const std::string names_file = "/media/dan/UBUNTU1/LAB/deep_sort_yolo/data/coco.names";

//     // load the neural network (config, weights, and class names)
//     DarkHelp darkhelp(config_file, weights_file, names_file);

//     // run the neural network predictions on an image, and save the output
//     darkhelp.predict("/media/dan/UBUNTU1/LAB/C++/test_opencv/test.jpg");
//     cv::Mat image = darkhelp.annotate();

//     // use standard OpenCV calls to write the annotated image to disk
//     cv::imwrite("output.jpg", image, {cv::IMWRITE_JPEG_QUALITY, 65});
//     cv::imshow("ok", image);
//     cv::waitKey(0);
//     return 0;
// }

int main()
{
    const std::string config_file   = "/media/dan/UBUNTU1/LAB/deep_sort_yolo/cfg/small_yolov4.cfg";
    const std::string weights_file  = "/media/dan/UBUNTU1/LAB/deep_sort_yolo/weights/yolov4.weights";
    const std::string names_file    = "/media/dan/UBUNTU1/LAB/deep_sort_yolo/data/coco.names";

    // load the neural network (config, weights, and class names)
    DarkHelp darkhelp(config_file, weights_file, names_file);

    // Create a VideoCapture object and use camera to capture the video
    VideoCapture cap(0);

    // Check if camera opened successfully
    if (!cap.isOpened())
    {
        cout << "Error opening video stream" << endl;
        return -1;
    }

    while (1)
    {
        Mat frame;

        // Capture frame-by-frame
        cap >> frame;
        
        // If the frame is empty, break immediately
        if (frame.empty())
            break;
        const auto results = darkhelp.predict(frame);

        for (const auto & detection : results)
        {
            std::cout
                << "-------------------------"                  << std::endl
                << "  name:   " << detection.name               << std::endl
                << "  rect:   "
                << "x="         << detection.rect.x             << " "
                << "y="         << detection.rect.y             << " "
                << "w="         << detection.rect.width         << " "
                << "h="         << detection.rect.height        << std::endl
                << "  class:  " << detection.best_class         << std::endl
                << "  prob.:  " << detection.best_probability   << std::endl;
        }

        cv::Mat image = darkhelp.annotate();

        // Display the resulting frame
        imshow("Output", image);

        // Press  ESC on keyboard to  exit
        char c = (char)waitKey(1);
        if (c == 27)
            break;
    }

    // When everything done, release the video capture and write object
    cap.release();

    // Closes all the windows
    destroyAllWindows();
    return 0;
}