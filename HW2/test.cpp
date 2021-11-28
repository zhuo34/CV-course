#include "FaceLoader.h"

#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cmdline.h"

using namespace std;

int main(int argc, char **argv) {
    cmdline::parser parser;
    parser.add<string>("face_file", 'f', "test face filename", true);
    parser.add<string>("model_file", 'm', "model filename", true);
    parser.add("test_all", 'a', "test all", false, false);
    parser.parse_check(argc, argv);

    bool test_all = parser.get<bool>("test_all");
    string model_file = parser.get<string>("model_file");
    string face_file = parser.get<string>("face_file");

    string dataset_dir = "att_faces";
    FaceLoader fl(dataset_dir);

    Mat face = imread(face_file);
    Mat res = fl.test(model_file, face);
    imshow("Test", res);
    waitKey();
    res *= 255;
    res.convertTo(res, CV_8UC3);
    imwrite("test.png", res);

    if (test_all)
        fl.test_all(model_file);

    return 0;
}