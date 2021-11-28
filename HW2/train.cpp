#include "FaceLoader.h"

#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cmdline.h"

using namespace std;

int main(int argc, char **argv) {
    cmdline::parser parser;

    parser.add<double>("energy", 'e', "energy ratio", false, 0.95);
    parser.add<string>("model_file", 'o', "model filename", true);
    parser.parse_check(argc, argv);

    double energy = parser.get<double>("energy");
    string model_file = parser.get<string>("model_file");

    string dataset_dir = "att_faces";

    FaceLoader fl(dataset_dir);
    Mat res = fl.train(energy, model_file);
    imshow("Eigenfaces", res);
    waitKey();
    res *= 255;
    res.convertTo(res, CV_8UC3);
    imwrite("train.png", res);

    return 0;
}