//
// Created by zhuoc on 2021/11/26.
//

#ifndef CV_COURSE_FACELOADER_H
#define CV_COURSE_FACELOADER_H

#include <string>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define get1DIdx(i, j, m) (((i) * (m)) + (j))


//class Face {
//public:
//    Mat img, vec;
//};

//inline Mat face2vec(Mat img) {
//
//}
//
//Mat vec2img(Mat vec);

class FaceMask {
private:
    int width, height;
    int x0, x1;
    int y;
public:
    FaceMask(int width, int height, int x0, int x1, int y) : width(width), height(height), x0(x0), x1(x1), y(y) {}
    Mat clip(Mat face, const Point& l, const Point& r) const;
    Mat face2vec(const Mat& face) const;
    Mat vec2face(const Mat& vec) const;
    Size getVecSize() const {
        return {width * height, 1};
    }
};

class FaceLoader {
private:
    string path;

    vector<Mat> faces, faces_vec;
    Mat x_train, x_mean;

    FaceMask fm{60, 72, 18, 42, 28};
    void load();

public:
    explicit FaceLoader(string path) : path(std::move(path)) {
        load();
    }
    static Mat preprocess(Mat img);
    Mat getFace(const Mat& img, const Point& l, const Point& r);
    Mat train(double energy_rate, const string &filename);
    int predict(const Mat &A, const Mat& img);
    int nearest2class(int nearest) {
        return nearest / 5;
    }
    Mat test(const string &model, const Mat& img);
    void test_all(const string &model);
    Mat reconstruct(const string &model, const Mat& img);
};


#endif //CV_COURSE_FACELOADER_H
