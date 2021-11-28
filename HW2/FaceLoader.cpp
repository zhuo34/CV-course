//
// Created by zhuoc on 2021/11/26.
//

#include "FaceLoader.h"

#include <fstream>

Mat FaceMask::clip(Mat face, const Point& l, const Point& r) const {
    int yy = (l.y + r.y) / 2;
    double dis = r.x - l.x;
    double ratio = (x1 - x0) / dis;
    resize(face, face, Size(face.cols * ratio, face.rows * ratio));
    int lx = l.x * ratio;
    int rx = r.x * ratio;
    int eyeY = yy * ratio;
    int dis_new = rx - lx;
    int lm = (width - dis_new) / 2;
    int tm = y;
    auto rect = cv::Rect(lx - lm, eyeY - tm, width, height);
    return face(rect);
}

Mat FaceMask::vec2face(const Mat& vec) const {
    return vec.reshape(0, height);
}

Mat FaceMask::face2vec(const Mat& face) const {
    return face.reshape(0, 1);
}

Mat FaceLoader::preprocess(Mat img) {
    cvtColor(img, img, COLOR_BGR2GRAY);
    equalizeHist(img, img);
    img.convertTo(img, CV_32F);
    img /= 255;
    return img;
}

void FaceLoader::load() {
    for (int i = 0; i < 41; ++i) {
        string dir = path + "/s" + to_string(i + 1);
        string eye_file = dir + "/eye.txt";
        ifstream f(eye_file);
        string postfix = ".pgm";
        if (i == 40)
            postfix = ".jpg";
        for (int j = 0; j < 10; ++j) {
            string face_file = dir + "/" + to_string(j + 1) + postfix;
            Mat face = imread(face_file);
            int lx, ly, rx, ry;
            f >> lx >> ly >> rx >> ry;
            face = getFace(face, Point(lx, ly), Point(rx, ry));
            imwrite(dir + "/" + to_string(j + 1) + ".png", face);
            if (j < 5) {
                face = preprocess(face);
                faces.push_back(face);
                faces_vec.push_back(fm.face2vec(face));
            }
        }
        f.close();
    }
    vconcat(faces_vec, x_train);
    x_mean = Mat(fm.getVecSize(), CV_32F);
    for (int i = 0; i < x_train.cols; i++) {
        x_mean.at<float>(0, i) = mean(x_train.col(i)).val[0];
    }
}

Mat FaceLoader::getFace(const Mat& img, const Point& l, const Point& r) {
    Point ll(l.x + img.cols/2, l.y + img.rows/2);
    Point rr(r.x + img.cols/2, r.y + img.rows/2);
    Mat face;
    copyMakeBorder(img, face, img.rows/2, img.rows/2, img.cols/2, img.cols/2, BORDER_REPLICATE);
    face = fm.clip(face, ll, rr);
    return face;
}

Mat FaceLoader::train(double energy_rate, const string &filename) {
    Mat x_train_new = x_train - repeat(x_mean, x_train.rows, 1);
    Mat sigma = x_train.t() * x_train / x_train.rows;

    Mat eigenValues, eigenVectors;
    cout << "Calculating eigenvalues ...";
    eigen(sigma, eigenValues, eigenVectors);
//    FileStorage all_file("models/model_all", FileStorage::READ);
//    all_file["eigenValues"] >> eigenValues;
//    all_file["eigenVectors"] >> eigenVectors;
//    all_file.release();
    cout << " done." << endl;

    float sum = 0;
    float target = cv::sum(eigenValues).val[0] * energy_rate;
    int cnt = 0;
    for (int i = 0; i < x_train.cols; i++) {
        sum += eigenValues.at<float>(i, 0);
        cnt ++;
        if (sum >= target)
            break;
    }
    cout << "number of PC: " << cnt << endl;

    FileStorage file(filename, FileStorage::WRITE);
    file << "eigenValues" << eigenValues;
    file << "eigenVectors" << eigenVectors.rowRange(0, cnt);
    file.release();

    vector<Mat> outputs;
    outputs.push_back(fm.vec2face(x_mean));
    for (int i = 0; i < 11; i++) {
        Mat vec;
        normalize(eigenVectors.row(i), vec, 0, 1, NORM_MINMAX);
        outputs.push_back(fm.vec2face(vec));
    }
    vector<Mat> outputRows;
    for (int i = 0; i < 3; i++) {
        vector<Mat> t;
        int begin = i * 4;
        for (int j = 0; j < 4; j++) {
            t.push_back(outputs[begin + j]);
        }
        Mat tt;
        hconcat(t, tt);
        outputRows.push_back(tt);
    }
    Mat outputImg;
    vconcat(outputRows, outputImg);

    return outputImg;
}

int FaceLoader::predict(const Mat &A, const Mat& face) {
    Mat res = A * fm.face2vec(face).t();
    Mat all = A * x_train.t();
    double min = 1000;
    int nearest = -1;
    for (int i = 0; i < x_train.rows; i++) {
        double dis = norm(res, all.col(i));
        if (dis < min) {
            nearest = i;
            min = dis;
        }
    }
    return nearest;
}

Mat FaceLoader::test(const string &model, const Mat& img) {
    Mat face = preprocess(img);
    Mat eigenValues, eigenVectors;
    FileStorage file(model, FileStorage::READ);
    file["eigenValues"] >> eigenValues;
    file["eigenVectors"] >> eigenVectors;
    file.release();
    cout << "number of PC(s): " << eigenVectors.rows << endl;

    int nearest = predict(eigenVectors, face);
    int cls= nearest2class(nearest);
    cout << "nearest image id: " << nearest << endl;
    cout << "class id: " << cls << endl;
    string text = "s" + to_string(cls + 1);
    if (cls == 40)
        text = "me";

    Mat face2 = fm.vec2face(x_train.row(nearest));
    resize(face, face, Size(face.cols * 2, face.rows * 2));
    resize(face2, face2, Size(face2.cols * 2, face2.rows * 2));
    cvtColor(face, face, COLOR_GRAY2BGR);
    cvtColor(face2, face2, COLOR_GRAY2BGR);
    putText(face, text, Point(0, 20), 0, 0.6, Scalar(0, 255, 0), 2);
    putText(face2, "Best fit", Point(0, 20), 0, 0.6, Scalar(0, 0, 255), 2);

    Mat out_img;
    hconcat(vector<Mat>{face, face2}, out_img);

    return out_img;
}

void FaceLoader::test_all(const string &model) {
    Mat eigenValues, eigenVectors;
    FileStorage file(model, FileStorage::READ);
    file["eigenValues"] >> eigenValues;
    file["eigenVectors"] >> eigenVectors;
    file.release();
    string res_file = "results.txt";
    ofstream f(res_file, ios::out);
    for (int k = 0; k < eigenVectors.rows; k++) {
        cout << "testing " << k+1 << " PC(s) ..." << endl;
        Mat A = eigenVectors.rowRange(0, k+1);
        int right_cnt = 0;
        int wrong_cnt = 0;
        for (int i = 0; i < 41; ++i) {
            string dir = path + "/s" + to_string(i + 1);
            for (int j = 5; j < 10; ++j) {
                string face_file = dir + "/" + to_string(j + 1) + ".png";
                Mat face = imread(face_file);
                face = preprocess(face);
                int cls = nearest2class(predict(A, face));
                if (cls == i)
                    right_cnt++;
                else
                    wrong_cnt++;
            }
        }
        f << k+1 << " " << right_cnt << " " << wrong_cnt << endl;
        cout << k+1 << " " << right_cnt << " " << wrong_cnt << endl;
    }
    f.close();
}

Mat FaceLoader::reconstruct(const string &model, const Mat &img) {
    Mat face = preprocess(img);
    Mat vec = fm.face2vec(face);

    Mat eigenValues, eigenVectors;
    FileStorage file(model, FileStorage::READ);
    file["eigenValues"] >> eigenValues;
    file["eigenVectors"] >> eigenVectors;
    file.release();

    vector<int> nPC{10, 25, 50, 100};
    vector<Mat> rec_imgs;
    for (int i : nPC) {
        if (i > eigenVectors.rows)
            i = eigenVectors.rows;
        Mat A = eigenVectors.rowRange(0, i);
        Mat rec = (vec - x_mean) * A.t() * A + x_mean;
        Mat rec_img = fm.vec2face(rec);
        resize(rec_img, rec_img, Size(rec_img.cols * 2, rec_img.rows * 2));
        cvtColor(rec_img, rec_img, COLOR_GRAY2BGR);
        putText(rec_img, to_string(i) + " PCs", Point(0, 20), 0, 0.6, Scalar(0, 255, 0), 2);
        rec_imgs.push_back(rec_img);
    }
    resize(face, face, Size(face.cols * 2, face.rows * 2));
    cvtColor(face, face, COLOR_GRAY2BGR);
    putText(face, "origin", Point(0, 20), 0, 0.6, Scalar(0, 255, 0), 2);
    rec_imgs.push_back(face);

    Mat output_img;
    hconcat(rec_imgs, output_img);
    return output_img;
}
