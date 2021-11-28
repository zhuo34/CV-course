//
// Created by zhuoc on 2021/11/13.
//

#include "anime_head.h"

void Head::draw(Canvas &canvas, int frameCnt) {
    if (frameCnt < 2 * ac.FPS()) {
        canvas.clear();
        double i = frameCnt / (2. * ac.FPS());

        int w = canvas.size().width;
        int h = canvas.size().height;
        int centerX = w / 2 - logo_img.cols / 2;
        int centerY = h / 2 - logo_img.rows / 2;
        int finalX = 10;
        int finalY = 0;
        double alpha = 1 - i;
        int x = (centerX - finalX) * alpha + finalX;
        int y = (centerY - finalY) * alpha + finalY;
        int size = (initSize - finalSize) * alpha + finalSize;
        cv::Mat logo;
        cv::resize(logo_img, logo, cv::Size(size, size));
        logo.copyTo(canvas.data()(cv::Rect(x, y, logo.cols, logo.rows)));
        return;
    }
    frameCnt -= 2 * ac.FPS();

    if (frameCnt < 1 * ac.FPS()) {
        double i = frameCnt / (1. * ac.FPS());
        auto img = cv::Mat(me_img * i);
        img.copyTo(canvas.data()(cv::Rect(100, 100, img.cols, img.rows)));
        int gv = 255 * (1 - i);
        cv::putText(canvas.data(), "Name: Zhuo Chen", cv::Point(110+img.cols, 120), 0, 0.8, cv::Scalar(gv, gv, gv));
        cv::putText(canvas.data(), "StuId: 12121103", cv::Point(110+img.cols, 150), 0, 0.8, cv::Scalar(gv, gv, gv));
        cv::putText(canvas.data(), "Computer Science", cv::Point(110+img.cols, 180), 0, 0.8, cv::Scalar(gv, gv, gv));
        return;
    }
//    frameCnt -= 1 * ac.FPS();
//    frameCnt -= 1 * ac.FPS();
}
