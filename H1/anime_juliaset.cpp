//
// Created by zhuoc on 2021/11/15.
//

#include "anime_juliaset.h"

#include <cmath>

#define PI acos(-1)

cv::Point mainCardioid(double theta, cv::Size size) {
    double cos_value = cos(theta);
    double sin_value = sin(theta);
    double x = (2 * (1-cos_value) * cos_value + 1) / 4;
    double y = (2 * (1-cos_value) * sin_value) / 4;

    int xx = size.width / 2 + x * size.height / 2;
    int yy = size.height / 2 + y * size.height / 2;
    return {xx, yy};
}

void JuliaSetAnime::draw(Canvas &canvas, int frameCnt) {
    static cv::Point lastPoint;
    if (frameCnt < 5 * ac.FPS()) {
        double i = frameCnt / (5. * ac.FPS());
        auto point = mainCardioid(i * 2 * PI, size);
        if (frameCnt == 0) {
            lastPoint = point;
            canvas.clear();
        }
        cv::line(canvas.data(), lastPoint, point, cv::Scalar(0, 0, 0));
        lastPoint = point;
        return;
    }
    frameCnt -= 5 * ac.FPS();

    if (frameCnt < 1 * ac.FPS()) {
        double i = frameCnt / (1. * ac.FPS());
        int gv = 255 * (1 - i);
        std::string text = "The main cardioid of Mandelbrot set";
        int baseline;
        auto textSize = cv::getTextSize(text, 0, 1, 1, &baseline);
        cv::Point origin(canvas.data().cols / 2 - textSize.width / 2, 50);
        cv::putText(canvas.data(), text, origin, 0, 1, cv::Scalar(gv, gv, gv));
        return;
    }
    frameCnt -= 1 * ac.FPS();

    if (frameCnt < 3 * ac.FPS()) {
        double i = frameCnt / (1. * ac.FPS());
        if (i > 1)
            i = 1;
        int gv = 255 * (1 - i);
        std::string text = "Now let's see its corresponding Julia set";
        int baseline;
        auto textSize = cv::getTextSize(text, 0, 0.8, 1, &baseline);
        cv::Point origin(canvas.data().cols / 2 - textSize.width / 2, 430);
        cv::putText(canvas.data(), text, origin, 0, 0.8, cv::Scalar(gv, gv, gv));
        return;
    }
    frameCnt -= 3 * ac.FPS();

    static auto img = cv::Mat(canvas.size(), canvas.type(), cv::Scalar(255, 255, 255));
    if (frameCnt < 0.5 * ac.FPS()) {
        canvas.clear();
        canvas = render(0);

        double i = frameCnt / (0.5 * ac.FPS());
        double r = 0.2 + (1 - i) * (1 - 0.2);
        auto s = size * r;
        auto rect = cv::Rect(0, size.height - s.height, s.width, s.height);

        img(rect).copyTo(canvas.data()(rect));
        return;
    }
    frameCnt -= 0.5 * ac.FPS();

    if (frameCnt <= 20 * ac.FPS()) {
        canvas.clear();
        double i = frameCnt / (20. * ac.FPS());
        canvas = render(i);

        double r = 0.2;
        auto s = size * r;
        auto point = mainCardioid(i * 2 * PI, s);
        point.y += size.height - s.height;
        if (frameCnt == 0) {
            lastPoint = point;
        }
        cv::line(img, lastPoint, point, cv::Scalar(0, 0, 0));
        lastPoint = point;

        auto rect = cv::Rect(0, size.height - s.height, s.width, s.height);
        img(rect).copyTo(canvas.data()(rect));
        return;
    }
    frameCnt -= 20 * ac.FPS() + 1;

    double i = frameCnt / (3. * ac.FPS() - 1);
    canvas = canvas.data() * (1 - i);
}

cv::Mat JuliaSetAnime::render(float state) {
#ifdef H1_JULIA_USE_GPU
    ::render(ptr, devPtr, state, size.width, size.height);
#else
    ::render(ptr, state, size.width, size.height);
#endif
    auto b = cv::Mat(size, CV_8UC1, ptr);
    auto g = cv::Mat(size, CV_8UC1, ptr + size.width * size.height);
    auto r = cv::Mat(size, CV_8UC1, ptr + 2 * size.width * size.height);

    std::vector<cv::Mat> channels;
    channels.push_back(b);
    channels.push_back(g);
    channels.push_back(r);
    cv::Mat img;
    cv::merge(channels, img);
    return img;
}
