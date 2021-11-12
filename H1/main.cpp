#include <iostream>
#include <opencv2/opencv.hpp>

#include "AnimeCreator.h"

class Anime1 : public Anime {
private:
    cv::Mat img;
    bool first = true;
public:
    explicit Anime1(int nf) {
        img = cv::imread(R"(D:\Projects\CV-course\H1\imgs\a.jpg)");
        frameNum = nf;
    }
    void draw(cv::Mat &canvas, int frameCnt) override {
        if (first) {
            auto frameSize = canvas.size();
            cv::resize(img, img, frameSize);
            first = false;
        }
        canvas = img;
    }
};

class Anime2 : public Anime {
public:
    explicit Anime2(int nf) {
        frameNum = nf;
    }
    void draw(cv::Mat &canvas, int frameCnt) override {

        cv::putText(canvas, std::to_string(frameCnt), cv::Point(100, 100), cv::FONT_HERSHEY_PLAIN, 14, cv::Scalar(0, 0, 0));
    }
};

int main() {
    int fps = 30;
    auto frameSize = cv::Size(600, 400);

    cv::VideoWriter vw;
    vw.open("video.avi", cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), fps, frameSize);

    AnimeCreator ac(fps, frameSize, "CV: Homework 1");
    ac.addAnime(new Anime1(fps));
    ac.addAnime(new Anime2(fps));

    ac.play(vw);

    cv::destroyAllWindows();
    return 0;
}
