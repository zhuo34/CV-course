//
// Created by zhuoc on 2021/11/15.
//

#ifndef CV_COURSE_ANIME_TAIL_H
#define CV_COURSE_ANIME_TAIL_H

#include <utility>

#include "AnimeCreator.h"


class Text {
public:
    std::string text;
    int fontFace = 0;
    double fontScale = 1;
    int thickness = 1;
    cv::Scalar color = {255, 255, 255};
//    int baseline = 0;
//    cv::Size size;

    Text(std::string text) : text(std::move(text)) {}
    Text(std::string text, int fontFace, double fontScale, int thickness)
            : text(std::move(text)), fontFace(fontFace), fontScale(fontScale), thickness(thickness) {}
    Text(std::string text, int fontFace, double  fontScale, int thickness, cv::Scalar &color)
        : text(std::move(text)), fontFace(fontFace), fontScale(fontScale), thickness(thickness), color(color) {}

    cv::Size getSize() const {
        int baseline;
        return cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
    }
};

class TextBox {
private:
    std::vector<Text> ts;
    std::vector<int> margins;
public:
    int height = 0;
    void add(const Text& s, int margin = 0) {
        ts.push_back(s);
        margins.push_back(margin);
        height += margin + s.getSize().height;
    }
    bool draw(cv::Mat &canvas, int y);
    void setColor(cv::Scalar color) {
        for (auto &t: ts) {
            t.color = color;
        }
    }
    Text &operator[](int index) {
        return ts[index];
    }
};

class Tail: public Anime {
private:
    std::vector<TextBox> s;
    TextBox thankTb;
    int margin = 100;
    int speed = 1;
    double sigma = 15000;
    bool freeze = false;
    int freezeCnt = 0;
public:
    explicit Tail(AnimeCreator &ac) : Anime(ac) {
        frameNum = 15 * ac.FPS();
        s = std::vector<TextBox>(3);
        s[0].add(Text("Powered by", 0, 0.6, 2));
        s[0].add(Text("C++ / CMake / OpenCV / CUDA"), 10);

        s[1].add(Text("Inspired by", 0, 0.6, 2));
        s[1].add(Text("Julia Set"), 10);

        s[2].add(Text("Author", 0, 0.6, 2));
        s[2].add(Text("Zhuo Chen"), 10);

        thankTb.add(Text("Thanks"));
    }
    void draw(Canvas &canvas, int frameCnt) override;
};

#endif //CV_COURSE_ANIME_TAIL_H
