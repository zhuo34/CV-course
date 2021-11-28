//
// Created by zhuoc on 2021/11/13.
//

#ifndef CV_COURSE_ANIME_HEAD_H
#define CV_COURSE_ANIME_HEAD_H

#include "AnimeCreator.h"

class Head: public Anime {
private:
    cv::Mat logo_img, me_img;
    int initSize, finalSize;
public:
    explicit Head(AnimeCreator &ac) : Anime(ac) {
        frameNum = 4 * ac.FPS();
        logo_img = cv::imread("imgs/logo.jpg");
        me_img = cv::imread("imgs/me.jpg");
        cv::resize(me_img, me_img, me_img.size() * 0.5);
        initSize = 200;
        finalSize = 80;
    }
    void draw(Canvas &canvas, int frameCnt) override;
};


#endif //CV_COURSE_ANIME_HEAD_H
