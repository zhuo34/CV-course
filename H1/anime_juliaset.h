//
// Created by zhuoc on 2021/11/15.
//

#ifndef CV_COURSE_ANIME_JULIASET_H
#define CV_COURSE_ANIME_JULIASET_H

#include "AnimeCreator.h"


extern void render(unsigned char* ptr, unsigned char* devPtr, float state, int w, int h);
extern void render(unsigned char* ptr, float state, int w, int h);
extern void gpuMemAlloc(unsigned char **p, size_t size);
extern void gpuMemFree(unsigned char *p);

cv::Point mainCardioid(double theta, cv::Size size);

class JuliaSetAnime: public Anime {
private:
    cv::Size size;
    unsigned char *ptr = nullptr;
    unsigned char *devPtr = nullptr;
    double sec = 32.5;

public:
    explicit JuliaSetAnime(AnimeCreator &ac): Anime(ac) {
        frameNum = int(ac.FPS() * sec);
        size = ac.frameSize();
        int w = size.width;
        int h = size.height;
        ptr = new unsigned char[3*w*h];
        gpuMemAlloc(&devPtr, sizeof(unsigned char) * 3 * w * h);
    }
    ~JuliaSetAnime() override {
        delete ptr;
        gpuMemFree(devPtr);
    }
    void draw(Canvas &canvas, int frameCnt) override;
    cv::Mat render(float state);
};


#endif //CV_COURSE_ANIME_JULIASET_H
