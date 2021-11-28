//
// Created by zhuoc on 2021/11/15.
//

#ifndef CV_COURSE_ANIME_JULIASET_H
#define CV_COURSE_ANIME_JULIASET_H

#include "AnimeCreator.h"

#ifdef H1_JULIA_USE_GPU
extern void render(unsigned char* ptr, unsigned char* devPtr, float state, int w, int h);
extern void gpuMemAlloc(unsigned char **p, size_t size);
extern void gpuMemFree(unsigned char *p);
#else
extern void render(unsigned char* ptr, float state, int w, int h);
#endif

cv::Point mainCardioid(double theta, cv::Size size);

class JuliaSetAnime: public Anime {
private:
    cv::Size size;
    double sec = 32.5;
    unsigned char *ptr = nullptr;
#ifdef H1_JULIA_USE_GPU
    unsigned char *devPtr = nullptr;
#endif

public:
    explicit JuliaSetAnime(AnimeCreator &ac): Anime(ac) {
        frameNum = int(ac.FPS() * sec);
        size = ac.frameSize();
        int w = size.width;
        int h = size.height;
        ptr = new unsigned char[3*w*h];
#ifdef H1_JULIA_USE_GPU
        gpuMemAlloc(&devPtr, sizeof(unsigned char) * 3 * w * h);
#endif
    }
    ~JuliaSetAnime() override {
        delete ptr;
#ifdef H1_JULIA_USE_GPU
        gpuMemFree(devPtr);
#endif
    }
    void draw(Canvas &canvas, int frameCnt) override;
    cv::Mat render(float state);
};


#endif //CV_COURSE_ANIME_JULIASET_H
