#include <iostream>
#include <opencv2/opencv.hpp>

#include "AnimeCreator.h"
#include "anime_head.h"
#include "anime_juliaset.h"
#include "anime_tail.h"


int main() {
    int fps = 60;
    auto frameSize = cv::Size(640, 480);

    cv::VideoWriter vw;
    vw.open("video.avi", cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), fps, frameSize);

    AnimeCreator ac(fps, frameSize, "CV: Homework 1");
    ac.addAnime(new Head(ac));
    ac.addAnime(new JuliaSetAnime(ac));
    ac.addAnime(new Tail(ac));
    ac.play(vw);

    cv::destroyAllWindows();
    return 0;
}
