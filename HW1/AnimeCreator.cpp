//
// Created by zhuoc on 2021/11/12.
//

#include "AnimeCreator.h"

using namespace std::chrono_literals;

void AnimeCreator::play(cv::VideoWriter &vw) {
    for (auto anime: animes) {
        for (int j = 0; j < anime->nFrame(); j++) {
            anime->draw(canvas, j);
            vw.write(canvas.data());
            cv::imshow(name, canvas.data());
            int key = cv::waitKey(delayTime(fps));
            keyboardCallback(key);
        }
    }
    cv::waitKey();
}

void AnimeCreator::keyboardCallback(int key) {
    if (key == 32) {
        if (!pause) {
            pause = true;
            keyboardCallback(cv::waitKey());
        } else {
            pause = false;
        }
    }
}