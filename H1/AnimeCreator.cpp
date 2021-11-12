//
// Created by zhuoc on 2021/11/12.
//

#include "AnimeCreator.h"

#include <thread>
#include <mutex>

using namespace std::chrono_literals;

void AnimeCreator::play(cv::VideoWriter &vw, CVKeyboardCallback keyboardCallback, cv::MouseCallback onMouse) {
    if (onMouse) {
        cv::setMouseCallback(name, onMouse);
    }
    finish = false;
    std::thread render_th([this, &vw]() {
        for (auto anime: animes) {
            for (int j = 0; j < anime->nFrame(); j++) {
//                std::cerr << "render begin " << j << std::endl;
                clear();
                anime->draw(canvas, j);
                vw.write(canvas);
//                std::cerr << "render end" << std::endl;
                std::unique_lock<std::mutex> lck_cv(mtx_cv);
//                std::cerr << "render wait " << j + 1 << std::endl;
                renderCV.wait(lck_cv);
            }
        }
        finish = true;
    });
    int cnt = 0;
    while (true) {
//        std::cout << "display " << cnt++ << std::endl;
        cv::imshow(name, canvas);
        renderCV.notify_one();
        int key = cv::waitKey(delayTime(fps));
        if (keyboardCallback) {
            keyboardCallback(key);
        }
        if (finish)
            break;
    }

    render_th.join();
}