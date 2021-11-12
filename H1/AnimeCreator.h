//
// Created by zhuoc on 2021/11/12.
//

#ifndef CV_COURSE_ANIMECREATOR_H
#define CV_COURSE_ANIMECREATOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>

typedef std::function<void(int)> CVKeyboardCallback;

class Anime {
protected:
    int frameNum = 0;
public:
    virtual ~Anime() = default;
    virtual void draw(cv::Mat &canvas, int frameCnt) = 0;
    int nFrame() const {
        return frameNum;
    }
};

class AnimeCreator {
private:
    std::mutex mtx_cv;
    std::condition_variable renderCV;
    std::string name;
    cv::Mat canvas, canvas_base;
    cv::Size frameSize;
    int frameNum = 0;
    int fps = 0;
    int frame_p = 0;
    std::vector<Anime*> animes;
    std::atomic_bool finish;

public:
    AnimeCreator(int fps, cv::Size frameSize, const std::string &name="") {
        this->fps = fps;
        this->frameSize = frameSize;
        this->name = name;
        canvas_base = cv::Mat(frameSize, CV_8SC3, cv::Scalar(255, 255, 255));
        canvas = canvas_base.clone();
    }
    ~AnimeCreator() {
        for (auto &anime: animes) {
            delete anime;
        }
    }
    void addAnime(Anime *anime) {
        animes.push_back(anime);
        frameNum += anime->nFrame();
    }
    void play(cv::VideoWriter &vw, CVKeyboardCallback keyboardCallback = nullptr, cv::MouseCallback mouseCallback = nullptr);
    int nFrame() const {
        return frameNum;
    }
    void clear() {
        canvas = canvas_base.clone();
    }
};

inline int delayTime(int fps) {
    return int(1000. / fps);
}


#endif //CV_COURSE_ANIMECREATOR_H
