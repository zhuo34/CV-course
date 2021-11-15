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


inline cv::Size operator* (cv::Size a, double r) {
    return {int(a.width * r), int(a.height * r)};
}

class Canvas {
private:
    cv::Mat canvas, base;
public:
    Canvas() = default;
    Canvas(cv::Size size, int type, const cv::Scalar& s) {
        base = cv::Mat(size, type, s);
        canvas = base.clone();
    }
    void clear() {
        canvas = base.clone();
    }
    Canvas& operator=(const cv::Mat &m) {
        canvas = m;
        return *this;
    }
    cv::Mat& data() {
        return canvas;
    }
    cv::Size size() const {
        return canvas.size();
    }
    int type() const {
        return canvas.type();
    }
};

class AnimeCreator;

class Anime {
protected:
    AnimeCreator &ac;
    int frameNum = 0;
public:
    explicit Anime(AnimeCreator &ac): ac(ac) {}
    virtual ~Anime() = default;
    virtual void draw(Canvas &canvas, int frameCnt) {}
    int nFrame() const {
        return frameNum;
    }
};

class AnimeCreator {
public:
    std::string name;
    Canvas canvas;
    cv::Size size;
    int frameNum = 0;
    int fps = 0;
    int frame_p = 0;
    std::vector<Anime*> animes;

    bool pause = false;
    void keyboardCallback(int key);

public:
    AnimeCreator(int fps, cv::Size frameSize, const std::string &name="") {
        this->fps = fps;
        this->size = frameSize;
        this->name = name;
        canvas = Canvas(frameSize, CV_8UC3, cv::Scalar(255, 255, 255));
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
    void play(cv::VideoWriter &vw);
    int nFrame() const {
        return frameNum;
    }
    int FPS() const {
        return fps;
    }
    cv::Size frameSize() {
        return size;
    }
};

inline int delayTime(int fps) {
    return int(1000. / fps);
}


#endif //CV_COURSE_ANIMECREATOR_H
