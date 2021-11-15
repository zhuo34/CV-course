//
// Created by zhuoc on 2021/11/15.
//

#include "anime_tail.h"

void Tail::draw(Canvas &canvas, int frameCnt) {
    canvas = cv::Mat(canvas.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    auto img = canvas.data();
    int cnt = 0;
    int initY = canvas.size().height;
    for (int i = 0; i < s.size(); i++) {
        int y = initY - frameCnt * speed;
        initY += margin + s[i].height;
        int x = y + s[i].height / 2 - img.rows / 2;
        int gv = exp(-x * x / sigma) * 255;
        s[i].setColor(cv::Scalar(gv, gv, gv));
        auto has = s[i].draw(canvas.data(), y);
        if (has)
            cnt ++;
    }

    int y = initY - frameCnt * speed;
    if (y > canvas.size().height)
        return;
    int h = thankTb.height;
    if (!freeze) {
        if (y + h / 2 <= img.rows/2) {
            freeze = true;
            freezeCnt = frameCnt;
        }
    }
    if (freeze) {
        y = initY - freezeCnt * speed;
    }
    int x = y + h / 2 - img.rows / 2;
    int gv = exp(-x * x / sigma) * 255;
    thankTb.setColor(cv::Scalar(gv, gv, gv));
    thankTb.draw(canvas.data(), y);
//    if (cnt == 0)
//        std::cout << frameCnt << std::endl;
}

bool TextBox::draw(cv::Mat &canvas, int y) {
    int cnt = 0;
    for (int i = 0; i < ts.size(); i++){
        cv::Size textSize = ts[i].getSize();
        if (y < -textSize.height || y > canvas.size().height)
            continue;
        y += margins[i] + textSize.height;
        cv::Point origin(canvas.cols / 2 - textSize.width / 2, y);
        cv::putText(canvas, ts[i].text, origin, ts[i].fontFace, ts[i].fontScale, ts[i].color, ts[i].thickness);
        cnt ++;
    }
    return cnt > 0;
}
