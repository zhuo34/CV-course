//
// Created by zhuoc on 2021/11/15.
//

#ifndef CV_COURSE_JULIA_CPU_H
#define CV_COURSE_JULIA_CPU_H

struct cuComplex {
    float   r;
    float   i;
    cuComplex( float a, float b ) : r(a), i(b)  {}
    float magnitude2() const {
        return r * r + i * i;
    }
    cuComplex operator*(const cuComplex& a) const {
        return {r*a.r - i*a.i, i*a.r + r*a.i};
    }
    cuComplex operator+(const cuComplex& a) const {
        return {r+a.r, i+a.i};
    }
};

float julia(int x, int y, cuComplex c, int w, int h);
cuComplex main_cardioid(float theta);
cuComplex second_circle(float theta);
void render(unsigned char* ptr, float state, int w, int h);

#endif //CV_COURSE_JULIA_CPU_H
