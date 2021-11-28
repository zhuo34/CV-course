//
// Created by zhuoc on 2021/9/26.
//

#ifndef JULIA_UTILITIES_H
#define JULIA_UTILITIES_H

#include <cmath>

#define PI acos(-1.f)
#define clamp(x, a, b) (fmin(fmax((x), (a)), (b)))

struct RGB {
    float r, g, b;
};

struct HSV {
    float h, s, v;
    HSV() : h(0), s(0), v(0)  {}
    HSV(float _h, float _s, float _v) : h(_h), s(_s), v(_v)  {}
    HSV(const HSV &a) = default;

    RGB toRGB() const;
};

struct JuliaSetColor {
    RGB in, out;
    float juliaFactor = 0.1;
    float colorFactor = 20;

    JuliaSetColor(const HSV &c);
    JuliaSetColor(float h, float s, float v);
    RGB getColor(float v) const;
private:
    void set_color(const HSV &c);
    static float map_color(float v, float a, float b, float factor);
};

#endif //JULIA_UTILITIES_H
