//
// Created by zhuoc on 2021/11/15.
//

#ifndef CV_COURSE_JULIA_GPU_H
#define CV_COURSE_JULIA_GPU_H

#include <cuda.h>

struct cuComplex {
    float   r;
    float   i;
    __host__ __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    __host__ __device__ float magnitude2() const {
        return r * r + i * i;
    }
    __host__ __device__ cuComplex operator*(const cuComplex& a) const {
        return {r*a.r - i*a.i, i*a.r + r*a.i};
    }
    __host__ __device__ cuComplex operator+(const cuComplex& a) const {
        return {r+a.r, i+a.i};
    }
};

__host__ __device__ float julia(int x, int y, cuComplex c, int w, int h);
__global__ void kernel(unsigned char* ptr, cuComplex c, JuliaSetColor color, int w, int h);
cuComplex main_cardioid(float theta);
cuComplex second_circle(float theta);
void render(unsigned char* ptr, unsigned char* devPtr, float state, int w, int h);
void render(unsigned char* ptr, float state, int w, int h);

void gpuMemAlloc(unsigned char **p, size_t size);

void gpuMemFree(unsigned char *p);

#endif //CV_COURSE_JULIA_GPU_H
