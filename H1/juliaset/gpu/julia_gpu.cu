#include "utilities.h"
#include "julia_gpu.h"


__host__ __device__ float julia(int x, int y, cuComplex c, int w, int h) {
    const float scale = 1.5;
    float jx = scale * (float)(x - w/2)/(h/2);
    float jy = scale * (float)(y - h/2)/(h/2);

    cuComplex a(jx, jy);

    int i;
    for (i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 4)
            return (float)i / 200;
    }

    return 0;
}

__global__ void kernel(unsigned char* ptr, cuComplex c, JuliaSetColor color, int w, int h) {
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // now calculate the value at that position
    float juliaValue = julia(x, y, c, w, h);
    // calculate color for this value
    RGB rgb = color.getColor(juliaValue);

    ptr[offset+0*w*h] = (unsigned char)(rgb.b * 255);
    ptr[offset+1*w*h] = (unsigned char)(rgb.g * 255);
    ptr[offset+2*w*h] = (unsigned char)(rgb.r * 255);
}

/**
 * Main cardioid of Mandelbrod Set
 * @param theta
 * @return
 */
cuComplex main_cardioid(float theta) {
    float cos_value = cos(theta);
    float sin_value = sin(theta);
    float r = (2 * (1-cos_value) * cos_value + 1) / 4;
    float i = (2 * (1-cos_value) * sin_value) / 4;
    return {r, i};
}

/**
 * Second circle of Mandelbrod Set
 * @param theta
 * @return
 */
cuComplex second_circle(float theta) {
    float cos_value = cos(theta);
    float sin_value = sin(theta);
    float r = cos_value / 4 - 1;
    float i = sin_value / 4;
    return {r, i};
}

/**
 * Callback function for glut
 * @param devPtr
 * @param datablock
 * @param tick
 */
void render(unsigned char* ptr, unsigned char* devPtr, float state, int w, int h) {
    dim3    grid(w, h);
    auto c = main_cardioid(state * 2 * PI);
    JuliaSetColor color(0.3, 0.87, 0.9);
    kernel<<<grid,1>>>(devPtr, c, color, w, h);
    cudaMemcpy(ptr, devPtr, sizeof(unsigned char) * 3 * w * h, cudaMemcpyDeviceToHost);
}

void gpuMemAlloc(unsigned char **p, size_t size) {
    cudaMalloc(p, size);
}

void gpuMemFree(unsigned char *p) {
    cudaFree(p);
}