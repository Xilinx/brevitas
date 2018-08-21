#ifndef MATH_GENERICS_INC
#define MATH_GENERICS_INC

#include <cuda.h>

namespace math_generics {

    template <typename T>
    __device__ __forceinline__ T max(T a, T b) {
        static_assert(sizeof(T) != sizeof(T), "Function max hasn't implemented the specialized type.");
    }

    template <typename T>
    __device__ __forceinline__ T min(T a, T b) {
        static_assert(sizeof(T) != sizeof(T), "Function min hasn't implemented the specialized type.");
    }

    template <typename T>
    __device__ __forceinline__ T exp(T x) {
        static_assert(sizeof(T) != sizeof(T), "Function exp hasn't implemented the specialized type.");
    }

    template <typename T>
    __device__ __forceinline__ T tanh(T x) {
        static_assert(sizeof(T) != sizeof(T), "Function tanh hasn't implemented the specialized type.");
    }

    template <typename T>
    __device__ __forceinline__ T pow(T b, T x) {
        static_assert(sizeof(T) != sizeof(T), "Function pow hasn't implemented the specialized type.");
    }

     template <typename T>
    __device__ __forceinline__ T round(T x) {
        static_assert(sizeof(T) != sizeof(T), "Function round hasn't implemented the specialized type.");
    }


    template <>
    __device__ __forceinline__ float max(float a, float b) {
        return fmaxf(a, b);
    }

    template <>
    __device__ __forceinline__  double max(double a, double b) {
        return fmax(a, b);
    }

    template <> 
    __device__ __forceinline__ float min(float a, float b) {
        return fminf(a, b);
    }

    template <> 
    __device__ __forceinline__ double min(double a, double b) {
        return fmin(a, b);
    }

    template <> 
    __device__ __forceinline__ float exp(float x) {
        return expf(x);
    }

    template <> 
    __device__ __forceinline__ double exp(double x) {
        return exp(x);
    }

    template <> 
    __device__ __forceinline__ float tanh(float x) {
        return tanhf(x);
    }

    template <> 
    __device__ __forceinline__ double tanh(double x) {
        return tanh(x);
    }    

        template <> 
    __device__ __forceinline__ float pow(float b, float x) {
        return powf(b, x);
    }

    template <> 
    __device__ __forceinline__ double pow(double b, double x) {
        return pow(b, x);
    } 

    template <> 
    __device__ __forceinline__ float round(float x) {
        return rintf(x);
    }

    template <> 
    __device__ __forceinline__ double round(double x) {
        return rint(x);
    }  
}


#endif // MATH_GENERICS_INC
