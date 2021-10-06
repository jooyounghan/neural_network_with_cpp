#pragma once
#include <random>
#include <future>
#include "constant.h"

#define ERR		1E-6

void asyncInitialize(float* node, const int& start, const int& end, std::mt19937& mt, std::normal_distribution<float>& dist);
void asyncMatMul(float* node, float* node_in, float* w, const int& N, const int& K, const int& M, const int& start, const int& end);
bool asyncIsSame(float* node, float* node_in, const int& start, const int& end);
void asyncRelu(float* node, float* node_in, const int& start, const int& end);
void asyncTranspose(float* node, float* node_in, const int& N, const int& M, const int& start, const int& end);
void asyncSubtract(float* node, float* node_from, float* node_sub, const int& start, const int& end);