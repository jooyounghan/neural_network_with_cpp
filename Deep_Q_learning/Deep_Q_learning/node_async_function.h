#pragma once
#include <random>
#include <future>
#include "constant.h"

#define ERR		1E-6

void asyncInitialize(float* node, const int& start, const int& end, std::mt19937& mt, std::normal_distribution<float>& dist);
void asyncMatMul(float* node, float* node_in, float* w, const int& N, const int& K, const int& M, const int& start, const int& end);
bool asyncIsSame(float* node, float* node_in, const int& start, const int& end);
void asyncRelu(float* node, float* node_in, const int& start, const int& end);

void asyncSquare(float* node, const int& start, const int& end);
void asyncGetSquare(float* node, float* node_in, const int& start, const int& end);
void asyncGetSqrt(float* node, float* node_in, const int& start, const int& end);
void asyncGetFraction(float* node, float* node_in, const int& start, const int& end);

void asyncGetNodeSubtract(float* node, float* node_from, float* node_sub, const int& start, const int& end);
void asyncGetNodeMul(float* node, float* node_from, float* node_mul, const int& start, const int& end);
void asycnGetConstantMul(float* node, float* node_in, const float& num, const int& start, const int& end);
void asyncTranspose(float* node, float* node_in, const int& N, const int& M, const int& start, const int& end);

void asyncNodeSubtract(float* node, float* node_from, float* node_sub, const int& start, const int& end);

void asyncCopy(float* node, float* node_in, const int& start, const int& end);

void asyncConstantMul(float* node, const float& num, const int& start, const int& end);
void asyncConstantMat(float* node, float* node_in, const float& num, const int& start, const int& end);
void asyncConstantAdd(float* node, float* node_in, const float& num, const int& start, const int& end);

void asyncNodeElementWiseAdd(float* node, float* node_in_1, float* node_in_2, const int& start, const int& end);