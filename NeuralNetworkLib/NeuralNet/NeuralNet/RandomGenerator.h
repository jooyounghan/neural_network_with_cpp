#pragma once
#include <random>
class CRandomGenerator
{
private:
	std::random_device rd;
	std::mt19937 gen;


public:
	CRandomGenerator() : rd{}, gen(std::mt19937{ rd() }) {}

public:
	double* getNormalDistVector(const int& size, const double& mean, const double& sigma = 1.0);
};

