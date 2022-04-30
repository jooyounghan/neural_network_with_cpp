#pragma once
#include "IncludePart.h"

class CRandomGenerator
{
private:
	std::random_device rd;
	std::mt19937 gen;

public:
	CRandomGenerator() : rd{}, gen(std::mt19937{ rd() }) {}

public:
	double* getNormalDistVector(const int& size, const double& mean, const double& sigma = 1.0);

	template<typename T>
	void ShuffleVector(std::vector<T>& inputVector, const uint32& seed);
};

template<typename T>
void CRandomGenerator::ShuffleVector(std::vector<T>& inputVector, const uint32& seed)
{
	srand(seed);
	std::random_shuffle(inputVector.begin(), inputVector.end());
}
