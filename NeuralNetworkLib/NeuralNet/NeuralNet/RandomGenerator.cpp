#include "pch.h"
#include "RandomGenerator.h"

double* CRandomGenerator::getNormalDistVector(const int& size, const double& mean, const double sigma)
{
	std::normal_distribution<double> normalDist{ mean, sigma };
	double* result = new double[size];
	for (int i = 0; i < size; ++i)
	{
		result[i] = normalDist(gen);
	}
	return result;
}
