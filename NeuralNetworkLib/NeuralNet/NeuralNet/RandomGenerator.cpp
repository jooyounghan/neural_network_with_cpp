#include "pch.h"
#include "RandomGenerator.h"

std::vector<double> RandomGenerator::getNormalDistVector(double, , const double& mean)
{
	std::normal_distribution<double> normalDist{ mean, sigma };
	std::vector<double> result;
	result.resize(size);
	for (int i = 0; i < size; ++i)
	{
		result[i] = normalDist(gen);
	}
	return result;
}
