#pragma once
#include <random>
class RandomGenerator
{
private:
	std::random_device rd;
	std::mt19937 gen;


public:
	RandomGenerator() : rd(random_device{}), gen(std::mt19937{ rd() }) {}

	std::vector<double> getNormalDistVector(const int& size, const double& mean, const double sigma = 1.0)
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
};

