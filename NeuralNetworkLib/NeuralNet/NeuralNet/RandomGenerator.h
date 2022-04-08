#pragma once
#include <random>
class RandomGenerator
{
private:
	std::random_device rd;
	std::mt19937 gen;


public:
	RandomGenerator() : rd(random_device{}), gen(std::mt19937{ rd() }) {}

	std::vector<double> getNormalDistVector(const int& size, const double& mean, const double sigma = 1.0);
};

