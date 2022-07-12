#include "pch.h"
#include "CInitializer.h"

std::mt19937 CInitializer::mersenne = std::mt19937{ CInitializer::rd() };

void CInitializer::NormalInitialize(const unsigned int& size, float* data)
{
	std::normal_distribution<float> dist{ 0.0, 10.0 };
	for (unsigned int idx = 0; idx < size; ++idx)
	{
		data[idx] = dist(CInitializer::mersenne);
	}
}

void CInitializer::HeInitialize(const unsigned int& size, float* data, const unsigned int& numIn)
{
	std::normal_distribution<float> dist{ 0.0, std::sqrt(2.0f / static_cast<float>(numIn)) };
	for (unsigned int idx = 0; idx < size; ++idx)
	{
		data[idx] = dist(CInitializer::mersenne);
	}
}

void CInitializer::XavierInitialize(const unsigned int& size, float* data, const unsigned int& numIn, const unsigned int& numOut)
{
	std::normal_distribution<float> dist{ 0.0, std::sqrt(2.0f / (static_cast<float>(numIn) + static_cast<float>(numOut))) };
	for (unsigned int idx = 0; idx < size; ++idx)
	{
		data[idx] = dist(CInitializer::mersenne);
	}
}
