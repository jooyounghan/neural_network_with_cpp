#include "pch.h"
#include "CInitializer.h"
#include "CMatrix.h"

std::random_device CInitializer::rd{};
std::mt19937 CInitializer::mersenne = std::mt19937{ CInitializer::rd() };

void CInitializer::NormalInitialize(CMatrix& mat)
{
	std::normal_distribution<double> dist{ 0.0, 1.0 };
	for (UINT idx = 0; idx < mat.size; ++idx)
	{
		mat.data[idx] = dist(CInitializer::mersenne);
	}
}

void CInitializer::HeInitialize(CMatrix& mat, const UINT& numIn)
{
	std::normal_distribution<double> dist{ 0.0, std::sqrt(2.0f / static_cast<double>(numIn)) };
	for (UINT idx = 0; idx < mat.size; ++idx)
	{
		mat.data[idx] = dist(CInitializer::mersenne);
	}
}

void CInitializer::XavierInitialize(CMatrix& mat, const UINT& numIn, const UINT& numOut)
{
	std::normal_distribution<double> dist{ 0.0, std::sqrt(2.0f / (static_cast<double>(numIn) + static_cast<double>(numOut))) };
	for (UINT idx = 0; idx < mat.size; ++idx)
	{
		mat.data[idx] = dist(CInitializer::mersenne);
	}
}
