#pragma once
#include <random>

class CInitializer
{
private:
	static std::random_device rd;
	static std::mt19937 mersenne;

public:
	static void NormalInitialize(const unsigned int& size, float* data);

	static void HeInitialize(const unsigned int& size, float* data, const unsigned int& numIn);


	static void XavierInitialize(const unsigned int& size, float* data, const unsigned int& numIn, const unsigned int& numOut);

};

