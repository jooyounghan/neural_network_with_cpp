#pragma once

class CMatrix;

class CInitializer
{
private:
	static std::random_device rd;
	static std::mt19937 mersenne;

public:
	enum INITTYPE
	{
		NORMAL,
		HE,
		XAVIER
	};

public:
	static void NormalInitialize(CMatrix& mat);

	static void HeInitialize(CMatrix& mat, const UINT& numIn);

	static void XavierInitialize(CMatrix& mat, const UINT& numIn, const UINT& numOut);
};

