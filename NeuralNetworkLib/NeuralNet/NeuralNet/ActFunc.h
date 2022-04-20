#pragma once
#include "Matrix.h"

class CActFunc
{
public:
	virtual CMatrix* GetResult(CMatrix* input) = 0;
	/*virtual CMatrix* GetDeriviate(CMatrix* input) = 0;*/

protected:
	CMatrix* ResultParallel(CMatrix* input, std::function<double(double&)> func);
	CMatrix* ResultSerial(CMatrix* input, std::function<double(double&)> func);
};


class Sigmoid : public CActFunc
{
public:
	CMatrix* GetResult(CMatrix* input);
	CMatrix* GetDeriviate(CMatrix* input);
};

class Relu : public CActFunc
{
public:
	CMatrix* GetResult(CMatrix* input);
	CMatrix* GetDeriviate(CMatrix* input);
};

class Identity : public CActFunc
{
public:
	CMatrix* GetResult(CMatrix* input);
	CMatrix* GetDeriviate(CMatrix* input);
};

//class Softmax : public CActFunc
//{
//public:
//	CMatrix* GetResult(CMatrix* input);
//	CMatrix* GetDeriviate(CMatrix* input);
//};