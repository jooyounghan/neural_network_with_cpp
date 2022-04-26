#pragma once
#include "Matrix.h"

class CActFunc
{
protected:
	virtual CMatrix* GetResult(CMatrix* input) = 0;
	virtual void CalcResult(CMatrix* refMat, CMatrix* input) = 0;
	/*virtual CMatrix* GetDeriviate(CMatrix* input) = 0;*/

private:
	CMatrix* ResultParallel(CMatrix* input, std::function<double(double&)> func);
	CMatrix* ResultSerial(CMatrix* input, std::function<double(double&)> func);
	void ResultParallel(CMatrix* refMat, CMatrix* input, std::function<double(double&)> func);
	void ResultSerial(CMatrix* refMat, CMatrix* input, std::function<double(double&)> func);
};


class Sigmoid : public CActFunc
{
public:
	CMatrix* GetResult(CMatrix* input);
	void CalcResult(CMatrix* refMat, CMatrix* input);
	CMatrix* GetDeriviate(CMatrix* input);
};

class Relu : public CActFunc
{
public:
	CMatrix* GetResult(CMatrix* input);
	void CalcResult(CMatrix* refMat, CMatrix* input);
	CMatrix* GetDeriviate(CMatrix* input);
};

class Identity : public CActFunc
{
public:
	CMatrix* GetResult(CMatrix* input);
	void CalcResult(CMatrix* refMat, CMatrix* input);
	CMatrix* GetDeriviate(CMatrix* input);
};

//class Softmax : public CActFunc
//{
//public:
//	CMatrix* GetResult(CMatrix* input);
//	CMatrix* GetDeriviate(CMatrix* input);
//};