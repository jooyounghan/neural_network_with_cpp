#pragma once
#include "Matrix.h"

class CActFunc
{
public:
	virtual CMatrix* GetResult(CMatrix* input) = 0;
	virtual void CalcResult(CMatrix* refMat, CMatrix* input) = 0;
	virtual CMatrix* GetDeriviate(CMatrix* input) = 0;
	virtual void CalcDeriviate(CMatrix* refMat, CMatrix* input) = 0;
protected:
	CMatrix* ResultParallel(CMatrix* input, std::function<double(double&)> func);
	CMatrix* ResultSerial(CMatrix* input, std::function<double(double&)> func);
	static void ResultParallel(CMatrix* refMat, CMatrix* input, std::function<double(double&)> func);
	static void ResultSerial(CMatrix* refMat, CMatrix* input, std::function<double(double&)> func);
};

class CSigmoid : public CActFunc
{
public:
	CMatrix* GetResult(CMatrix* input);
	void CalcResult(CMatrix* refMat, CMatrix* input);
	CMatrix* GetDeriviate(CMatrix* input);
	void CalcDeriviate(CMatrix* refMat, CMatrix* input);
};

class CRelu : public CActFunc
{
public:
	CMatrix* GetResult(CMatrix* input);
	void CalcResult(CMatrix* refMat, CMatrix* input);
	CMatrix* GetDeriviate(CMatrix* input);
	void CalcDeriviate(CMatrix* refMat, CMatrix* input);
};

class CIdentity : public CActFunc
{
public:
	CMatrix* GetResult(CMatrix* input);
	void CalcResult(CMatrix* refMat, CMatrix* input);
	CMatrix* GetDeriviate(CMatrix* input);
	void CalcDeriviate(CMatrix* refMat, CMatrix* input);
};