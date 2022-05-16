#pragma once
#include "Matrix.h"

using ACTID = unsigned short;

class CActFunc
{
public:
	enum ActID
	{
		SIGMOID,
		RELU,
		IDENTITY
	};

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
	CMatrix* GetResult(CMatrix* input) override;
	void CalcResult(CMatrix* refMat, CMatrix* input) override;
	CMatrix* GetDeriviate(CMatrix* input) override;
	void CalcDeriviate(CMatrix* refMat, CMatrix* input) override;
};

class CRelu : public CActFunc
{
public:
	CMatrix* GetResult(CMatrix* input) override;
	void CalcResult(CMatrix* refMat, CMatrix* input) override;
	CMatrix* GetDeriviate(CMatrix* input) override;
	void CalcDeriviate(CMatrix* refMat, CMatrix* input) override;
};

class CIdentity : public CActFunc
{
public:
	CMatrix* GetResult(CMatrix* input) override;
	void CalcResult(CMatrix* refMat, CMatrix* input) override;
	CMatrix* GetDeriviate(CMatrix* input) override;
	void CalcDeriviate(CMatrix* refMat, CMatrix* input) override;
};