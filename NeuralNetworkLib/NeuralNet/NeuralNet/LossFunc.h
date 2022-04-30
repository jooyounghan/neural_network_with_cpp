#pragma once
#include "Matrix.h"

class CLossFunc
{
public:
	std::mutex mtx;	

public:
	virtual double GetResult(CMatrix* input, CMatrix* label) = 0;
	virtual void CalcResult(double& refDouble, CMatrix* input, CMatrix* label) = 0;
	/*virtual CMatrix* GetDeriviate(CMatrix* input) = 0;*/
protected:
	double ResultParallel(CMatrix* input, CMatrix* label, std::function<double(double&, double&)> func);
	double ResultSerial(CMatrix* input, CMatrix* label, std::function<double(double&, double&)> func);
	void ResultParallel(double& refDouble, CMatrix* input, CMatrix* label, std::function<double(double&, double&)> func);
	void ResultSerial(double& refDouble, CMatrix* input, CMatrix* label, std::function<double(double&, double&)> func);
};

class CSumation: public CLossFunc
{
public:
	double GetResult(CMatrix* input, CMatrix* label);
	void CalcResult(double& refDouble, CMatrix* input, CMatrix* label);
	CMatrix* GetDeriviate(CMatrix* input, CMatrix* label);
};

class CSoftmax : public CLossFunc
{
public:
	double GetResult(CMatrix* input);
	void CalcResult(double& refDouble, CMatrix* input, CMatrix* label);
	CMatrix* GetDeriviate(CMatrix* input, CMatrix* label);
};


