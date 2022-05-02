#pragma once
#include "Matrix.h"

class CLossFunc
{
protected:
	CLossFunc();
public:
	~CLossFunc() {}

public:
	virtual void GetResult(CMatrix* refMat, CMatrix* inputMat) = 0;
	virtual void GetLossGradient(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat) = 0;
	static void GetLoss(double& refDouble, CMatrix* inputMat);
};

class CSumation : public CLossFunc
{
public:
	CSumation() {};
	~CSumation() {};
public:
	void GetResult(CMatrix* refMat, CMatrix* inputMat);
	void GetLossGradient (CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat);
private:
	static void LossGradientParallel(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat);
	static void LossGradientSerial(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat);
};

class CSoftmax : public CLossFunc
{
public:
	CSoftmax() {};
	~CSoftmax() {};
public:
	void GetResult(CMatrix* refMat, CMatrix* inputMat);
	void GetLossGradient(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat);
private:
	static void ResultParallel(CMatrix* refMat, CMatrix* inputMat);
	static void ResultSerial(CMatrix* refMat, CMatrix* inputMat);
	static void LossGradientParallel(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat);
	static void LossGradientSerial(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat);
};