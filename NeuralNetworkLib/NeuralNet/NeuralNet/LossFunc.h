#pragma once
#include "Matrix.h"

using LOSSFUNCID = unsigned short;

class CLossFunc
{
public:
	enum LossID
	{
		SUMATION,
		SOFTMAX,
	};

public:
	CLossFunc();
	~CLossFunc() {}

public:
	virtual void GetResult(CMatrix* refMat, CMatrix* inputMat) = 0;
	virtual void GetLossGradient(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat) = 0;
	static void GetLoss(double& refDouble, CMatrix* resultMat, CMatrix* labelMat);
};

class CSumation : public CLossFunc
{
public:
	CSumation() {};
	~CSumation() {};
public:
	void GetResult(CMatrix* refMat, CMatrix* inputMat) override;
	void GetLossGradient (CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat) override;
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
	void GetResult(CMatrix* refMat, CMatrix* inputMat) override;
	void GetLossGradient(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat) override;
	
private:
	static void LossGradientParallel(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat);
	static void LossGradientSerial(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat);
};