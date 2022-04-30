#pragma once
#include "Matrix.h"

class CLossFunc
{
protected:
	CMatrix* predictedMatrix;

protected:
	CLossFunc(CMatrix* outputMatrix);
public:
	~CLossFunc() {}

public:
	virtual void GetResult(CMatrix* inputMat, CMatrix* labelMat) = 0;
	void GetLoss(double& refDouble, CMatrix* inputMat);
};

class CSumation : public CLossFunc
{
public:
	CSumation(CMatrix* outputMatrix) : CLossFunc(outputMatrix) {};
	~CSumation() {};
public:
	void GetResult(CMatrix* inputMat, CMatrix* labelMat);
private:
	static void ResultParallel(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat);
	static void ResultSerial(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat);
};

class CSoftmax : public CLossFunc
{
public:
	CSoftmax(CMatrix* outputMatrix) : CLossFunc(outputMatrix) {};
	~CSoftmax() {};
public:
	void GetResult(CMatrix* inputMat, CMatrix* labelMat);
private:
	static void ResultParallel(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat);
	static void ResultSerial(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat);
};