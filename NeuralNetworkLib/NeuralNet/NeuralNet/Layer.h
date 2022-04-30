#pragma once
#include "ActFunc.h"

class CLayer2D
{
private:
	CMatrix* inputMatrix;
	CMatrix* activatedMatrix;
	CMatrix* weight;
	CMatrix* gradient;

private:
	CMatrix* derivativeActFunc;
	CMatrix* transposedWeight;
	CMatrix* activatedTransposedInput;
	CMatrix* weightGradient;

private:
	CActFunc* activationFunc;

public:
	CLayer2D* former;
	CLayer2D* latter;

public:
	CLayer2D();
	~CLayer2D();

public:
	CMatrix*& GetInput();
	CMatrix*& GetActivatedInput();
	CMatrix*& GetWeight();
	CMatrix*& GetGradient();
	CActFunc*& GetActivationFunc();

public:
	CMatrix*& GetDerivativeActFunc();
	CMatrix*& GetTransposedWeight();
	CMatrix*& GetActivatedTransposedInput();
	CMatrix*& GetWeightGradient();

public:
	void SetInput(CMatrix* newInput);
	void SetActivavtedInput(CMatrix* newInput);
	void SetWeight(CMatrix* newWeight);
	void SetGradient(CMatrix* newGradient);
	void SetActivateFunc(ACTID actId);

	void SetDerivativeActivatedInput(CMatrix* newInput);
	void SetTransposedWeight(CMatrix* newWeight);
	void SetActivatedTransposedInput(CMatrix* newInput);
	void SetWeigthGradient(CMatrix* newWeight);
};
