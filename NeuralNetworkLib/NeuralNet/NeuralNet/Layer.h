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
	void SetInput(CMatrix* newInput);
	void SetWeight(CMatrix* newWeight);
	void SetGradient(CMatrix* newGradient);
	void SetActivateFunc(uint16 ACTID);
};
