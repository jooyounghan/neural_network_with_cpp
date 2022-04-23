#pragma once
#include "ActFunc.h"

class CLayer2D
{
private:
	CMatrix* inputMatrix;
	CMatrix* weight;
	CMatrix* gradient;

public:
	CLayer2D* former;
	CLayer2D* latter;

private:
	CActFunc* activationFunc;

public:
	CLayer2D();
	~CLayer2D();

public:
	CMatrix* GetInput();
	CMatrix* GetWeight();
	CMatrix* GetGradient();

public:
	void SetInput(CMatrix* newInput);
	void SetWeight(CMatrix* newWeight);
	void SetGradient(CMatrix* newGradient);
	void SetActivateFunc(uint16 ACTID);
};
