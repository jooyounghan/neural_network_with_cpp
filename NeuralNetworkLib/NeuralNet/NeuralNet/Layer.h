#pragma once
#include "ActFunc.h"
#include "Matrix.h"

class CLayer2D
{
private:
	CMatrix* inputMatrix;
	CMatrix* weight;
	CMatrix* gradient;

private:
	CActFunc<CMatrix>* activationFunc;

public:
	CLayer2D();
	~CLayer2D();
};
