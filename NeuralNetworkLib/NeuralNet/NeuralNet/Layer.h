#pragma once
#include "ActFunc.h"

class CLayer2D
{
private:
	CMatrix* inputMatrix;
	CMatrix* weight;
	CMatrix* gradient;

private:
	CActFunc* activationFunc;

public:
	CLayer2D();
	~CLayer2D();
};
