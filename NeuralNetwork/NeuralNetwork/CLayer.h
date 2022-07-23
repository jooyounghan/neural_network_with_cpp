#pragma once

#include "CMatrix.h"

class CMatrix;

class CLayer
{
friend class CInitializer;

protected:
	CMatrix* input;
	CMatrix* output;
	CMatrix* gradient;

public:
	CLayer();
	~CLayer();

public:
	void SetInput(CMatrix* newInput);
	virtual void SetOutput() = 0;
	void SetGradient(CMatrix* newGradient);

public:
	void Connect(CLayer* layer);

public:
	virtual void ForwardProp() = 0;
	virtual CMatrix BackwardProp(CLayer* layer) = 0;

public:
	CMatrix* GetInput();
	CMatrix* GetOutput();
	CMatrix* GetGradient();
};

