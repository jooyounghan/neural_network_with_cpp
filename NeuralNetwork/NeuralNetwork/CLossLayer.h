#pragma once
#include "CLayer.h"

class CMatrix;

class CLossLayer : public CLayer
{
protected:
	CMatrix* labelMatrix;

public:
	CLossLayer();
	~CLossLayer();

public:
	void SetLabel(CMatrix* label);
	virtual void SetOutput() = 0;
	virtual void ForwardProp() = 0;
	virtual CMatrix BackwardProp(CLayer* layer) = 0;
};

