#pragma once
#include "CLayer.h"

class CMatrix;

class CLossLayer : public CLayer
{
protected:
	CMatrix* labelMatrix;

public:
	CLossLayer();
	~CLossLayer() override;

public:
	void SetLabel(CMatrix* label);
	CMatrix* GetLabel();
	virtual void SetOutput() = 0;
	virtual void ForwardProp() = 0;
	virtual CMatrix BackwardProp(CLayer* layer) = 0;
};

