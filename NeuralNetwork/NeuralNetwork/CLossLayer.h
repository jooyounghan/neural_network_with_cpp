#pragma once
#include "CLayer.h"

class CMatrix;

class CLossLayer : public CLayer
{
public:
	CLossLayer();
	~CLossLayer();

public:
	virtual void SetOutput() = 0;
	void ForwardProp() override;
	CMatrix BackwardProp(CLayer* layer) override;
};

