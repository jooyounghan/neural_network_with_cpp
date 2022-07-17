#pragma once
#include "CLayer.h"

class CMatrix;

class CActivationLayer : public CLayer
{
public:
	CActivationLayer();
	~CActivationLayer();

public:
	void SetOutput();
	virtual void ForwardProp() = 0;
	virtual CMatrix BackwardProp(CLayer* layer) = 0;
	virtual CMatrix GetDeriviated() = 0;
};

