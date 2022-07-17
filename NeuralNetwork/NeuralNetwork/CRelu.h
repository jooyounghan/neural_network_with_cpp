#pragma once
#include "CActivationLayer.h"

class CMatrix;

class CRelu : public CActivationLayer
{
public:
	CRelu();
	~CRelu();

public:
	void ForwardProp() override;
	CMatrix BackwardProp(CLayer* layer) override;
	CMatrix GetDeriviated() override;
};

