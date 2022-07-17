#pragma once
#include "CActivationLayer.h"

class CMatrix;

class CSigmoid : public CActivationLayer
{
public:
	CSigmoid();
	~CSigmoid();

public:
	void ForwardProp() override;
	CMatrix BackwardProp(CLayer* layer) override;
	CMatrix GetDeriviated() override;
};

