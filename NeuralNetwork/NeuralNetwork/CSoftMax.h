#pragma once
#include "CActivationLayer.h"

class CMatrix;

class CSoftMax : public CActivationLayer
{
public:
	CSoftMax();
	~CSoftMax();

public:
	void ForwardProp() override;
	CMatrix BackwardProp(CLayer* layer) override;
	CMatrix GetDeriviated() override;
};