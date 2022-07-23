#pragma once
#include "CLossLayer.h"

class CMatrix;

class CLeastSquare : public CLossLayer
{
public:
	CLeastSquare();
	~CLeastSquare();

public:
	void SetOutput() override;
	void ForwardProp() override;
	CMatrix BackwardProp(CLayer* layer) override;
};