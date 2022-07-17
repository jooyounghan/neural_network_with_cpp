#pragma once
#include "CLayer.h"

class CMatrix;

class CHiddenLayer : public CLayer
{
protected:
	UINT outputDim;
	CMatrix* weight;

public:
	CHiddenLayer(const UINT outputDim);
	~CHiddenLayer();
public:
	void SetOutput();
	void ForwardProp() override;
	CMatrix BackwardProp(CLayer* layer) override;

public:
	void InitializeWeight(const UINT& type);
};
