#pragma once
#include "CLayer.h"
#include "CInitializer.h"

class CMatrix;

class CHiddenLayer : public CLayer
{
protected:
	UINT outputDim;
	CMatrix* weight;

public:
	CHiddenLayer(const UINT outputDim);
	~CHiddenLayer() override;
public:
	void SetOutput();
	void ForwardProp() override;
	CMatrix BackwardProp(CLayer* layer) override;

	CMatrix* GetWeight();
	CMatrix GetWeightGradient(CLayer* layer);
public:
	void InitializeWeight(CInitializer::INITTYPE type);
};

