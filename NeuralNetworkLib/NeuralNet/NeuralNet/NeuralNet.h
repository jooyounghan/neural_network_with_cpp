#pragma once
#include "Layer.h"
#include "Timer.h"

class CNeuralNetwork
{
private:
	std::vector<CLayer2D*> layers;

private:
	CMatrix* outputMatrix;
	CMatrix* lossGradient;

public:
	CNeuralNetwork();
	~CNeuralNetwork();

public:
	void SetLayers(const uint32& dimension, const uint32 numLayers, ...);
	void SetInputNum(const uint32& number);
	void SetInput(CMatrix* inputMatrix);
	void SetLossGradient(CMatrix* inputGradient);

public:
	void SetActivationFunc(...);
	void InitializeWeight(...);

public:
	void ForwardPropagation();
	void BackwardPropagation();
};