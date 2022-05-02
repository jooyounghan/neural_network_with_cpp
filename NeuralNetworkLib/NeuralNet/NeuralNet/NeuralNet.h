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
	void SetLayers(const uint32& dimension, const uint32 numLayers, va_list& val);
	void SetInputNum(const uint32& number);
	void SetInput(CMatrix* inputMatrix);
	void SetLossGradient(CMatrix* inputGradient);
	void FlushInput();

public:
	CMatrix*& GetOutputMatrix();
	CMatrix*& GetLossGradient();
	const uint32& GetLayersNum();

public:
	void SetActivationFunc(va_list& ActiveIDs);
	void InitializeWeight(va_list& InitializerIDs);

public:
	void ForwardPropagation();
	void BackwardPropagation(const double& learningRate);
};