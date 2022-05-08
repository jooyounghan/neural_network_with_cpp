#pragma once
#include "Layer.h"
#include "Timer.h"

class CNeuralNetwork
{
private:
	std::vector<std::shared_ptr<CLayer2D>> layers;

private:
	std::shared_ptr <CMatrix> outputMatrix;
	std::shared_ptr <CMatrix> lossGradient;

public:
	CNeuralNetwork();
	~CNeuralNetwork();

public:
	void SetLayers(const uint32& inputDimension, const uint32& outputDimension, const uint32 numLayers, va_list& val);
	void SetInputNum(const uint32& number);
	void SetInput(std::shared_ptr <CMatrix> inputMatrix);
	void SetLossGradient(std::shared_ptr <CMatrix> inputGradient);

public:
	std::shared_ptr<CMatrix>& GetOutputMatrix();
	std::shared_ptr<CMatrix>& GetLossGradient();
	const uint32 GetLayersNum();

public:
	void SetActivationFunc(va_list& ActiveIDs);
	void InitializeWeight(va_list& InitializerIDs);

public:
	void ForwardPropagation();
	void BackwardPropagation(const double& learningRate);
};