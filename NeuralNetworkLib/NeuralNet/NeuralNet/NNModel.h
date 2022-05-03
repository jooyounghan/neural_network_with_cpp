#pragma once
#include "NeuralNet.h"
#include "LossFunc.h"

class CNNModel
{
private:
	CNeuralNetwork* NeuralNet;
	CLossFunc* lossFunc;
	double loss;

public:
	CNNModel(const uint32& inputCnt, const uint32& dimension, const uint32 numLayers, ...);
	~CNNModel();

public:
	void SetActivationFunc(const uint32 numLayers, ...);
	void InitializeWeights(const uint32 numLayers, ...);
	void SetLossFunc(const LOSSFUNCID& lfID);


public:
	void Train(std::vector<CMatrix> inputVector, std::vector<CMatrix> labelVector, const uint32& iterations, const double& learningRate);
	void PushInput(CMatrix inputMatrix, CMatrix labelMatrix, const double& learningRate);

	CMatrix* GetResult(CMatrix inputMatrix);

private:
	void Propagation(CMatrix* inputMatrix, CMatrix* labelMatrix, const double& learningRate);

};