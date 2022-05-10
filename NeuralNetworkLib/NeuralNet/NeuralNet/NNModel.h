#pragma once
#include "NeuralNet.h"
#include "LossFunc.h"

class CNNModel
{
private:
	std::shared_ptr<CNeuralNetwork> NeuralNet;
	std::shared_ptr<CLossFunc> lossFunc;
	double loss;

public:
	CNNModel(const uint32& inputCnt, const uint32& inputDimension, const uint32& outputDimension, const uint32 numLayers, ...);
	~CNNModel();

public:
	void SetActivationFunc(const uint32 numLayers, ...);
	void InitializeWeights(const uint32 numLayers, ...);
	void SetLossFunc(const LOSSFUNCID& lfID);


public:
	void Train(std::vector<std::shared_ptr<CMatrix>> inputVector, std::vector<std::shared_ptr<CMatrix>> labelVector, const uint32& iterations, const double& learningRate);
	void PushInput(std::shared_ptr<CMatrix> inputMatrix, std::shared_ptr<CMatrix> labelMatrix, const double& learningRate);

public:
	std::shared_ptr<CMatrix> GetResult(std::shared_ptr<CMatrix> inputMatrix);
	const double& GetLoss();
	std::shared_ptr<CNeuralNetwork> GetNeuralNetwork();
private:
	void Propagation(std::shared_ptr<CMatrix> inputMatrix, std::shared_ptr<CMatrix> labelMatrix, const double& learningRate);

};