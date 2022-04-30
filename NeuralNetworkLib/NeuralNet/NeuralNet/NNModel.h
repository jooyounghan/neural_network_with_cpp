#pragma once
#include "NeuralNet.h"

class CNNModel
{
private:
	CNeuralNetwork* NeuralNet;
	double loss;

public:
	CNNModel(LOSSFUNCID lossFuncId);
	~CNNModel();

public:

	// Pseudo Code of Training
	//void Train(std::vector<CMatrix*> inputVector, const uint32& iterations)
	//{
	//	for (uint32 iter = 0; iter < iterations; ++iter)
	// {
	//		NeuralNet->SetInput(~);
	//		NeuralNet->ForwardPropagation();
	//		lossVector.push_back(~);
	//		NeuralNet->SetLossGradient(~);
	//		NeuralNet->BackwardPropagation();
	// }
	//}
};