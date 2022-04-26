#pragma once
#include "NeuralNet.h"

class NNModel
{
private:
	CNeuralNetwork* NeuralNet;
	std::vector<double> lossVector;

public:

	// Pseudo Code of Training
	//void Train()
	//{
	//	NeuralNet->SetInput(~);
	//	NeuralNet->ForwardPropagation();
	//	lossVector.push_back(~);
	//	NeuralNet->SetLossGradient(~);
	//	NeuralNet->BackwardPropagation();
	//}
};