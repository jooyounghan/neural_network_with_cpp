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
	CNNModel(const uint32& number, const uint32& dimension, const uint32& numLayers, ...);
	~CNNModel();

public:


	void Train(std::vector<CMatrix> inputVector, std::vector<CMatrix> labelVector, const uint32& iterations)
	{
		ASSERT_CRASH(NeuralNet != nullptr);
		ASSERT_CRASH(lossFunc != nullptr);

		CRandomGenerator randGen;
		for (uint32 iter = 0; iter < iterations; ++iter)
		{
			// Shuffle with Same Order
			randGen.ShuffleVector(inputVector, 100);
			randGen.ShuffleVector(labelVector, 100);

			for (CMatrix inputMatrix : inputVector)
			{
				NeuralNet->SetInput(&inputMatrix);
				NeuralNet->ForwardPropagation();
				//lossFunc->GetResult(NeuralNet)
			}
		}
	}
};