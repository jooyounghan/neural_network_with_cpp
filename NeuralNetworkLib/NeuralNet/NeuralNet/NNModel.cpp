#include "pch.h"
#include "NNModel.h"

CNNModel::CNNModel(LOSSFUNCID lossFuncId) : NeuralNet(nullptr), loss(0)
{
	NeuralNet = new CNeuralNetwork();
	switch (lossFuncId)
	{
	case SUMATION:
		
	case SOFTMAX:
		
	default:
		break;
	}
}

CNNModel::~CNNModel()
{
	DELETEPTR(NeuralNet);
}
