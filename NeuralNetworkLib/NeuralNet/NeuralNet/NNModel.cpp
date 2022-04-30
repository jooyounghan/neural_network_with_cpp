#include "pch.h"
#include "NNModel.h"

CNNModel::CNNModel(const uint32& inputCnt, const uint32& dimension, const uint32& numLayers, ...)
	: NeuralNet(nullptr), lossFunc(nullptr), loss(0.0)
{
	va_list val;
	va_start(val, numLayers);
	NeuralNet = new CNeuralNetwork();
	NeuralNet->SetLayers(dimension, numLayers, val);
	NeuralNet->SetInputNum(inputCnt);
	va_end(val);
}

CNNModel::~CNNModel()
{
	DELETEPTR(NeuralNet);
	DELETEPTR(lossFunc);
}
