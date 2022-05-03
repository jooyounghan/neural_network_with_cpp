#include "pch.h"
#include "NNModel.h"

CNNModel::CNNModel(const uint32& inputCnt, const uint32& dimension, const uint32 numLayers, ...)
	: NeuralNet(nullptr), lossFunc(nullptr), loss(0.0)
{
	va_list argsList;
	va_start(argsList, numLayers);

	NeuralNet = new CNeuralNetwork();
	NeuralNet->SetLayers(dimension, numLayers, argsList);
	NeuralNet->SetInputNum(inputCnt);
	va_end(argsList);
}

CNNModel::~CNNModel()
{
	DELETEPTR(NeuralNet);
	DELETEPTR(lossFunc);
}

void CNNModel::SetActivationFunc(const uint32 numLayers, ...)
{
	ASSERT_CRASH(NeuralNet != nullptr);
	ASSERT_CRASH(numLayers == NeuralNet->GetLayersNum());
	va_list val;
	va_start(val, numLayers);
	NeuralNet->SetActivationFunc(val);
	va_end(val);
}

void CNNModel::SetLossFunc(const LOSSFUNCID& lfID)
{
	DELETEPTR(lossFunc);
	switch (lfID)
	{
	case LF_SUMATION :
		lossFunc = new CSumation();
		break;
	case LF_SOFTMAX :
		lossFunc = new CSoftmax();
		break;
	default:
		break;
	}
}

void CNNModel::InitializeWeights(const uint32 numLayers, ...)
{
	ASSERT_CRASH(NeuralNet != nullptr);
	ASSERT_CRASH(numLayers == NeuralNet->GetLayersNum());
	va_list val;
	va_start(val, numLayers);
	NeuralNet->InitializeWeight(val);
	va_end(val);
}

void CNNModel::Train(std::vector<CMatrix> inputVector, std::vector<CMatrix> labelVector, const uint32& iterations, const double& learningRate)
{
	const uint32& inputSize = inputVector.size();
	const uint32& labelSize = labelVector.size();
	ASSERT_CRASH(NeuralNet != nullptr);
	ASSERT_CRASH(lossFunc != nullptr);
	ASSERT_CRASH(inputSize == labelSize);

	CRandomGenerator randGen;
	for (uint32 iter = 0; iter < iterations; ++iter)
	{
		// Shuffle with Same Order
		//randGen.ShuffleVector(inputVector, 100);
		//randGen.ShuffleVector(labelVector, 100);
		for (uint32 idx = 0; idx < inputVector.size(); ++idx)
		{
			Propagation(&inputVector[idx], &labelVector[idx], learningRate);
		}
	}
}

void CNNModel::PushInput(CMatrix inputMatrix, CMatrix labelMatrix, const double& learningRate)
{
	ASSERT_CRASH(NeuralNet != nullptr);
	ASSERT_CRASH(lossFunc != nullptr);

	Propagation(&inputMatrix, &labelMatrix, learningRate);
}

CMatrix* CNNModel::GetResult(CMatrix inputMatrix)
{
	NeuralNet->FlushInput();
	NeuralNet->SetInput(&inputMatrix);
	NeuralNet->ForwardPropagation();
	CMatrix* ResultMatrix = NeuralNet->GetOutputMatrix()->GetCopyMatrix();
	return ResultMatrix;
}

void CNNModel::Propagation(CMatrix* inputMatrix, CMatrix* labelMatrix, const double& learningRate)
{
	NeuralNet->FlushInput();
	NeuralNet->SetInput(inputMatrix);
	NeuralNet->ForwardPropagation();

	CMatrix*& outputMatrix = NeuralNet->GetOutputMatrix();
	CMatrix*& lossMatrix = NeuralNet->GetLossGradient();

	lossFunc->GetResult(outputMatrix, inputMatrix);
	lossFunc->GetLossGradient(lossMatrix, outputMatrix, labelMatrix);

	NeuralNet->BackwardPropagation(learningRate);
	NeuralNet->FlushInput();
}
