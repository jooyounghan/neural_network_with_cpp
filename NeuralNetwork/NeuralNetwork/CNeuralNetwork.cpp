#include "pch.h"
#include "CNeuralNetwork.h"

#include "CMatrix.h"

#include "CRelu.h"
#include "CSigmoid.h"
#include "CSoftMax.h"

#include "CLeastSquare.h"
#include "CCrossEntropy.h"

#include "CHiddenLayer.h"

CNeuralNetwork::CNeuralNetwork() : optimizer(nullptr)
{
}

CNeuralNetwork::~CNeuralNetwork()
{

}

bool CNeuralNetwork::AddActivationLayer(ACTTYPE type)
{
	bool result = true;
	CLayer* layer = nullptr;
	switch (type)
	{
	case ACTTYPE::RELU:
		layer = new CRelu();
		break;
	case ACTTYPE::SIGMOID:
		layer = new CSigmoid();
		break;
	case ACTTYPE::SOFTMAX:
		layer = new CSoftMax();
		break;
	default:
		result = false;
	}
	if (layer != nullptr) layers.push_back(layer);
	return result;
}

bool CNeuralNetwork::AddLossLayer(LOSSTYPE type)
{
	bool result = true;
	CLayer* layer = nullptr;
	switch (type)
	{
	case LOSSTYPE::LEASTSQR:
		layer = new CLeastSquare();
		break;
	case LOSSTYPE::CROSSENT:
		//layer = new CCrossEntropy();
		//break;
	default:
		result = false;
		break;
	}
	if (layer != nullptr) layers.push_back(layer);
	return result;
}

bool CNeuralNetwork::AddHiddenLayer(const UINT& dim)
{
	CLayer* layer = new CHiddenLayer(dim);
	layers.push_back(layer);
	return true;
}

bool CNeuralNetwork::SetInputProperties(const UINT& inputDim, const UINT& inputDatanum)
{
	const UINT& layerNum = layers.size();
	ASSERT_CRASH(layerNum > 0);

	CLayer& firstLayer = *layers[0];
	CLayer& lastLayer = *layers[0];
	CMatrix* newInput = new CMatrix(inputDim, inputDatanum);

	firstLayer.SetInput(newInput);
	for (UINT idx = 0; idx < layerNum - 1; ++idx)
	{
		layers[idx]->Connect(layers[idx + 1]);
	}
	lastLayer.SetOutput();

	return true;
}

bool CNeuralNetwork::CompareInputProperties(CMatrix* input)
{
	const UINT& layerNum = layers.size();
	ASSERT_CRASH(layerNum > 0);

	CLayer& firstLayer = *layers[0];
	CMatrix* firstLayerInput = firstLayer.GetInput();

	if (firstLayerInput != nullptr && firstLayerInput->GetCol() == input->GetCol())
		return false;
	else
		return true;
	
}

CMatrix CNeuralNetwork::Predict(CMatrix* input)
{
	if (CompareInputProperties(input))
	{
		const UINT& inputDim = input->GetRow();
		const UINT& inputDatanum = input->GetCol();
		SetInputProperties(inputDim, inputDatanum);
	}

	const UINT& layerNum = layers.size();
	ASSERT_CRASH(layerNum > 0);

	CLayer& firstLayer = *layers[0];
	CMatrix* firstLayerInput = firstLayer.GetInput();

	firstLayer.SetInput(input);
	for (UINT idx = 0; idx < layerNum; ++idx)
	{
		layers[idx]->ForwardProp();
	}
	CLayer& lastLayer = *layers[layerNum - 1];
	CMatrix* lastLayerOutput = lastLayer.GetOutput();

	return lastLayerOutput->Copy();
}

double CNeuralNetwork::Train(CMatrix* input)
{
	if (CompareInputProperties(input))
	{
		const UINT& inputDim = input->GetRow();
		const UINT& inputDatanum = input->GetCol();
		SetInputProperties(inputDim, inputDatanum);
	}

	const UINT& layerNum = layers.size();
	ASSERT_CRASH(layerNum > 0);

	CLayer& firstLayer = *layers[0];
	CMatrix* firstLayerInput = firstLayer.GetInput();

	firstLayer.SetInput(input);
	for (auto layerIter = layers.begin(); layerIter != layers.end(); ++layerIter)
	{
		(*layerIter)->ForwardProp();
	}

	CLayer& lastLayer = *layers[layerNum - 1];
	CMatrix* lastLayerOutput = lastLayer.GetOutput();

	double loss = lastLayerOutput->GetSum();

	for (auto layerIter = layers.rbegin(); layerIter != layers.rend(); ++layerIter)
	{
		CMatrix* backwardMatrix;
		CLayer* layer = *layerIter;
		if (layerIter == layers.rbegin())
		{
			backwardMatrix = new CMatrix(layer->BackwardProp(nullptr));
		}
		else
		{
			CLayer* latterLayer = *(layerIter--);
			backwardMatrix = new CMatrix(layer->BackwardProp(latterLayer));

			if (typeid(CHiddenLayer).name() == typeid(*layer).name())
			{
				CHiddenLayer* hiddenLayer = reinterpret_cast<CHiddenLayer*>(layer);
				
				CMatrix* weight = hiddenLayer->GetWeight();
				weight->RefSub(hiddenLayer->GetWeightGradient(latterLayer), *weight);
			}

			layer->SetGradient(backwardMatrix);
		}
	}
	
	return loss;
}
