#include "pch.h"
#include "NeuralNet.h"


#pragma region Constructor
NeuralNet::NeuralNet() {}
#pragma endregion


#pragma region Destructor
NeuralNet::~NeuralNet()
{
	for (CLayer2D*& layer : layers)
	{
		DELETEPTR(layer);
	}
}
#pragma endregion


#pragma region Setter
void NeuralNet::SetLayers(const uint32& dimension, const uint32 numLayers, ...)
{
	va_list layerInfoArgs;
	va_start(layerInfoArgs, numLayers);
	uint32 cmpInfo = dimension;
	for (uint32 layerIdx = 0; layerIdx < numLayers; ++layerIdx)
	{
		uint32 layerInfo = va_arg(layerInfoArgs, uint32);
		CLayer2D* newLayer = new CLayer2D();
		newLayer->SetWeight(new CMatrix(cmpInfo, layerInfo));
		newLayer->SetGradient(new CMatrix(cmpInfo, layerInfo));
		layers.push_back(newLayer);
		cmpInfo = layerInfo;
	}

	for (uint32 layerIdx = 0; layerIdx < numLayers; ++layerIdx)
	{
		CLayer2D*& layer = layers[layerIdx];
		if (layerIdx == 0)
		{
			layer->latter = layers[layerIdx + 1];
		}
		else if (layerIdx == numLayers - 1)
		{
			layer->former = layers[layerIdx - 1];
		}
		else
		{
			layer->former = layers[layerIdx - 1];
			layer->latter = layers[layerIdx + 1];
		}
	}

	va_end(layerInfoArgs);
}

void NeuralNet::SetInputNum(const uint32& number)
{
	ASSERT_CRASH(layers.size());
	CLayer2D*& inputLayer = layers[0];
	inputLayer->SetInput(new CMatrix(number, inputLayer->GetWeight()->GetRow()));
}

void NeuralNet::SetActivationFunc(...)
{
	uint32 layerSize = layers.size();
	ASSERT_CRASH(layerSize);
	va_list ActiveIDs;
	va_start(ActiveIDs, layerSize);

	for (uint32 layerIdx = 0; layerIdx < layerSize; ++layerIdx)
	{
		uint32 ActiveID = va_arg(ActiveIDs, uint32);
		layers[layerIdx]->SetActivateFunc(ActiveID);
	}
}
#pragma endregion


#pragma region Initializer
void NeuralNet::InitializeWeight(...)
{
	uint32 layerSize = layers.size();
	ASSERT_CRASH(layerSize);
	va_list ActiveIDs;
	va_start(ActiveIDs, layerSize);

	for (uint32 layerIdx = 0; layerIdx < layerSize; ++layerIdx)
	{
		uint32 ActiveID = va_arg(ActiveIDs, uint32);
		CMatrix* weight = layers[layerIdx]->GetWeight();
		switch (ActiveID)
		{
		case XAVIER:
			if (layerIdx == 0)
			{
				weight->XavierNormalInitialize(0, layers[layerIdx + 1]->GetWeight()->GetDataNum());
			}
			else if (layerIdx == layerSize - 1)
			{
				weight->XavierNormalInitialize(layers[layerIdx - 1]->GetWeight()->GetDataNum(), 0);
			}
			else
			{
				weight->XavierNormalInitialize(layers[layerIdx - 1]->GetWeight()->GetDataNum(), layers[layerIdx + 1]->GetWeight()->GetDataNum());
			}
			break;
		case HE:
			if (layerIdx == 0)
			{
				weight->HeNormalInitialize(0);
			}
			else if (layerIdx == layerSize - 1)
			{
				weight->HeNormalInitialize(layers[layerIdx - 1]->GetWeight()->GetDataNum());
			}
			else
			{
				weight->HeNormalInitialize(layers[layerIdx - 1]->GetWeight()->GetDataNum());
			}
			break;
		default:
			break;
		}
	}
}
#pragma endregion

#pragma region Propagation
void NeuralNet::PushInput(CMatrix* inputMatrix)
{
	ASSERT_CRASH(layers.size());
	layers[0]->SetInput(inputMatrix);
}

void NeuralNet::ForwardPropagation()
{
	ASSERT_CRASH(layers.size());
	CLayer2D*& calcLayer = layers[0];

	while (calcLayer != nullptr)
	{
		calcLayer->latter->GetInput()->GetMatMul(calcLayer->GetInput(), calcLayer->latter->GetWeight());
		calcLayer = calcLayer->latter;
	}
}

void NeuralNet::BackwardPropagation()
{
	ASSERT_CRASH(layers.size());

}
#pragma endregion
