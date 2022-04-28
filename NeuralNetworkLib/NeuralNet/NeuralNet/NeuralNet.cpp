#include "pch.h"
#include "NeuralNet.h"


#pragma region Constructor
CNeuralNetwork::CNeuralNetwork() : outputMatrix(nullptr), lossGradient(nullptr) {}
#pragma endregion


#pragma region Destructor
CNeuralNetwork::~CNeuralNetwork()
{
	for (CLayer2D*& layer : layers)
	{
		DELETEPTR(layer);
	}
}
#pragma endregion


#pragma region Setter
void CNeuralNetwork::SetLayers(const uint32& dimension, const uint32 numLayers, ...)
{
	va_list layerInfoArgs;
	va_start(layerInfoArgs, numLayers);
	uint32 cmpInfo = dimension;
	for (uint32 layerIdx = 0; layerIdx < numLayers; ++layerIdx)
	{
		uint32 layerInfo = va_arg(layerInfoArgs, uint32);
		CLayer2D* newLayer = new CLayer2D();

		newLayer->SetWeight(new CMatrix(cmpInfo, layerInfo));
		newLayer->SetWeigthGradient(new CMatrix(cmpInfo, layerInfo));
		newLayer->SetTransposedWeight(new CMatrix(layerInfo, cmpInfo));
		layers.push_back(newLayer);
		cmpInfo = layerInfo;
	}

	CLayer2D*& lastLayer = layers[layers.size() - 1];
	CLayer2D* newLastLayer = new CLayer2D();

	const uint32& lastWeightCol = lastLayer->GetWeight()->GetCol();
	newLastLayer->SetWeight(new CMatrix(lastWeightCol, 1));
	newLastLayer->SetWeigthGradient(new CMatrix(lastWeightCol, 1));
	newLastLayer->SetTransposedWeight(new CMatrix(lastWeightCol, 1));
	layers.push_back(newLastLayer);

	const uint32& lastNumLayers = layers.size();
	for (uint32 layerIdx = 0; layerIdx < lastNumLayers; ++layerIdx)
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

void CNeuralNetwork::SetInputNum(const uint32& number)
{
	const uint32& numLayers = layers.size();
	ASSERT_CRASH(numLayers - 1 > 0);

	for (uint32 idx = 0; idx < numLayers - 1; ++idx)
	{
		CLayer2D*& thisLayer = layers[idx];
		CLayer2D*& nextLayer = layers[idx + 1];

		const uint32& thisWeightRow = thisLayer->GetWeight()->GetRow();
		const uint32& thisWeightCol = thisLayer->GetWeight()->GetCol();
		if (idx == 0)
		{
			thisLayer->SetInput(new CMatrix(number, thisWeightRow));
			thisLayer->SetActivavtedInput(new CMatrix(number, thisWeightRow));
			thisLayer->SetGradient(new CMatrix(number, thisWeightRow));
			thisLayer->SetDerivativeActivatedInput(new CMatrix(number, thisWeightRow));
			thisLayer->SetActivatedTransposedInput(new CMatrix(thisWeightRow, number));
		}

		const uint32& thisInputRow = thisLayer->GetInput()->GetRow();
		const uint32& thisInputCol = thisLayer->GetInput()->GetCol();

		nextLayer->SetInput(new CMatrix(thisInputRow, thisWeightCol));
		nextLayer->SetActivavtedInput(new CMatrix(thisInputRow, thisWeightCol));
		nextLayer->SetGradient(new CMatrix(thisInputRow, thisWeightCol));
		nextLayer->SetDerivativeActivatedInput(new CMatrix(thisInputRow, thisWeightCol));
		nextLayer->SetActivatedTransposedInput(new CMatrix(thisWeightCol, thisInputRow));
	}
}

void CNeuralNetwork::SetActivationFunc(...)
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

void CNeuralNetwork::SetInput(CMatrix* inputMatrix)
{
	ASSERT_CRASH(layers.size());
	layers[0]->SetInput(inputMatrix);
}

void CNeuralNetwork::SetLossGradient(CMatrix* inputGradient)
{
	ASSERT_CRASH(layers.size());
	lossGradient = inputGradient;
}

#pragma endregion


#pragma region Initializer
void CNeuralNetwork::InitializeWeight(...)
{
	uint32 layerSize = layers.size();
	ASSERT_CRASH(layerSize > 0);
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
void CNeuralNetwork::ForwardPropagation()
{
	ASSERT_CRASH(layers.size());
	CLayer2D*& calcLayer = layers[0];

	while (calcLayer->latter != nullptr)
	{
		CActFunc*& activatedFunc = calcLayer->GetActivationFunc();
		CMatrix*& activatedInputMatrix = calcLayer->GetActivatedInput();
		CMatrix*& inputMatrix = calcLayer->GetInput();
		CMatrix*& inputWeightMatrix = calcLayer->GetWeight();

		// Activation Function
		activatedFunc->CalcResult(activatedInputMatrix, inputMatrix);

		// Matrix Multiplication
		calcLayer->latter->GetInput()->MatMul(inputMatrix, inputWeightMatrix);
		calcLayer = calcLayer->latter;
	}

	CLayer2D*& lastLayer = layers[layers.size() - 1];
	outputMatrix->MatMul(lastLayer->GetInput(), lastLayer->GetWeight());
}

void CNeuralNetwork::BackwardPropagation()
{
	ASSERT_CRASH(layers.size());
	ASSERT_CRASH(lossGradient != nullptr);
	CLayer2D*& calcLayer = layers[layers.size() - 1];

	while (calcLayer == nullptr)
	{
		CMatrix*& inputMatrix = calcLayer->GetInput();
		CMatrix*& weightMatrix = calcLayer->GetWeight();
		CMatrix*& activatedInputMatrix = calcLayer->GetActivatedInput();
		CMatrix*& gradientMatrix = calcLayer->GetGradient();

		CMatrix*& derivativeActFunc = calcLayer->GetDerivativeActFunc();
		CMatrix*& transposedWeight = calcLayer->GetTransposedWeight();
		CMatrix*& activatedTransposedInput = calcLayer->GetActivatedTransposedInput();
		CMatrix*& weightGradient = calcLayer->GetWeightGradient();

		CActFunc*& CalcActFunc = calcLayer->GetActivationFunc();

		CalcActFunc->CalcResult(derivativeActFunc, inputMatrix);
		transposedWeight->Transpose(weightMatrix);
		activatedTransposedInput->Transpose(activatedInputMatrix);

		if (calcLayer->latter == nullptr)
		{
			derivativeActFunc->ElementWiseMul(derivativeActFunc, lossGradient);
			weightGradient->MatMul(activatedTransposedInput, lossGradient);
		}
		else
		{
			derivativeActFunc->ElementWiseMul(derivativeActFunc, calcLayer->latter->GetGradient());
			weightGradient->MatMul(activatedTransposedInput, calcLayer->latter->GetGradient());
		}

		gradientMatrix->MatMul(derivativeActFunc, transposedWeight);
		
		// Updating
		weightMatrix->Subtract(weightGradient);

		calcLayer = calcLayer->former;
	}
}


#pragma endregion
