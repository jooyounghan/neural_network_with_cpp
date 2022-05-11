#include "pch.h"
#include "NeuralNet.h"


#pragma region Constructor
CNeuralNetwork::CNeuralNetwork() : outputMatrix(nullptr), lossGradient(nullptr){}
#pragma endregion


#pragma region Destructor
CNeuralNetwork::~CNeuralNetwork()
{
}
#pragma endregion


#pragma region Setter
void CNeuralNetwork::SetLayers(const uint32& inputDimension, const uint32& outputDimension, const uint32 numLayers, va_list& val)
{
	uint32 cmpInfo = inputDimension;
	for (uint32 layerIdx = 0; layerIdx < numLayers; ++layerIdx)
	{
		uint32 layerInfo = va_arg(val, uint32);
		std::shared_ptr<CLayer2D> newLayer = std::make_shared<CLayer2D>();

		newLayer->SetWeight(std::make_shared<CMatrix>(cmpInfo, layerInfo));
		newLayer->SetWeigthGradient(std::make_shared <CMatrix>(cmpInfo, layerInfo));
		newLayer->SetTransposedWeight(std::make_shared<CMatrix>(layerInfo, cmpInfo));
		layers.push_back(newLayer);
		cmpInfo = layerInfo;
	}

	std::shared_ptr<CLayer2D>& lastLayer = layers[layers.size() - 1];
	std::shared_ptr<CLayer2D> newLastLayer = std::make_shared<CLayer2D>();

	const uint32& lastWeightCol = lastLayer->GetWeight()->GetCol();
	newLastLayer->SetWeight(std::make_shared<CMatrix>(lastWeightCol, outputDimension));
	newLastLayer->SetWeigthGradient(std::make_shared<CMatrix>(lastWeightCol, outputDimension));
	newLastLayer->SetTransposedWeight(std::make_shared<CMatrix>(outputDimension, lastWeightCol));
	layers.push_back(newLastLayer);

	const uint32& lastNumLayers = layers.size();
	for (uint32 layerIdx = 0; layerIdx < lastNumLayers; ++layerIdx)
	{
		std::shared_ptr<CLayer2D>& layer = layers[layerIdx];
		if (layerIdx == 0)
		{
			layer->latter = layers[layerIdx + 1];
		}
		else if (layerIdx == numLayers)
		{
			layer->former = layers[layerIdx - 1];
		}
		else
		{
			layer->former = layers[layerIdx - 1];
			layer->latter = layers[layerIdx + 1];
		}
	}
}

void CNeuralNetwork::SetInputNum(const uint32& number)
{
	const uint32& numLayers = layers.size();
	ASSERT_CRASH(numLayers - 1 > 0);

	for (uint32 idx = 0; idx < numLayers - 1; ++idx)
	{
		std::shared_ptr<CLayer2D>& thisLayer = layers[idx];
		std::shared_ptr<CLayer2D>& nextLayer = layers[idx + 1];

		const uint32& thisWeightRow = thisLayer->GetWeight()->GetRow();
		const uint32& thisWeightCol = thisLayer->GetWeight()->GetCol();
		if (idx == 0)
		{
			thisLayer->SetInput(std::make_shared<CMatrix>(number, thisWeightRow));
			thisLayer->SetActivavtedInput(std::make_shared<CMatrix>(number, thisWeightRow));
			thisLayer->SetGradient(std::make_shared<CMatrix>(number, thisWeightRow));
			thisLayer->SetDerivativeActivatedInput(std::make_shared<CMatrix>(number, thisWeightRow));
			thisLayer->SetActivatedTransposedInput(std::make_shared<CMatrix>(thisWeightRow, number));
		}

		const uint32& thisInputRow = thisLayer->GetInput()->GetRow();
		const uint32& thisInputCol = thisLayer->GetInput()->GetCol();

		nextLayer->SetInput(std::make_shared<CMatrix>(thisInputRow, thisWeightCol));
		nextLayer->SetActivavtedInput(std::make_shared<CMatrix>(thisInputRow, thisWeightCol));
		nextLayer->SetGradient(std::make_shared<CMatrix>(thisInputRow, thisWeightCol));
		nextLayer->SetDerivativeActivatedInput(std::make_shared<CMatrix>(thisInputRow, thisWeightCol));
		nextLayer->SetActivatedTransposedInput(std::make_shared<CMatrix>(thisWeightCol, thisInputRow));
	}

	std::shared_ptr<CLayer2D>& lastLayer = layers[layers.size() - 1];
	const uint32& outputRow = lastLayer->GetInput()->GetRow();
	const uint32& outputCol = lastLayer->GetWeight()->GetCol();

	outputMatrix = std::make_shared<CMatrix>(outputRow, outputCol);
	lossGradient = std::make_shared<CMatrix>(outputRow, outputCol);
}

void CNeuralNetwork::SetActivationFunc(va_list& ActiveIDs)
{
	const uint32& layerSize = layers.size();
	ASSERT_CRASH(layerSize);

	for (uint32 layerIdx = 0; layerIdx < layerSize; ++layerIdx)
	{
		uint32 ActiveID = va_arg(ActiveIDs, uint32);
		layers[layerIdx]->SetActivateFunc(ActiveID);
	}
}

void CNeuralNetwork::SetInput(std::shared_ptr<CMatrix> inputMatrix)
{
	ASSERT_CRASH(layers.size());
	layers[0]->SetInput(inputMatrix);
}

void CNeuralNetwork::SetLossGradient(std::shared_ptr<CMatrix> inputGradient)
{
	ASSERT_CRASH(layers.size());
	lossGradient = inputGradient;
}
#pragma endregion

#pragma region Getter
std::shared_ptr<CMatrix>& CNeuralNetwork::GetOutputMatrix()
{
	return outputMatrix;
}

std::shared_ptr<CMatrix>& CNeuralNetwork::GetLossGradient()
{
	return lossGradient;
}

const uint32 CNeuralNetwork::GetLayersNum()
{
	return layers.size();
}

#pragma endregion

#pragma region Initializer
void CNeuralNetwork::InitializeWeight(va_list& InitializerIDs)
{
	uint32 layerSize = layers.size();
	ASSERT_CRASH(layerSize > 0);

	for (uint32 layerIdx = 0; layerIdx < layerSize; ++layerIdx)
	{
		uint32 ActiveID = va_arg(InitializerIDs, uint32);
		std::shared_ptr<CMatrix> weight = layers[layerIdx]->GetWeight();
		switch (ActiveID)
		{
		case NORMAL:
			weight->NormalInitialize(0, 0.1);
			break;
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
				weight->HeNormalInitialize(200);
				// Average : 0, Sigma : 0.1
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
	std::shared_ptr<CLayer2D> calcLayer = layers[0];

	while (calcLayer->latter != nullptr)
	{
		std::shared_ptr<CActFunc>& activatedFunc = calcLayer->GetActivationFunc();
		std::shared_ptr<CMatrix>& activatedInputMatrix = calcLayer->GetActivatedInput();
		std::shared_ptr<CMatrix>& inputMatrix = calcLayer->GetInput();
		std::shared_ptr<CMatrix>& inputWeightMatrix = calcLayer->GetWeight();

		// Activation Function
		activatedFunc->CalcResult(activatedInputMatrix.get(), inputMatrix.get());

		// Matrix Multiplication
		calcLayer->latter->GetInput()->MatMul(activatedInputMatrix.get(), inputWeightMatrix.get());
		calcLayer = calcLayer->latter;
	}

	std::shared_ptr<CLayer2D>& lastLayer = layers[layers.size() - 1];
	std::shared_ptr<CActFunc>& lastActivatedFunc = lastLayer->GetActivationFunc();
	std::shared_ptr<CMatrix>& lastActivatedInputMatrix = lastLayer->GetActivatedInput();
	std::shared_ptr<CMatrix>& lastInputMatrix = lastLayer->GetInput();
	std::shared_ptr<CMatrix>& lastInputWeightMatrix = lastLayer->GetWeight();

	lastActivatedFunc->CalcResult(lastActivatedInputMatrix.get(), lastInputMatrix.get());
	outputMatrix->MatMul(lastActivatedInputMatrix.get(), lastInputWeightMatrix.get());
}

void CNeuralNetwork::BackwardPropagation(const double& learningRate)
{
	ASSERT_CRASH(layers.size());
	ASSERT_CRASH(lossGradient != nullptr);
	std::shared_ptr<CLayer2D> calcLayer = layers[layers.size() - 1];

	while (calcLayer != nullptr)
	{
		std::shared_ptr<CMatrix>& inputMatrix = calcLayer->GetInput();
		std::shared_ptr<CMatrix>& weightMatrix = calcLayer->GetWeight();
		std::shared_ptr<CMatrix>& activatedInputMatrix = calcLayer->GetActivatedInput();
		std::shared_ptr<CMatrix>& gradientMatrix = calcLayer->GetGradient();

		std::shared_ptr<CMatrix>& derivativeActFunc = calcLayer->GetDerivativeActFunc();
		std::shared_ptr<CMatrix>& transposedWeight = calcLayer->GetTransposedWeight();
		std::shared_ptr<CMatrix>& activatedTransposedInput = calcLayer->GetActivatedTransposedInput();
		std::shared_ptr<CMatrix>& weightGradient = calcLayer->GetWeightGradient();

		std::shared_ptr<CActFunc>& calcActFunc = calcLayer->GetActivationFunc();

		calcActFunc->CalcDeriviate(derivativeActFunc.get(), inputMatrix.get());
		transposedWeight->Transpose(weightMatrix.get());
		activatedTransposedInput->Transpose(activatedInputMatrix.get());

		if (calcLayer->latter == nullptr)
		{
			gradientMatrix->MatMul(lossGradient.get(), transposedWeight.get());
			weightGradient->MatMul(activatedTransposedInput.get(), lossGradient.get());
		}
		else
		{
			gradientMatrix->MatMul(calcLayer->latter->GetGradient().get(), transposedWeight.get());
			weightGradient->MatMul(activatedTransposedInput.get(), calcLayer->latter->GetGradient().get());
			gradientMatrix->ElementWiseMul(derivativeActFunc.get(), gradientMatrix.get());
		}
		
		// Updating
		weightGradient->ConstantMul(learningRate);
		weightMatrix->Subtract(weightGradient.get());

		calcLayer = calcLayer->former;
	}
}


#pragma endregion
