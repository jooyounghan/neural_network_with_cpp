#include "pch.h"
#include "NNModel.h"

CNNModel::CNNModel(const uint32& inputCnt, const uint32& inputDimension, const uint32& outputDimension, const uint32 numLayers, ...)
	: NeuralNet(nullptr), lossFunc(nullptr), loss(0.0)
{
	va_list argsList;
	va_start(argsList, numLayers);

	NeuralNet = std::make_shared<CNeuralNetwork>();
	NeuralNet->SetLayers(inputDimension, outputDimension, numLayers, argsList);
	NeuralNet->SetInputNum(inputCnt);
	va_end(argsList);
}

CNNModel::~CNNModel()
{
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
	switch (lfID)
	{
	case CLossFunc::LossID::SUMATION:
		ASSERT_CRASH(GetNeuralNetwork()->GetOutputMatrix()->GetCol() == 1)
		lossFunc = std::make_shared<CSumation>();
		break;
	case CLossFunc::LossID::SOFTMAX:
		ASSERT_CRASH(GetNeuralNetwork()->GetOutputMatrix()->GetCol() != 1)
		lossFunc = std::make_shared<CSoftmax>();
		break;
	default:
		break;
	}
}

void CNNModel::SetOptimizer(const OPTIMIZERID& optID)
{
	switch (optID)
	{
	case COptimizer::OptimizerId::GD:
		optimizer = std::make_shared<CGD>();
		break;
	case COptimizer::OptimizerId::NAG:
		optimizer = std::make_shared<CNAG>();
		break;
	case COptimizer::OptimizerId::ADAGRAD:
		optimizer = std::make_shared<CAdagrad>();
		break;
	case COptimizer::OptimizerId::ADAM:
		optimizer = std::make_shared<CAdam>();
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

void CNNModel::Train(std::vector<std::shared_ptr<CMatrix>> inputVector, std::vector<std::shared_ptr<CMatrix>> labelVector, const uint32& iterations, const double& learningRate)
{
	const uint32& inputSize = inputVector.size();
	const uint32& labelSize = labelVector.size();
	ASSERT_CRASH(NeuralNet != nullptr);
	ASSERT_CRASH(inputSize == labelSize);
	ASSERT_CRASH(lossFunc != nullptr);
	ASSERT_CRASH(optimizer != nullptr);

	CRandomGenerator randGen;
	for (uint32 iter = 0; iter < iterations; ++iter)
	{
		// Shuffle with Same Order
		//randGen.ShuffleVector(inputVector, 100);
		//randGen.ShuffleVector(labelVector, 100);
		for (uint32 idx = 0; idx < inputVector.size(); ++idx)
		{
			SYNCINPUT(NeuralNet, inputVector[idx]);
			Propagation(inputVector[idx], labelVector[idx], learningRate);
		}
	}
}

void CNNModel::PushInput(std::shared_ptr<CMatrix> inputMatrix, std::shared_ptr<CMatrix> labelMatrix, const double& learningRate)
{
	ASSERT_CRASH(NeuralNet != nullptr);
	ASSERT_CRASH(lossFunc != nullptr);
	ASSERT_CRASH(optimizer != nullptr);

	SYNCINPUT(NeuralNet, inputMatrix);
	Propagation(inputMatrix, labelMatrix, learningRate);
}

std::shared_ptr<CMatrix> CNNModel::GetResult(std::shared_ptr<CMatrix> inputMatrix)
{
	ASSERT_CRASH(lossFunc.get() != nullptr);

	SYNCINPUT(NeuralNet, inputMatrix);
	NeuralNet->SetInput(inputMatrix);
	NeuralNet->ForwardPropagation();
	std::shared_ptr<CMatrix> ResultMatrix = NeuralNet->GetOutputMatrix();
	lossFunc->GetResult(ResultMatrix.get(), ResultMatrix.get());
	return ResultMatrix;
}

const double& CNNModel::GetLoss()
{
	return loss;
}

std::shared_ptr<CNeuralNetwork> CNNModel::GetNeuralNetwork()
{
	return NeuralNet;
}

void CNNModel::Propagation(std::shared_ptr<CMatrix> inputMatrix, std::shared_ptr<CMatrix> labelMatrix, const double& learningRate)
{
	ASSERT_CRASH(NeuralNet != nullptr);
	ASSERT_CRASH(lossFunc != nullptr);
	ASSERT_CRASH(optimizer != nullptr);

	NeuralNet->SetInput(inputMatrix);
	NeuralNet->ForwardPropagation();

	std::shared_ptr<CMatrix>& outputMatrix = NeuralNet->GetOutputMatrix();
	std::shared_ptr<CMatrix>& lossMatrix = NeuralNet->GetLossGradient();

	lossFunc->GetResult(outputMatrix.get(), outputMatrix.get());
	lossFunc->GetLossGradient(lossMatrix.get(), outputMatrix.get(), labelMatrix.get());

	CLossFunc::GetLoss(loss, NeuralNet->GetOutputMatrix().get(), labelMatrix.get());
	NeuralNet->BackwardPropagation();

	optimizer->Update(NeuralNet, learningRate);
}
