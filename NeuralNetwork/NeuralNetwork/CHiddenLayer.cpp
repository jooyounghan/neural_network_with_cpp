#include "pch.h"
#include "CHiddenLayer.h"

#include "CInitializer.h"
#include "CMatrix.h"


CHiddenLayer::CHiddenLayer(const UINT outputDim)
	: outputDim(outputDim), weight(nullptr), CLayer()
{
}

CHiddenLayer::~CHiddenLayer()
{
	DELETEPTR(weight);
}

void CHiddenLayer::SetOutput()
{
	ASSERT_CRASH(input != nullptr);
	output = std::make_shared<CMatrix>(outputDim, input->GetCol());
}

void CHiddenLayer::ForwardProp()
{
	ASSERT_CRASH(input != nullptr);
	ASSERT_CRASH(output != nullptr);
	input->RefMatMul(*weight, *output);
}

CMatrix CHiddenLayer::BackwardProp(CLayer* layer)
{
	ASSERT_CRASH(input != nullptr);
	ASSERT_CRASH(weight != nullptr)
	CMatrix& postLayerGradient = *(layer->GetGradient());
	CMatrix& weightMatrix = *(weight);

	return weightMatrix.Transpose() * input->Transpose();
}

CMatrix* CHiddenLayer::GetWeight()
{
	return weight;
}

CMatrix CHiddenLayer::GetWeightGradient(CLayer* layer)
{
	ASSERT_CRASH(input != nullptr);
	ASSERT_CRASH(weight != nullptr)
	CMatrix& postLayerGradient = *(layer->GetGradient());
	return postLayerGradient * input->Transpose();
}

void CHiddenLayer::InitializeWeight(CInitializer::INITTYPE type)
{
	ASSERT_CRASH(input != nullptr);
	const UINT& inputDim = input->GetRow();

	DELETEPTR(weight);
	weight = new CMatrix{ outputDim, inputDim };

	switch (type)
	{
	case CInitializer::INITTYPE::NORMAL:
		CInitializer::NormalInitialize(*weight);
		break;
	case CInitializer::INITTYPE::HE:
		CInitializer::HeInitialize(*weight, input->GetRow());
		break;
	case CInitializer::INITTYPE::XAVIER:
		CInitializer::XavierInitialize(*weight, input->GetCol(), output->GetCol());
		break;
	default:
		break;
	}
	return;
}
