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
}

void CHiddenLayer::SetOutput()
{

}

void CHiddenLayer::ForwardProp()
{

}

CMatrix CHiddenLayer::BackwardProp()
{

}

void CHiddenLayer::InitializeWeight(const UINT& type)
{
	ASSERT_CRASH(input != nullptr);
	const UINT& inputDim = input->GetRow();

	DELETEARRAYPTR(weight);

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
