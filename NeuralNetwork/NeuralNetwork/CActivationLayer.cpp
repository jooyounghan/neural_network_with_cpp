#include "pch.h"
#include "CActivationLayer.h"

#include "CMatrix.h"

CActivationLayer::CActivationLayer() : CLayer() {}

CActivationLayer::~CActivationLayer()
{
}

void CActivationLayer::SetOutput()
{
	ASSERT_CRASH(input != nullptr);
	output = std::make_shared<CMatrix>(input->GetRow(), input->GetCol());
}
