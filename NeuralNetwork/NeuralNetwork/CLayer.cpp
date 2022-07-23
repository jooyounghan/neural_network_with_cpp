#include "pch.h"
#include "CLayer.h"

CLayer::CLayer() : input(nullptr), output(nullptr), gradient(nullptr) {}

// TODO : ¼Ò¸êÀÚ¿¡ ´ëÇÑ °í¹Î ÇÊ¿ä
CLayer::~CLayer()
{
}

void CLayer::SetInput(std::shared_ptr<CMatrix> newInput)
{
	input = newInput;
}

void CLayer::SetGradient(std::shared_ptr<CMatrix> newGradient)
{
	gradient = newGradient;
}

void CLayer::Connect(CLayer* layer)
{
	if (output == nullptr)
	{
		this->SetOutput();
	}
	layer->input = this->output;
}

std::shared_ptr<CMatrix> CLayer::GetInput()
{
	return input;
}

std::shared_ptr<CMatrix> CLayer::GetOutput()
{
	return output;
}

std::shared_ptr<CMatrix> CLayer::GetGradient() { return gradient; }


