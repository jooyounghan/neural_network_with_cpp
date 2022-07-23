#include "pch.h"
#include "CLayer.h"

CLayer::CLayer() : input(nullptr), output(nullptr) {}

// TODO : ¼Ò¸êÀÚ¿¡ ´ëÇÑ °í¹Î ÇÊ¿ä
CLayer::~CLayer()
{
}

void CLayer::SetInput(CMatrix* newInput)
{
	DELETEPTR(input);
	input = newInput;
}

void CLayer::SetGradient(CMatrix* newGradient)
{
	DELETEPTR(gradient);
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

CMatrix* CLayer::GetInput()
{
	return input;
}

CMatrix* CLayer::GetOutput()
{
	return output;
}

CMatrix* CLayer::GetGradient() { return gradient; }


