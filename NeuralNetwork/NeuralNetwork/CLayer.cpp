#include "pch.h"
#include "CLayer.h"
#include "CMatrix.h"

CLayer::CLayer() : input(nullptr), output(nullptr), gradient(nullptr), frontLayer(nullptr), rearLayer(nullptr) {}

// ¼Ò¸êÀÚ¿¡ ´ëÇÑ °í¹Î ÇÊ¿ä1
CLayer::~CLayer() {}

void CLayer::SetInput(CMatrix* newInput)
{
	// Has to be the first Layer
	ASSERT_CRASH(frontLayer == nullptr);
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
	this->rearLayer = layer;
	layer->frontLayer = this;
}

CMatrix* CLayer::GetGradient() { return gradient; }


