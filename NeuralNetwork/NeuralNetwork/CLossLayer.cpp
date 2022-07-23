#include "pch.h"
#include "CLossLayer.h"

#include "CMatrix.h"

CLossLayer::CLossLayer() : CLayer() {}

CLossLayer::~CLossLayer()
{
}

void CLossLayer::SetLabel(CMatrix* label)
{
	DELETEPTR(labelMatrix);
	labelMatrix = label;
}
