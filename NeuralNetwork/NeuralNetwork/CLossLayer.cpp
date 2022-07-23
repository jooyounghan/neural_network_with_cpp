#include "pch.h"
#include "CLossLayer.h"

#include "CMatrix.h"

CLossLayer::CLossLayer() : CLayer() {}

CLossLayer::~CLossLayer()
{
	DELETEPTR(labelMatrix);
}

void CLossLayer::SetLabel(CMatrix* label)
{
	DELETEPTR(labelMatrix);
	labelMatrix = label;
}

CMatrix* CLossLayer::GetLabel()
{
	return labelMatrix;
}
