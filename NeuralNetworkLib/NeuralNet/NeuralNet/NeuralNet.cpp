#include "pch.h"
#include "NeuralNet.h"

NeuralNet::NeuralNet()
{
}

void NeuralNet::SetLayer(int32 layerCount, ...)
{
	this->clear();
	va_list layerLists;
	va_start(layerLists, layerCount);

	for (int32 layerIdx = 0; layerIdx < layerCount; ++layerIdx)
	{
		this->emplace_back(va_arg(layerLists, Dim2D));
	}

	va_end(layerLists);

	for (int32 layerIdx = 0; layerIdx < layerCount - 1; ++layerIdx)
	{
		Layer2D& from = this->at(layerIdx);
		Layer2D& to = this->at(layerIdx + 1);
		from.ConnectTo(to);
		ASSERT_CRASH(from.dimension.row == to.dimension.row);
		weights.push_back(Dim2D(from.dimension.col, to.dimension.col));
	}
}