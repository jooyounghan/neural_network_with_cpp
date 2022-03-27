#include "pch.h"
#include "NeuralNet.h"

NeuralNet::NeuralNet()
{
}

void NeuralNet::SetLayer(int32 layerCount, ...)
{
	this->clear();
	va_list nodeNumLists;
	va_start(nodeNumLists, layerCount);

	this->resize(layerCount);
	for (int32 layerIdx = 0; layerIdx < layerCount; ++layerIdx)
	{
		this->at(layerIdx).SetNodeNum(va_arg(nodeNumLists, int32));
	}

	va_end(nodeNumLists);

	for (int32 layerIdx = 0; layerIdx < layerCount - 1; ++layerIdx)
	{
		(this->at(layerIdx)).ConnectTo(this->at(layerIdx + 1));
	}
}

Layer NeuralNet::GetLayer(int32 index)
{
	return this->at(index);
}
