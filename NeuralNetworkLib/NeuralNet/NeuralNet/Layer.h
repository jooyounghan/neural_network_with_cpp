#pragma once
#include "Neuron.h"

struct Dim2D
{
	int32 row;
	int32 col;
	int32 totalCount;
	Dim2D(const int32 row_in, const int32 col_in);
	Dim2D(const Dim2D& dim);
};

class Layer2D
{
public:
	Dim2D dimension;
	std::vector<Neuron> values;
	Layer2D* in;
	Layer2D* out;
	
public:
	Layer2D(const Dim2D& dim);
	void ConnectTo(Layer2D& layer);
};
