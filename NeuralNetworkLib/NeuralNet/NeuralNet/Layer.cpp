#include "pch.h"
#include "Layer.h"

Dim2D::Dim2D(const int32 row_in, const int32 col_in)
	: row(row_in), col(col_in), totalCount(row_in* col_in)
{}

Dim2D::Dim2D(const Dim2D& dim)
	: row(dim.row), col(dim.col), totalCount(dim.totalCount)
{}

Layer2D::Layer2D(const Dim2D& dim)
	: dimension(dim), in(nullptr), out(nullptr)
{
	values.resize(dimension.totalCount);
}

void Layer2D::ConnectTo(Layer2D& layer)
{
	out = &layer;
	layer.in = this;
}

