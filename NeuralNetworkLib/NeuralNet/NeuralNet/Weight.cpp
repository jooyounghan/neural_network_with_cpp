#include "pch.h"
#include "Weight.h"

Weight::Weight(const Dim2D& dim)
	: dimension(dim)
{
	values.resize(dim.totalCount);
}
