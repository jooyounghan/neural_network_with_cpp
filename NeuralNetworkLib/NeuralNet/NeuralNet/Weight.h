#pragma once
struct Dim2D;

struct Weight
{
	Dim2D dimension;
	std::vector<double> values;

	Weight(const Dim2D& dim);
};