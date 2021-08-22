#pragma once

class Function;

class Variable
{
private : 
	float* data;
	float* grad;
	Function* creator;

public :
	Variable(const int& row, const int& col, Function* f) : data(new float[row * col]), grad(new float[row * col]), creator(f) {}
	
	~Variable()
	{
		delete[] data;
		delete[] grad;
		delete creator;
	}
};