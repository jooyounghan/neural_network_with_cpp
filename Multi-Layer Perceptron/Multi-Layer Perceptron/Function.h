#pragma once
#include "Variable.h"

class Function
{
public :
	//virtual Variable forward() = 0;
	//virtual void backward() = 0;
};

class DotProduct : public Function
{
public :
	Variable* w;
	Variable* b;

public : 
	DotProduct(Variable* _w, Variable* _b) : w(_w), b(_b) {}
	DotProduct(const int& row, const int& col) : w(new Variable(row, col, this)), b(new Variable(row, col, this)) {}

	//virtual Variable forward() override {}
	//virtual void backward() override {}
};

class Sum : public Function
{
public:
	Sum(){}
	//virtual Variable forward() override {}
	//virtual void backward() override {}

};

namespace Activation
{
	class Relu : public Function
	{
	public:
		Relu(){}
		//virtual Variable forward() override { return ; }
		//virtual void backward() override {}
	};

	class Sigmoid : public Function
	{
	public:
		Sigmoid() {}
		//virtual Variable forward() override {}
		//virtual void backward() override {}
	};
}

