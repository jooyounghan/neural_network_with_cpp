#pragma once
#include "Variable.h"

class Function
{
	/*
		Function Class with Function* next for forward(), and Function* back for backward()
	*/
public :
	Function* next = nullptr;
	Function* back = nullptr;
	virtual Variable forward(Variable& _v) = 0;
	//virtual void backward() = 0;
	virtual std::string function_name() = 0;
};

class DotProduct : public Function
{
public :
	Variable* w;
	Variable* b;

public : 
	DotProduct(Variable* _w, Variable* _b = nullptr) : w(_w), b(_b) {}

	//DotProduct(const int& row, const int& col, Variable* _w, Variable* _b)
	//	: w(new Variable(row, col, this)), b(new Variable(row, col, this)) {}

	virtual Variable forward(Variable& _v) override { return Variable{}; }
	//virtual void backward() override {}

	virtual std::string function_name() override {
		return "DotProduct";
	}
};

class Sum : public Function
{
public:
	Sum(){}
	virtual Variable forward(Variable& _v) override { return Variable{}; }
	//virtual void backward() override {}
	virtual std::string function_name() override {
		return "Sum";
	}
};

namespace Activation
{
	class Relu : public Function
	{
	public:
		Relu(){}
		virtual Variable forward(Variable& _v) override { return Variable{}; }
		//virtual void backward() override {}
		virtual std::string function_name() override {
			return "Relu Activation";
		}
	};

	class Sigmoid : public Function
	{
	public:
		Sigmoid() {}
		virtual Variable forward(Variable& _v) override { return Variable{}; }
		//virtual void backward() override {}
		virtual std::string function_name() override {
			return "Sigmoid Activation";
		}
	};
}

//Variable& linked_forward(Function*& f, Variable& input)
//{
//	input = f->forward(input);
//	if (f->next == nullptr) {
//		return input;
//	}
//	return linked_forward(f->next, input);
//}

// linked_forward���� �� �Լ��� ���� ������� ��� ó���� ������ Ȯ��
// �� ������ �����Ͱ� ����Ǿ�� �ϴ��� ���� Ȯ���Ѵ�
// ���� �����Ͱ� ������ ����Ǿ�� �ϸ�, �Լ� ���ο� ���� ���� ��� Ȯ���غ���
