#pragma once

#define PASS (void)0
#include "Variable.h"
#include "math.h"

class Function
{
	/*
		Function Class with Function* next for forward() (forward Propagation),
		and Function* back for backward() (Back Propagation)
	*/

public :
	Function* next = nullptr;
	Function* back = nullptr;
	virtual Variable forward(const Variable& _v) = 0;
	virtual void backward(const Variable& _v, const float& lr) = 0;
	virtual std::string function_name() = 0;
	virtual void reset() = 0;
};

class Multiplication : public Function
{
public :
	Variable& w;

public : 
	Multiplication(Variable& _w) : w(_w) {}

	virtual Variable forward(const Variable& v_input) override {
		float*& w_data = w.data;
		float* const& input_data = v_input.data;
		int row_size = w.row;
		int col_size = v_input.col;
		int k_size = w.col;
		float* result_data = new float[row_size * col_size];
		for (int r = 0; r < row_size; ++r) {
			for (int c = 0; c < col_size; ++c) {
				result_data[r * col_size + c] = 0;
				for (int k = 0; k < k_size; ++k) {
					result_data[r * col_size + c] += w_data[k_size * r + k] * input_data[col_size * k + c];
				}
			}
		}

		if (this->next != nullptr) {
			return this->next->forward(Variable(row_size, col_size, result_data));
		}
		return Variable(row_size, col_size, result_data);
	}

	virtual void backward(const Variable& _v, const float& lr) override {
		PASS;
	}

	virtual std::string function_name() override {
		return "Multiplication";
	}

	virtual void reset() override {
		PASS;
	}
};

//class Sum : public Function
//{
//public:
//	Sum(){}
//	virtual void forward(Variable& _v) override { }
//	virtual void backward() override {}
//
//	virtual std::string function_name() override {
//		return "Sum";
//	}
//
//	virtual void reset() override {
//		PASS;
//	}
//};

namespace Activation
{
	class Relu : public Function
	{
	public:
		Variable* grad = nullptr;
		Variable* data = nullptr;
	public:
		Relu(){}

		virtual Variable forward(const Variable& v_input) override {
			if (data == nullptr) {
				data = new Variable();
			}
			data->copy(v_input);
			float*& relu_data = data->data;
			if (grad == nullptr) {
				grad = new Variable(v_input.row, v_input.col);
				float*& grad_data = grad->data;
				for (int i = 0; i < v_input.size; ++i) {
					// Relu Activation Function
					grad_data[i] = relu_data[i] < 0 ? 0 : 1;
				}
			}
			for (int i = 0; i < v_input.size; ++i) {
				// Relu Activation Function
				relu_data[i] = relu_data[i] < 0 ? 0 : relu_data[i];
			}

			if (this->next != nullptr) {
				return this->next->forward(*data);
			}
			return Variable(data->row, data->col, relu_data);
		}

		virtual void backward(const Variable& _v, const float& lr) override {
			PASS;
		}
		
		virtual std::string function_name() override {
			return "Relu Activation";
		}

		virtual void reset() override {
			if (data != nullptr) {
				delete data;
				data = nullptr;
			}
			if (grad != nullptr) {
				delete grad;
				grad = nullptr;
			}
		}
	};

	class Sigmoid : public Function
	{
	public:
		Variable* grad = nullptr;
		Variable* data = nullptr;

	public:
		Sigmoid() {}

		virtual Variable forward(const Variable& v_input) override {
			if (data == nullptr) {
				data = new Variable();
			}
			data->copy(v_input);
			float*& sig_data = data->data;
			if (grad == nullptr) {
				grad = new Variable(v_input.row, v_input.col);
				float*& grad_data = grad->data;
				for (int i = 0; i < v_input.size; ++i) {
					float sigmoid = 1 / (1 + std::exp(-sig_data[i]));
					// Sigmoid Activation Function
					grad_data[i] = sigmoid * (1 - sigmoid);
					sig_data[i] = sigmoid;
				}
			}
			else {
				for (int i = 0; i < v_input.size; ++i) {
					// Sigmoid Activation Function
					float sigmoid = 1 / (1 + std::exp(-sig_data[i]));
					sig_data[i] = sigmoid;
				}
			}

			if (this->next != nullptr) {
				std::cout << "=================================" << std::endl;
				data->print();
				return this->next->forward(*data);
			}
			return Variable(data->row, data->col, sig_data);
		}

		virtual void backward(const Variable& _v, const float& lr) override {
			PASS;
		}

		virtual std::string function_name() override {
			return "Relu Activation";
		}

		virtual void reset() override {
			if (data != nullptr) {
				delete data;
				data = nullptr;
			}
			if (grad != nullptr) {
				delete grad;
				grad = nullptr;
			}
		}
	};
}

