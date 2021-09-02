#pragma once
#include "Optimizer.h"

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
	virtual Variable forward(Variable& _v) = 0;
	virtual void backward(Variable& _v, const float& lr, Optimizer*& op) = 0;
	virtual std::string function_name() = 0;
};

class Multiplication : public Function
{
public :
	Variable& w;
	Variable gradient;
	Variable* input_data;

public : 
	Multiplication(Variable& _w) : w(_w), input_data(nullptr) {}

	virtual Variable forward(Variable& v_input) override {
		input_data = &v_input;
		float*& w_data = w.data;
		float* const& v_input_data = v_input.data;
		int& row_size = w.row;
		int& col_size = v_input.col;
		int& k_size = w.col;
		float* result_data = new float[row_size * col_size];
		for (int r = 0; r < row_size; ++r) {
			for (int c = 0; c < col_size; ++c) {
				result_data[r * col_size + c] = 0;
				for (int k = 0; k < k_size; ++k) {
					result_data[r * col_size + c] += w_data[k_size * r + k] * v_input_data[col_size * k + c];
				}
			}
		}
		if (this->next != nullptr) {
			Variable param = Variable(row_size, col_size, result_data);
			return this->next->forward(param);
		}
		return Variable(row_size, col_size, result_data);
	}

	virtual void backward(Variable& _v, const float& lr, Optimizer*& op) override {

		Variable present_gradient = _v.matmul(input_data->transpose());
		op->optimize(gradient, present_gradient);
		w = w + gradient;

		if (this->back != nullptr) {
			Variable param = w.transpose().matmul(_v);
			return this->back->backward(param, lr, op);
		}
		return;
	}

	virtual std::string function_name() override {
		return "Multiplication";
	}
};

class Softmax : public Function
{
public:
	Softmax(){}

	virtual Variable forward(Variable& v_input) override {
		float* const& v_input_data = v_input.data;
		int& row_size = v_input.row;
		int& col_size = v_input.col;
		int& input_size = v_input.size;
		float* result_data = new float[v_input.size];
		for (int c = 0; c < col_size; ++c) {
			float temp_sum = 0;
			for (int r = 0; r < row_size; ++r) {
				temp_sum += std::exp(v_input_data[r * col_size + c]);
			}
			for (int r = 0; r < row_size; ++r) {
				result_data[r * col_size + c] = std::exp(v_input_data[r * col_size + c]) / temp_sum;
			}
		}
		if (this->next != nullptr) {
			std::cout << "Softmax funciton has to be put on the output layer" << std::endl;
			return Variable();
		}
		return Variable(row_size, col_size, result_data);
	}

	virtual void backward(Variable& _v, const float& lr, Optimizer*& op) override {
		if (this->back != nullptr) {
			return this->back->backward(_v, lr, op);
		}
		return;
	}

	virtual std::string function_name() override {
		return "Softmax";
	}
};

namespace Activation
{
	class Relu : public Function
	{
	public:
		Variable* grad = nullptr;
		Variable* data = nullptr;
	public:
		Relu(){}

		virtual Variable forward(Variable& v_input) override {
			if (data == nullptr) {
				data = new Variable();
			}
			data->copy(v_input);
			float*& relu_data = data->data;
			if (grad == nullptr) {
				grad = new Variable(v_input.row, v_input.col);
			}
			float*& grad_data = grad->data;
			for (int i = 0; i < v_input.size; ++i) {
				// Relu Activation Function
				relu_data[i] = relu_data[i] < 0 ? 0 : relu_data[i];
				grad_data[i] = relu_data[i] < 0 ? 0 : 1;
			}
			if (this->next != nullptr) {
				return this->next->forward(*data);
			}
			return Variable(data->row, data->col, relu_data);
		}

		virtual void backward(Variable& _v, const float& lr, Optimizer*& op) override {
			if (this->back != nullptr) {
				Variable param = grad->element_mul(_v);
				return this->back->backward(param, lr, op);
			}
			return;
		}
		
		virtual std::string function_name() override {
			return "Relu Activation";
		}

		~Relu() {
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

		virtual Variable forward(Variable& v_input) override {
			if (data == nullptr) {
				data = new Variable();
			}
			data->copy(v_input);
			float*& sig_data = data->data;
			if (grad == nullptr) {
				grad = new Variable(v_input.row, v_input.col);
			}
			float*& grad_data = grad->data;
			for (int i = 0; i < v_input.size; ++i) {
				float sigmoid = 1 / (1 + std::exp(-sig_data[i]));
				// Sigmoid Activation Function
				sig_data[i] = sigmoid;
				grad_data[i] = sigmoid * (1 - sigmoid);
			}

			if (this->next != nullptr) {
				return this->next->forward(*data);
			}
			return Variable(data->row, data->col, sig_data);
		}


		virtual void backward(Variable& _v, const float& lr, Optimizer*& op) override {
			if (this->back != nullptr) {
				Variable param = grad->element_mul(_v);
				return this->back->backward(param, lr, op);
			}
			return;
		}

		virtual std::string function_name() override {
			return "Relu Activation";
		}

		~Sigmoid() {
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

