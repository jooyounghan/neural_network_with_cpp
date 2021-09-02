#pragma once
#include <random>
#include <assert.h>
#include "constatns.h"

class Variable
{
public:
	int row, col;
	int size;
	float* data;

public:
	Variable() : row(0), col(0), size(0), data(nullptr) {}
	Variable(const int& _row, const int& _col) : row(_row), col(_col), size(_row * _col), data(new float[size]) {}
	Variable(const int& _row, const int& _col, float*& _data) : row(_row), col(_col), size(_row * _col), data(_data) {
		_data = nullptr;
	}

	Variable(Variable&& v_in) : row(v_in.row), col(v_in.col), size(v_in.row * v_in.col), data(v_in.data) {
		v_in.data = nullptr;
	}

	Variable(const std::initializer_list<std::initializer_list<float>>& init_list) : row(0), col(0), data(nullptr) {
		// initializer_list constructor for test input data
		row = init_list.size();
		for (auto& lst : init_list) {
			if (lst.size() > col) {
				col = lst.size();
			}
		}
		size = row * col;
		data = new float[size];

		int temp_row = 0;
		for (auto& lst : init_list) {
			int temp_col = 0;
			for (auto& ls : lst) {
				data[temp_row * col + temp_col] = float(ls);
				temp_col += 1;
			}
			while (temp_col != col) {
				data[temp_row * col + temp_col] = float(0);
				temp_col += 1;
			}
			temp_row += 1;
		}
	}
	
	void reset() {
		if (data != nullptr) {
			delete[] data;
			data = nullptr;
		}
	}

	void He_initialization(const int& _row, const int& _col) {
		reset();
		row = _row;
		col = _col;
		size = row * col;
		data = new float[size];
		std::random_device rd;
		std::mt19937 mt(rd());
		std::normal_distribution<float> gaussian(0, static_cast<float>(std::sqrt(2.0 / row)));
		for (int i = 0; i < row * col; i += 1) {
			data[i] = gaussian(mt);
		}
	}

	void Xavier_initialization(const int& _row, const int& _col) {
		reset();
		row = _row;
		col = _col;
		size = row * col;
		data = new float[size];
		std::random_device rd;
		std::mt19937 mt(rd());
		std::normal_distribution<float> gaussian(0, static_cast<float>(std::sqrt(2.0 / row + col)));
		for (int i = 0; i < row * col; i += 1) {
			data[i] = gaussian(mt);
		}
	}

	void init_to_easy_weight(const int& _row, const int& _col) {
		reset();
		row = _row;
		col = _col;
		size = row * col;
		data = new float[size];
		for (int i = 0; i < row * col; i += 1) {
			data[i] = i;
		}
	}

	void copy(const Variable& v_in) {
		reset();
		row = v_in.row;
		col = v_in.col;
		size = this->row * this->col;
		data = new float[size];
		float* const input_data = v_in.data;
		for (int i = 0; i < size; ++i) {
			data[i] = input_data[i];
		}
	}

	float sum() {
		float total_sum = 0;
		for (int i = 0; i < size; ++i) {
			total_sum += std::abs(data[i]);
		}
		return total_sum;
	}

	void print(){
		for (int r = 0; r < row; ++r) {
			for (int c = 0; c < col; ++c) {
				std::cout << data[r * col + c] << " ";
			}
			std::cout << std::endl;
		}
	}

	Variable transpose() {
		float* result_data = new float[size];
		for (int r = 0; r < row; ++r) {
			for (int c = 0; c < col; ++c) {
				result_data[c * row + r] = data[r * col + c];
			}
		}
		return Variable(col, row, result_data);
	}

	Variable matmul(const Variable& v_in) {
		int row_size = row;
		int col_size = v_in.col;
		assert(col == v_in.row && "matrix multipication index error");
		int k_size = col;
		float* v_in_data = v_in.data;
		float* result_data = new float[row_size * col_size];
		for (int r = 0; r < row_size; ++r) {
			for (int c = 0; c < col_size; ++c) {
				result_data[r * col_size + c] = 0;
				for (int k = 0; k < k_size; ++k) {
					result_data[r * col_size + c] += data[k_size * r + k] * v_in_data[col_size * k + c];
				}
			}
		}
		return Variable(row_size, col_size, result_data);
	}

	Variable element_mul (Variable& v_in) {
		float* result_data = new float[v_in.size];
		float* const& v_in_data = v_in.data;
		int& v_in_row = v_in.row;
		int& v_in_col = v_in.col;
		if (row == v_in_row && col == v_in_col) {
			for (int r = 0; r < row; ++r) {
				for (int c = 0; c < col; ++c) {
					result_data[r * col + c] = data[r * col + c] * v_in_data[r * col + c];
				}
			}
			return Variable(row, col, result_data);
		}
		if (col == 1 && row == v_in_row) {
			for (int r = 0; r < v_in_row; ++r) {
				for (int c = 0; c < v_in_col; ++c) {
					result_data[r * v_in_col + c] = data[r] * v_in_data[r * v_in_col + c];
				}
			}
			return Variable(v_in_row, v_in_col, result_data);
		}
		assert(false && "can't elementwise multiply each other (N X 1 and N X M or N X M and N X M case only accepted)");
	}

	void move(Variable& v_in) {
		row = v_in.row;
		col = v_in.col;
		size = v_in.size;
		if (data != nullptr) { delete[] data; }
		data = v_in.data;
		v_in.data = nullptr;
	}


	Variable& operator = (Variable& v_in) {
		reset();
		data = new float[size];
		float* const& v_in_data = v_in.data;
		for (int r = 0; r < row; ++r) {
			for (int c = 0; c < col; ++c) {
				data[r * col + c] = v_in_data[r * col + c];
			}
		}
		return *this;
	}

	Variable& operator = (Variable&& v_in) {
		reset();
		row = v_in.row;
		col = v_in.col;
		size = v_in.size;
		data = v_in.data;
		v_in.data = nullptr;
		return *this;
	}

	Variable operator - (const Variable& v_in) {
		assert((row == v_in.row && col == v_in.col) && "for elementwise subtraction, row and column have to be same");
		float* result_data = new float[size];
		float* const& v_in_data = v_in.data;
		for (int r = 0; r < row; ++r) {
			for (int c = 0; c < col; ++c) {
				result_data[r * col + c] = data[r * col + c] - v_in_data[r * col + c];
			}
		}
		return Variable(row, col, result_data);
	}

	Variable operator + (const Variable& v_in) {
		assert((row == v_in.row && col == v_in.col) && "for elementwise plus, row and column have to be same");
		float* result_data = new float[size];
		float* const& v_in_data = v_in.data;
		for (int r = 0; r < row; ++r) {
			for (int c = 0; c < col; ++c) {
				result_data[r * col + c] = data[r * col + c] + v_in_data[r * col + c];
			}
		}
		return Variable(row, col, result_data);
	}

	Variable operator * (const float& f_in) {
		float* result_data = new float[size];
		for (int r = 0; r < row; ++r) {
			for (int c = 0; c < col; ++c) {
				result_data[r * col + c] = data[r * col + c] * f_in;
			}
		}
		return Variable(row, col, result_data);
	}

	Variable log_mat() {
		float* result_data = new float[this->size];
		int& row = this->row;
		int& col = this->col;
		float*& data = this->data;
		for (int r = 0; r < row; ++r) {
			for (int c = 0; c < col; ++c) {
				result_data[r * col + c] = std::exp(data[r * col + c]);
			}
		}
		return Variable(row, col, result_data);
	}


	friend Variable operator + (const Variable& v_1, const Variable& v_2);
	friend Variable operator - (const Variable& v_in);
	friend Variable operator * (const float& x, const Variable& v_in);

	~Variable() {
		if (data != nullptr) {
			delete[] data;
			data = nullptr;
		}
	}
};


Variable operator * (const float& x, Variable& v_in) {
	float* result_data = new float[v_in.size];
	int row = v_in.row;
	int col = v_in.col;
	float* data = v_in.data;
	for (int r = 0; r < row; ++r) {
		for (int c = 0; c < col; ++c) {
			result_data[r * col + c] = data[r * col + c] * x;
		}
	}
	return Variable(row, col, result_data);
}


Variable operator + (Variable& v_1, Variable& v_2) {
	assert((v_1.row == v_2.row && v_1.col == v_2.col) && "for elementwise plus, row and column have to be same");
	float* result_data = new float[v_1.size];
	int row = v_1.row;
	int col = v_1.col;
	float*& data_1 = v_1.data;
	float*& data_2 = v_2.data;
	for (int r = 0; r < row; ++r) {
		for (int c = 0; c < col; ++c) {
			result_data[r * col + c] = data_1[r * col + c] + data_2[r * col + c];
		}
	}
	return Variable(row, col, result_data);
}

Variable operator - (const Variable& v_in) {
	float* result_data = new float[v_in.size];
	int row = v_in.row;
	int col = v_in.col;
	float* data = v_in.data;
	for (int r = 0; r < row; ++r) {
		for (int c = 0; c < col; ++c) {
			result_data[r * col + c] = -data[r * col + c];
		}
	}
	return Variable(row, col, result_data);
}

