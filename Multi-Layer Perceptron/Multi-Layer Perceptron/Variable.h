#pragma once
#include <random>
class Variable
{
public:
	int row, col;
	int size;
	float* data;

public:
	Variable() : row(0), col(0), size(0), data(nullptr) {}
	Variable(const int& _row, const int& _col) : row(_row), col(_col), size(_row * _col), data(new float[size]) {}
	Variable(const std::initializer_list<std::initializer_list<float>>& init_list) : row(0), col(0), data(nullptr)
	// initializer_list constructor for test input data
	{
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

	Variable(const std::initializer_list<float>& init_list) : row(1), col(0), size(0), data(nullptr) {
		// initializer_list constructor for test input data
		int col = init_list.size();
		size = col;
		data = new float[size];
		int temp_idx = 0;
		for (auto &ls : init_list) {
			data[temp_idx] = float(ls);
			temp_idx += 1;
		}
	}
	
	void reset() {
		if (this != nullptr) {
			row = 0;
			col = 0;
			size = 0;
			if (data != nullptr) {
				delete[] data;
				data = nullptr;
			}
		}
	}

	void init_weight(const int& _row, const int& _col) {
		reset();
		row = _row;
		col = _col;
		size = row * col;
		data = new float[size];
		std::random_device rd;
		std::mt19937 mt(rd());
		std::normal_distribution<float> gaussian(0, static_cast<float>(std::sqrt(1.0 / col)));
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

	void copy(Variable& v_in) {
		reset();
		row = v_in.row;
		col = v_in.col;
		size = this->row * this->col;
		data = new float[size];
		float*& input_data = v_in.data;
		for (int i = 0; i < size; ++i) {
			data[i] = input_data[i];
		}
	}

	void print(){
		for (int r = 0; r < row; r += 1) {
			for (int c = 0; c < col; c += 1) {
				std::cout << data[r * col + c] << " ";
			}
			std::cout << std::endl;
		}
	}


	~Variable() {
		if (data != nullptr)
			delete[] data;
	}
};