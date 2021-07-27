#pragma once
#include <random>

class Function;

class Variable
{
private:
	int row, col;
	float* data;
	float* grad;
	Function* creator;

public:
	Variable() : row(0), col(0), data(nullptr), grad(nullptr), creator(nullptr) {}
	Variable(const int& _row, const int& _col, Function* f) : row(_row), col(_col), data(new float[row * col]), grad(new float[row * col]), creator(f) {}
	Variable(const std::initializer_list<std::initializer_list<float>>& init_list) : row(0), col(0), data(nullptr), grad(nullptr), creator(nullptr)
	// initializer_list constructor for test input data
	{
		row = init_list.size();
		for (auto& lst : init_list)
		{
			if (lst.size() > col)
			{
				col = lst.size();
			}
		}
		data = new float[row * col];

		int temp_row = 0;
		for (auto& lst : init_list)
		{
			int temp_col = 0;
			for (auto& ls : lst)
			{
				data[temp_row * col + temp_col] = static_cast<float>(ls);
				temp_col += 1;
			}
			while (temp_col != col)
			{
				data[temp_row * col + temp_col] = 0.0f;
				temp_col += 1;
			}
			temp_row += 1;
		}
	}

	Variable(const std::initializer_list<float>& init_list) : row(1), col(0), data(nullptr), grad(nullptr), creator(nullptr)
		// initializer_list constructor for test input data
	{
		int col = init_list.size();
		data = new float[col];
		int temp_idx = 0;
		for (auto &ls : init_list)
		{
			data[temp_idx] = static_cast<float>(ls);
			temp_idx += 1;
		}
	}

	void reset()
	{
		row = 0;
		col = 0;
		if (data != nullptr) {
			delete[] data;
			data = nullptr;
		}
		if (grad != nullptr) {
			delete[] grad;
			grad = nullptr;
		}
		creator = nullptr;
	}

	void init_weight(const int& _row, const int& _col)
	{
		reset();
		row = _row;
		col = _col;
		data = new float[row * col];
		grad = new float[row * col] {0};
		std::random_device rd;
		std::mt19937 mt(rd());
		std::normal_distribution<float> gaussian(0, static_cast<float>(std::sqrt(1.0 / col)));
		for (int i = 0; i < row * col; i += 1)
		{
			data[i] = gaussian(mt);
		}
	}

	void print()
	{
		for (int r = 0; r < row; r += 1)
		{
			for (int c = 0; c < col; c += 1)
			{
				std::cout << data[r * col + c] << " ";
			}
			std::cout << std::endl;
		}
	}


	~Variable()
	{
		if (data != nullptr)
			delete[] data;
		if (grad != nullptr)
			delete[] grad;
	}
};