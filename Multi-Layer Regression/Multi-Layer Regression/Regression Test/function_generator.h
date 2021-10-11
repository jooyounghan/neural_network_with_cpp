#pragma once
#include <random>
#include <vector>
#include <iostream>
#include <functional>

class function_generator {
public:
	float start, end;
	int num_data;
	float step_size;
	std::vector<float> x_data;
	std::vector<float> y_data;

public:
	function_generator()
		: start(0), end(0), num_data(0), step_size(0) {}

	void setXRange(const float& start_in, const float& end_in, const int& steps) {
		start = start_in;
		end = end_in;
		num_data = steps + 1;
		step_size = (end - start) / steps;
		x_data.resize(num_data);
		y_data.resize(num_data);

		for (int i = 0; i < num_data; ++i) {
			x_data[i] = start + step_size * i;
		}
	}

	void setYRange(std::function<const float&(const float&)> func_in) {
		for (int i = 0; i < num_data; ++i) {
			y_data[i] = func_in(x_data[i]);
		}
	}

	void printData() {
		for (int i = 0; i < num_data; ++i) {
			std::cout << x_data[i] << " " << y_data[i] << '\n';
		}
	}

	std::vector<float> getXData() {
		return x_data;
	}

	std::vector<float> getYData() {
		return y_data;
	}

	std::pair<std::vector<float>, std::vector<float>> getShuffledData() {
		std::random_device rd;
		std::mt19937_64 mt(rd());
		int random_idx;
		int temp_x_num;
		int temp_y_num;
		std::vector<float> new_x_data;
		std::vector<float> new_y_data;
		new_x_data.resize(this->x_data.size());
		new_y_data.resize(this->y_data.size());
		std::copy(this->x_data.begin(), this->x_data.end(), new_x_data.begin());
		std::copy(this->y_data.begin(), this->y_data.end(), new_y_data.begin());

		for (int i = 0; i < num_data; ++i) {
			random_idx = rand() % (num_data - i) + i;
			temp_x_num = new_x_data[i];
			temp_y_num = new_y_data[i];
			new_x_data[i] = new_x_data[random_idx];
			new_y_data[i] = new_y_data[random_idx];
			new_x_data[random_idx] = temp_x_num;
			new_y_data[random_idx] = temp_y_num;
		}
		return std::make_pair(new_x_data, new_y_data);
	}

	void makeNoise(const float& min_num, const float& max_num) {
		std::random_device rd;
		std::mt19937_64 mt(rd());
		std::uniform_real_distribution<float> range(min_num, max_num);
		for (int i = 0; i < this->y_data.size(); ++i) {
			y_data[i] += range(mt);
		}
		return;
	}

	std::vector<std::vector<float>> getRandomBatchXData(const int& batch_size) {
		std::vector<std::vector<float>> result;
		result.reserve(x_data.size() / batch_size + 1);
		for (int total_idx = 0; total_idx < x_data.size(); ++total_idx) {
			
		}
	}
};