#include <iostream>

constexpr int NUM_NODE = 3;	// (X, Y)의 좌표계를 나타내는 input과 편향(bias)를 위한 input
constexpr int NUM_DATA = 4;
constexpr float LR = 0.1;
constexpr float ITERS = 10;


float dot(float*& _weight, float* _input, const int& _length)
{
	float result = 0;
	for (int i = 0; i < _length; i += 1)
	{
		result += _weight[i] * _input[i];
	}
	return result;
}

int step(const float& _result)
{
	return (_result > 0) ? 1 : 0; // 삼항 연산자
}

void train(float*& _weight, float* _input, const int& _length, const float& _label, const float& _lr)
{
	float _tlr = (_label - step(dot(_weight, _input, _length))) * _lr;
	for (int i = 0; i < _length; i += 1)
	{
		_weight[i] += _tlr * _input[i];
	}
}


int main()
{
	float* weight = new float[NUM_NODE]{0}; 
	float input[NUM_DATA][NUM_NODE] = { {1, 0, 0}, {1, 1, 0}, {1, 0, 1}, {1, 1, 1} };
	float or_label[NUM_DATA] = { 0, 1, 1, 1 };
	float and_label[NUM_DATA] = { 0, 0, 0, 1 };

	std::cout << "OR OPERATOR With Single Layer Perceptron" << std::endl;
	for (int iter = 0; iter < ITERS; iter += 1)
	{
		for (int data_idx = 0; data_idx < NUM_DATA; data_idx += 1)
		{
			train(weight, input[data_idx], NUM_NODE, or_label[data_idx], LR);
		}
		std::cout << "iteration : " << iter + 1 << " / ";
		for (int w = 0; w < NUM_NODE; w += 1)
		{
			std::cout << weight[w] << " ";
		}
		std::cout << std::endl;
	}
	for (int idx = 0; idx < NUM_DATA; idx += 1)
	{
		std::cout << step(dot(weight, input[idx], NUM_NODE)) << " ";
	}
	std::cout << std::endl;
	delete[] weight;


	std::cout << "AND OPERATOR With Single Layer Perceptron" << std::endl;
	weight = new float[NUM_NODE] {0};

	for (int iter = 0; iter < ITERS; iter += 1)
	{
		for (int data_idx = 0; data_idx < NUM_DATA; data_idx += 1)
		{
			train(weight, input[data_idx], NUM_NODE, and_label[data_idx], LR);
		}
		std::cout << "iteration : " << iter + 1 << " / ";
		for (int w = 0; w < NUM_NODE; w += 1)
		{
			std::cout << weight[w] << " ";
		}
		std::cout << std::endl;
	}

	for (int idx = 0; idx < NUM_DATA; idx += 1)
	{
		std::cout << step(dot(weight, input[idx], NUM_NODE)) << " ";
	}
	std::cout << std::endl;
	delete[] weight;

}