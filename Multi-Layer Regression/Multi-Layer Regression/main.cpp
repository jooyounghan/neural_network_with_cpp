#include "Multi-Layer Perceptron/neural_network.h"
#include "Regression Test/graph_generator.h"
/*
* - Expressions in this Project
* 1. class : PascalCase
* 2. funciton : camelCase
* 3. varialbe : snake_case
*/

int main() {

	graph_generator graph_gen;

	function_generator label_func;
	label_func.setXRange(-10, 10, 100);
	label_func.setYRange([](const float& num_in)->const float& {return std::cosf(num_in) * num_in; });
	graph_gen.setLabelData(label_func);

	function_generator noised_func;
	noised_func.setXRange(-10, 10, 100);
	noised_func.setYRange([](const float& num_in)->const float& {return std::cosf(num_in) * num_in; });
	noised_func.makeNoise(-2, 2);
	graph_gen.setNoisedData(noised_func);

	NeuralNetwork nn;

	Node weight1(64, 2);
	Node weight2(128, 64);
	Node weight3(512, 128);

	HiddenLayer l1(weight1);
	HiddenLayer l2(weight2);
	HiddenLayer l3(weight3);

	Relu act1;
	Relu act2;

	nn.link(l1);
	nn.link(act1);
	nn.link(l2);
	nn.link(act2);
	nn.link(l3);

	nn.setInputSize(2, batch_size);
	nn.setOptimizer(GRADIENT_DESCENT, 0.001, 0.9);
	nn.weightInitialize(XAVIER_INITIALIZE);

	std::pair<std::vector<float>, std::vector<float>> shuffled_data = noised_func.getShuffledData();
	nn.naiveTrain(100, shuffled_data.first, shuffled_data.second, batch_size);

	graph_gen.draw();

	
}