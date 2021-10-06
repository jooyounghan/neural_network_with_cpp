#include "environment.h"
#include "agent.h"
#include "timer.h"
/*
* - Expressions in this Project
* 1. class : PascalCase
* 2. funciton : camelCase
* 3. varialbe : snake_case
*/

int main() {

	while (true) {
		Node input(1, 6);
		for (int i = 0; i < 6; ++i) {
			input.node[i] = i;
		}

		Node weight1(6, 128);
		Node weight2(128, 64);
		Node weight3(64, 1);
		
		weight1.heInitialize(input);
		weight2.heInitialize(weight1);
		weight3.heInitialize(weight2);

		HiddenLayer l1(weight1);
		HiddenLayer l2(weight2);
		HiddenLayer l3(weight3);
		Relu act1;
		Relu act2;

		NeuralNetwork nn;
		nn.link(l1);
		nn.link(act1);
		nn.link(l2);
		nn.link(act2);
		nn.link(l3);

		nn.setOptimizer(GRADIENT_DESCENT);

		nn.forwardPropagate(input);
		std::cout << "Input data" << std::endl;
		nn.getInput();
		std::cout << "Output data" << std::endl;
		nn.getOutput();


	}


}