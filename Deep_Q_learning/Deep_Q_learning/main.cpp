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
		Node input{ {0, 0, 1, 1}, {0, 1, 0, 1} };
		Node label{ {0, 1, 1, 0} };
		Node weight1(128, 2);
		Node weight2(64, 128);
		Node weight3(1, 64);
		
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

		nn.setOptimizer(GRADIENT_DESCENT, 0.9);

		nn.forwardPropagate(input);
		std::cout << "Input data" << std::endl;
		nn.getInput();
		std::cout << "Output data" << std::endl;
		nn.getOutput();
		nn.backwardPropagate(label);
	}


}