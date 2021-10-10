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
		Node weight1(64, 2);
		Node weight2(128, 64);
		Node weight3(512, 128);
		Node weight4(128, 512);
		Node weight5(64, 128);
		Node weight6(1, 64);
		
		HiddenLayer l1(weight1);
		HiddenLayer l2(weight2);
		HiddenLayer l3(weight3);
		HiddenLayer l4(weight4);
		HiddenLayer l5(weight5);
		HiddenLayer l6(weight6);

		Relu act1;
		Relu act2;
		Relu act3;
		Relu act4;
		Relu act5;

		NeuralNetwork nn;
		nn.link(l1);
		nn.link(act1);
		nn.link(l2);
		nn.link(act2);
		nn.link(l3);
		nn.link(act3);
		nn.link(l4);
		nn.link(act4);
		nn.link(l5);
		nn.link(act5);
		nn.link(l6);

		nn.setInput(input);
		nn.setOptimizer(GRADIENT_DESCENT, 0.001, 0.9);
		nn.weightInitialize(XAVIER_INITIALIZE);

		int iterations = 1;
		timer time;
		time.checkStart();
		while (iterations < 100) {
			input = { {0, 0, 1, 1}, {0, 1, 0, 1} };
			if (iterations % 10 == 0) {
				time.checkEnd();
				time.elasped("100 time spent : ");
				std::cout << iterations << std::endl;
				time.checkStart();
			}
			nn.setInput(input);
			nn.forwardPropagate();
			nn.backwardPropagate(label);
			iterations++;
		}
		std::cout << "Output data" << std::endl;
		nn.getOutput();
	}


}