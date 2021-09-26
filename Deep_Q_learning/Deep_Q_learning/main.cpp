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
		Node weight3(64, 3);
		
		weight1.heInitialize(input);
		weight2.heInitialize(weight1);
		weight3.heInitialize(weight2);

		Optimizer opt;
		HiddenLayer l1(weight1, &opt);
		HiddenLayer l2(weight2, &opt);
		HiddenLayer l3(weight3, &opt);
		Relu act1;
		Relu act2;

		NeuralNetwork nn;
		nn.link(l1);
		nn.link(act1);
		nn.link(l2);
		nn.link(act2);
		nn.link(l3);

		nn.forwardPropagate(input);
		nn.getOutput();

	}

	//Environment env;
	//AI ai;

	//while (true) {
	//	ai.setInput(env.getCrurrentState());

	//	env.processInput(ai.getActions());

	//	float reward;

	//	int flag = true;
	//	env.update(0.01, reward, flag);

	//	ai.train(reward);
	//}

}