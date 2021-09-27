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
		
		Node n1(3, 3);
		Node n2(3, 2);
		Node n3(3, 2);
		Node n4(3, 2);
		n1.baseInitialize();
		n2.baseInitialize();
		n3.matMul(n1, n2);
		n4.naiveMatMul(n1, n2);
		n1.print();
		n2.print();
		n3.print();
		n4.print();
		Node n5 = n3.transpose();
		n5.print();

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