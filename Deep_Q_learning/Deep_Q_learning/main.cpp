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
		Node weight1(6, 128);
		Node weight2(128, 64);
		Node weight3(64, 3);
		
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

		nn.checkLayer();
		nn.link(l1);



		//timer t1;
		//timer t2;

		//Node node1(128, 64);
		//node1.heInitialize(node1);

		//Node node2(64, 32);
		//node2.heInitialize(node2);

		//Node node3(128, 32);
		//Node node4(128, 32);

		//t1.checkStart();
		//node3.naiveMatMul(node1, node2);
		//t1.checkEnd();
		//t1.elasped("naiveMatMul task");

		//t2.checkStart();
		//node4.matMul(node1, node2);
		//t2.checkEnd();
		//t2.elasped("asyncMatMul task");

		//timer t3;
		//timer t4;

		//bool flag1, flag2;

		//t3.checkStart();
		//flag1= node3.naiveIsSame(node4);
		//t3.checkEnd();
		//t3.elasped("naiveIsSame task");

		//t4.checkStart();
		//flag2 = node3.isSame(node4);
		//t4.checkEnd();
		//t4.elasped("asyncIsSame task");
		//
		//std::cout << flag1 << flag2 << std::endl;
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