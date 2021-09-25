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
		timer t1;

		t1.checkStart();
		Node node1(64, 128);
		node1.heInitialize(node1);
		t1.checkEnd();
		t1.elasped();

		t1.checkStart();
		Node node2(64, 128);
		node2.naive_heInitialize(node2);
		t1.checkEnd();
		t1.elasped();
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