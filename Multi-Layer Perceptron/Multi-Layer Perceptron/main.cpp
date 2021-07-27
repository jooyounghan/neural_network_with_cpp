#include <iostream>
#include "Model.h"

constexpr int NUM_DATA = 4;
constexpr int NUM_NODE1 = 8;

constexpr float lr = 0.1;
constexpr int iters = 10;

int main()
{
	Variable weight1, weight2;
	weight1.init_weight(NUM_DATA, NUM_NODE1);
	weight2.init_weight(NUM_NODE1, NUM_DATA);
	weight1.print();
	weight2.print();
	Variable input = { {1, 0, 0}, {1, 1, 0}, {1, 0, 1}, {1, 1, 1} };
	Variable or_label = { 0, 1, 1, 1 };
	Variable and_label = { 0, 0, 0, 1 };
	Variable xor_label = { 0, 1, 1, 0 };

	Model study_model;
	study_model.addFunc(new DotProduct(NUM_DATA, NUM_NODE1));
	study_model.addFunc(new Activation::Relu());
	study_model.addFunc(new DotProduct(NUM_NODE1, NUM_DATA));
	study_model.addFunc(new Activation::Relu());
	study_model.addFunc(new Sum());

	//study_model.train(, )
}