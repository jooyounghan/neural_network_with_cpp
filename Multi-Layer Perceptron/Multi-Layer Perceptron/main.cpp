#include <iostream>
#include "Model.h"

constexpr int NUM_DATA = 2;
constexpr int NUM_NODE1 = 4;

constexpr float lr = 0.1f;
constexpr int iters = 10;

int main()
{
	Variable weight1, weight2;
	weight1.init_to_easy_weight(NUM_NODE1, NUM_DATA);
	weight2.init_to_easy_weight(NUM_DATA, NUM_NODE1);
	weight1.print();
	std::cout << "+++++++++++++++++++" << std::endl;
	weight2.print();

	Model study_model;
	study_model.addFunc(new Multiplication(weight1));
	study_model.addFunc(new Activation::Relu());
	study_model.addFunc(new Multiplication(weight2));
	study_model.addFunc(new Activation::Sigmoid());
	study_model.addFunc(new Activation::Relu());

	std::cout << std::endl;
	//linking test
	for (auto& f : study_model.funcQueue)
	{
		std::cout << f->function_name() << " / " << f->back << " " << f << " " << f->next << std::endl;
	}

	//study_model.train(input, or_label, NUM_DATA, iters, lr);
	Variable input = Variable{ {1, 1}, {2, 2} };
	study_model.train(input);
}