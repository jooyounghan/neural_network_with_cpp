#include <iostream>
#include "Model.h"

constexpr int NUM_DATA = 4;
constexpr int NUM_NODE1 = 4;
constexpr int NUM_NODE2 = 4;
constexpr int NUM_INPUT_DATA = 2;
constexpr int NUM_OUPUT_DATA = 1;

constexpr float lr = 0.1f;
constexpr int iters = 10;

int main()
{
	while (true) {
		Variable weight1, weight2, weight3;
		weight1.init_to_easy_weight(NUM_NODE1, NUM_INPUT_DATA);
		weight2.init_to_easy_weight(NUM_NODE2, NUM_NODE1);
		weight3.init_to_easy_weight(NUM_OUPUT_DATA, NUM_NODE2);

		weight1.print();
		std::cout << "+++++++++++++++++++" << std::endl;
		weight2.print();
		std::cout << "+++++++++++++++++++" << std::endl;
		weight3.print();

		Model study_model;
		study_model.addFunc(new Multiplication(weight1));
		study_model.addFunc(new Activation::Relu());
		study_model.addFunc(new Multiplication(weight2));
		study_model.addFunc(new Activation::Relu());
		study_model.addFunc(new Multiplication(weight3));


		std::cout << std::endl;
		//linking test
		for (auto& f : study_model.funcQueue)
		{
			std::cout << f->function_name() << " / " << f->back << " " << f << " " << f->next << std::endl;
		}

		//study_model.train(input, or_label, NUM_DATA, iters, lr);
		Variable input = Variable{ {0, 0}, {0, 1}, {1, 0}, {1, 1} };
		Variable xor_label = Variable{ 0, 1, 1, 0 };
		input.print();
		xor_label.print();

	
		study_model.train(input, xor_label, lr, iters);
	}
}