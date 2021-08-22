#include <iostream>
#include "Model.h"

int main()
{
	while (true) {
		Variable weight1, weight2, weight3, weight4;
		weight1.init_weight(NUM_NODE1, NUM_INPUT_DATA);
		weight2.init_weight(NUM_NODE2, NUM_NODE1);
		weight3.init_weight(NUM_NODE3, NUM_NODE2);
		weight4.init_weight(NUM_OUPUT_DATA, NUM_NODE3);

		Model study_model;
		study_model.addFunc(new Multiplication(weight1));
		study_model.addFunc(new Activation::Relu());
		study_model.addFunc(new Multiplication(weight2));
		study_model.addFunc(new Activation::Relu());
		study_model.addFunc(new Multiplication(weight3));
		study_model.addFunc(new Activation::Relu());
		study_model.addFunc(new Multiplication(weight4));

		std::cout << "TEST OF THE LINKING STATE OF THE FUNCITON (Backward / Function / Forward)" << std::endl;
		std::cout << std::endl;
		for (auto& f : study_model.funcQueue)
		{
			std::cout << f->function_name() << " / " << f->back << " " << f << " " << f->next << std::endl;
		}
		std::cout << "----------------------------------------------------------" << std::endl;
		std::cout << std::endl;

		std::cout << "Test of the nonlinear Model with XOR" << std::endl;
		Variable input = Variable{ {0, 0, 1, 1}, {0, 1, 0, 1}};
		Variable xor_label = Variable{ {0, 1, 1, 0} };
		study_model.train(input, xor_label, lr, iters);
		std::cout << "----------------------------------------------------------" << std::endl;
		std::cout << std::endl;
	};
}