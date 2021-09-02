#include <iostream>
#include "Model.h"

int main()
{
	while (true) {
		Variable weight1, weight2, weight3, weight4;
		weight1.Xavier_initialization(NUM_NODE1, NUM_INPUT_DATA);
		weight2.Xavier_initialization(NUM_NODE2, NUM_NODE1);
		weight3.Xavier_initialization(NUM_NODE3, NUM_NODE2);
		weight4.Xavier_initialization(NUM_OUPUT_DATA, NUM_NODE3);

		Model study_model;

		Multiplication node1{ weight1 };
		Activation::Sigmoid active1 = Activation::Sigmoid();
		Multiplication node2{ weight2 };
		Activation::Sigmoid active2 = Activation::Sigmoid();
		Multiplication node3{ weight3 };
		Activation::Sigmoid active3 = Activation::Sigmoid();
		Multiplication node4{ weight4 };

		study_model.addFunc(node1);
		study_model.addFunc(active1);
		study_model.addFunc(node2);
		study_model.addFunc(active2);
		study_model.addFunc(node3);
		study_model.addFunc(active3);
		study_model.addFunc(node4);

		GradientDescent GD_Optimizer(lr);

		study_model.setOptimizer(&GD_Optimizer);


		std::cout << "TEST OF THE LINKING STATE OF THE FUNCITON (Backward / Function / Forward)" << "\n";
		std::cout << "\n";
		for (auto& f : study_model.funcQueue)
		{
			std::cout << f->function_name() << " / " << f->back << " " << f << " " << f->next << "\n";
		}
		std::cout << "----------------------------------------------------------" << "\n";
		std::cout << "\n";

		std::cout << "Test of the nonlinear Model with XOR" << "\n";
		Variable input = Variable{ {0, 0, 1, 1}, {0, 1, 0, 1} };
		Variable xor_label = Variable{ {0, 1, 1, 0} };
		study_model.train(iters, input, xor_label);
		std::cout << "----------------------------------------------------------" << "\n";
		std::cout << "\n";

		std::cout << "============================================================" << "\n";
		std::cout << "============================================================" << "\n";

		Variable weight5, weight6, weight7;
		weight5.Xavier_initialization(CLASS_NUM_NODE1, CLASS_NUM_INPUT_DATA);
		weight6.Xavier_initialization(CLASS_NUM_NODE2, CLASS_NUM_NODE1);
		weight7.Xavier_initialization(CLASS_NUM_LABEL, CLASS_NUM_NODE2);

		Model study_model_classify;

		Multiplication node5{ weight5 };
		Activation::Sigmoid active4 = Activation::Sigmoid();
		Multiplication node6{ weight6 };
		Activation::Sigmoid active5 = Activation::Sigmoid();
		Multiplication node7{ weight7 };
		Softmax softmax = Softmax();

		study_model_classify.addFunc(node5);
		study_model_classify.addFunc(active4);
		study_model_classify.addFunc(node6);
		study_model_classify.addFunc(active5);
		study_model_classify.addFunc(node7);
		study_model_classify.addFunc(softmax);

		Momentum Momentum_Optimizer(0.9, lr);

		study_model_classify.setOptimizer(&Momentum_Optimizer);

		std::cout << "TEST OF THE LINKING STATE OF THE FUNCITON (Backward / Function / Forward)" << "\n";
		std::cout << "\n";
		for (auto& f : study_model_classify.funcQueue)
		{
			std::cout << f->function_name() << " / " << f->back << " " << f << " " << f->next << "\n";
		}
		std::cout << "----------------------------------------------------------" << "\n";
		std::cout << "\n";

		std::cout << "Test of Classification Model with Quadrant" << "\n";
		Variable input_class = Variable{ {1, 2, -1, -2, -4, -1, 5, 3}, {1, 2, 1, 1, -3, -2, -2, -6} };
		Variable quadrant_label = Variable{ {1, 1, 0, 0, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 1} };
		study_model_classify.train(iters, input_class, quadrant_label);
		std::cout << "----------------------------------------------------------" << "\n";
		std::cout << "\n";
	}


}