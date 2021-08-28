#include <iostream>
#include "Model.h"

int main()
{
	while (true) {
		//Variable weight1, weight2, weight3, weight4;
		//weight1.init_weight(NUM_NODE1, NUM_INPUT_DATA);
		//weight2.init_weight(NUM_NODE2, NUM_NODE1);
		//weight3.init_weight(NUM_NODE3, NUM_NODE2);
		//weight4.init_weight(NUM_OUPUT_DATA, NUM_NODE3);

		//Model study_model;
		//study_model.addFunc(new Multiplication(weight1));
		//study_model.addFunc(new Activation::Relu());
		//study_model.addFunc(new Multiplication(weight2));
		//study_model.addFunc(new Activation::Relu());
		//study_model.addFunc(new Multiplication(weight3));
		//study_model.addFunc(new Activation::Relu());
		//study_model.addFunc(new Multiplication(weight4));

		//Momentum Momentum_Optimizer(0.9, lr);

		//study_model.setOptimizer(&Momentum_Optimizer);

		//std::cout << "TEST OF THE LINKING STATE OF THE FUNCITON (Backward / Function / Forward)" << "\n";
		//std::cout << "\n";
		//for (auto& f : study_model.funcQueue)
		//{
		//	std::cout << f->function_name() << " / " << f->back << " " << f << " " << f->next << "\n";
		//}
		//std::cout << "----------------------------------------------------------" << "\n";
		//std::cout << "\n";

		//std::cout << "Test of the nonlinear Model with XOR" << "\n";
		//Variable input = Variable{ {0, 0, 1, 1}, {0, 1, 0, 1} };
		//Variable xor_label = Variable{ {0, 1, 1, 0} };
		//study_model.train(iters, input, xor_label);
		//std::cout << "----------------------------------------------------------" << "\n";
		//std::cout << "\n";

		//std::cout << "============================================================" << "\n";
		//std::cout << "============================================================" << "\n";

		Variable weight5, weight6, weight7;
		weight5.init_weight(CLASS_NUM_NODE1, CLASS_NUM_INPUT_DATA);
		weight6.init_weight(CLASS_NUM_NODE2, CLASS_NUM_NODE1);
		weight7.init_weight(CLASS_NUM_LABEL, CLASS_NUM_NODE2);

		Model study_model_classify;
		study_model_classify.addFunc(new Multiplication(weight5));
		study_model_classify.addFunc(new Activation::Relu());
		study_model_classify.addFunc(new Multiplication(weight6));
		study_model_classify.addFunc(new Activation::Relu());
		study_model_classify.addFunc(new Multiplication(weight7));
		study_model_classify.addFunc(new Softmax());

		GradientDescent GD_Optimizer(lr);

		study_model_classify.setOptimizer(&GD_Optimizer);

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