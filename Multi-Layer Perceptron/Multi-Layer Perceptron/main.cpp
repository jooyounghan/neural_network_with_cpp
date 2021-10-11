#include <iostream>
#include "Model.h"

int main()
{
	while (true) {
		Variable m_weight1, m_weight2, m_weight3, m_weight4;
		m_weight1.Xavier_initialization(NUM_NODE1, NUM_INPUT_DATA);
		m_weight2.Xavier_initialization(NUM_NODE2, NUM_NODE1);
		m_weight3.Xavier_initialization(NUM_NODE3, NUM_NODE2);
		m_weight4.Xavier_initialization(NUM_OUPUT_DATA, NUM_NODE3);

		Model m_study_model;

		Adam Adam_Optimizer1(1E-8, 0.9, 0.999, lr_2);
		Adam Adam_Optimizer2(1E-8, 0.9, 0.999, lr_2);
		Adam Adam_Optimizer3(1E-8, 0.9, 0.999, lr_2);
		Adam Adam_Optimizer4(1E-8, 0.9, 0.999, lr_2);

		Multiplication m_node1{ m_weight1, &Adam_Optimizer1 };
		Activation::Sigmoid m_active1 = Activation::Sigmoid();
		Multiplication m_node2{ m_weight2, &Adam_Optimizer2 };
		Activation::Sigmoid m_active2 = Activation::Sigmoid();
		Multiplication m_node3{ m_weight3, &Adam_Optimizer3 };
		Activation::Sigmoid m_active3 = Activation::Sigmoid();
		Multiplication m_node4{ m_weight4, &Adam_Optimizer4 };

		m_study_model.addFunc(m_node1);
		m_study_model.addFunc(m_active1);
		m_study_model.addFunc(m_node2);
		m_study_model.addFunc(m_active2);
		m_study_model.addFunc(m_node3);
		m_study_model.addFunc(m_active3);
		m_study_model.addFunc(m_node4);

		std::cout << "TEST OF THE LINKING STATE OF THE FUNCITON (Backward / Function / Forward)" << "\n";
		std::cout << "\n";
		for (auto& f : m_study_model.funcQueue)
		{
			std::cout << f->function_name() << " / " << f->back << " " << f << " " << f->next << "\n";
		}
		std::cout << "----------------------------------------------------------" << "\n";
		std::cout << "\n";

		std::cout << "Test of the nonlinear Model with Momentum Optimizer and XOR" << "\n";
		Variable input = Variable{ {-5, -3, -1, 0, 1, 3, 5 } };
		Variable xor_label = Variable{ {-125, -8, -1, 0, 1, 8, 125} };
		m_study_model.train(iters, input, xor_label);
		std::cout << "----------------------------------------------------------" << "\n";
		std::cout << "\n";
		Variable new_input = Variable{ {-4, -2, -5, 0, 3, 4, 5 } };
		Variable new_output = m_study_model.getResult(new_input);
		new_output.print();
	}

	Variable input = Variable{ {0, 0, 1, 1}, {0, 1, 0, 1} };
	Variable xor_label = Variable{ {0, 1, 1, 0} };

	Variable n_weight1, n_weight2, n_weight3, n_weight4;
	n_weight1.Xavier_initialization(NUM_NODE1, NUM_INPUT_DATA);
	n_weight2.Xavier_initialization(NUM_NODE2, NUM_NODE1);
	n_weight3.Xavier_initialization(NUM_NODE3, NUM_NODE2);
	n_weight4.Xavier_initialization(NUM_OUPUT_DATA, NUM_NODE3);

	Model n_study_model;

	NAG NAG_Optimizer1(0.9, lr);
	NAG NAG_Optimizer2(0.9, lr);
	NAG NAG_Optimizer3(0.9, lr);
	NAG NAG_Optimizer4(0.9, lr);

	Multiplication n_node1{ n_weight1, &NAG_Optimizer1 };
	Activation::Relu n_active1 = Activation::Relu();
	Multiplication n_node2{ n_weight2, &NAG_Optimizer2 };
	Activation::Relu n_active2 = Activation::Relu();
	Multiplication n_node3{ n_weight3, &NAG_Optimizer3 };
	Activation::Relu n_active3 = Activation::Relu();
	Multiplication n_node4{ n_weight4, &NAG_Optimizer4 };

	n_study_model.addFunc(n_node1);
	n_study_model.addFunc(n_active1);
	n_study_model.addFunc(n_node2);
	n_study_model.addFunc(n_active2);
	n_study_model.addFunc(n_node3);
	n_study_model.addFunc(n_active3);
	n_study_model.addFunc(n_node4);



	std::cout << "Test of the nonlinear Model with NAG Optimizer and XOR" << "\n";
	n_study_model.train(iters, input, xor_label);
	std::cout << "----------------------------------------------------------" << "\n";
	std::cout << "\n";


	std::cout << "============================================================" << "\n";
	std::cout << "============================================================" << "\n";

	Variable m_weight5, m_weight6, m_weight7;
	m_weight5.Xavier_initialization(CLASS_NUM_NODE1, CLASS_NUM_INPUT_DATA);
	m_weight6.Xavier_initialization(CLASS_NUM_NODE2, CLASS_NUM_NODE1);
	m_weight7.Xavier_initialization(CLASS_NUM_LABEL, CLASS_NUM_NODE2);

	Model m_study_model_classify;

	Momentum Momentum_Optimizer_classify1(0.9, lr);
	Momentum Momentum_Optimizer_classify2(0.9, lr);
	Momentum Momentum_Optimizer_classify3(0.9, lr);


	Multiplication m_node5{ m_weight5, &Momentum_Optimizer_classify1 };
	Activation::Relu m_active4 = Activation::Relu();
	Multiplication m_node6{ m_weight6, &Momentum_Optimizer_classify2 };
	Activation::Relu m_active5 = Activation::Relu();
	Multiplication m_node7{ m_weight7, &Momentum_Optimizer_classify3 };
	Softmax m_softmax = Softmax();

	m_study_model_classify.addFunc(m_node5);
	m_study_model_classify.addFunc(m_active4);
	m_study_model_classify.addFunc(m_node6);
	m_study_model_classify.addFunc(m_active5);
	m_study_model_classify.addFunc(m_node7);
	m_study_model_classify.addFunc(m_softmax);



	std::cout << "TEST OF THE LINKING STATE OF THE FUNCITON (Backward / Function / Forward)" << "\n";
	std::cout << "\n";
	for (auto& f : m_study_model_classify.funcQueue)
	{
		std::cout << f->function_name() << " / " << f->back << " " << f << " " << f->next << "\n";
	}
	std::cout << "----------------------------------------------------------" << "\n";
	std::cout << "\n";

	std::cout << "Test of Classification Model with Momentum Optimizer Quadrant" << "\n";
	Variable input_class = Variable{ {1, 2, -1, -2, -4, -1, 5, 3}, {1, 2, 1, 1, -3, -2, -2, -6} };
	Variable quadrant_label = Variable{ {1, 1, 0, 0, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 1} };
	m_study_model_classify.train(iters, input_class, quadrant_label);
	std::cout << "----------------------------------------------------------" << "\n";
	std::cout << "\n";


	Variable n_weight5, n_weight6, n_weight7;
	n_weight5.Xavier_initialization(CLASS_NUM_NODE1, CLASS_NUM_INPUT_DATA);
	n_weight6.Xavier_initialization(CLASS_NUM_NODE2, CLASS_NUM_NODE1);
	n_weight7.Xavier_initialization(CLASS_NUM_LABEL, CLASS_NUM_NODE2);

	Model n_study_model_classify;

	NAG NAG_Optimizer_classify1(0.9, lr);
	NAG NAG_Optimizer_classify2(0.9, lr);
	NAG NAG_Optimizer_classify3(0.9, lr);

	Multiplication n_node5{ n_weight5, &NAG_Optimizer_classify1 };
	Activation::Relu n_active4 = Activation::Relu();
	Multiplication n_node6{ n_weight6, &NAG_Optimizer_classify2 };
	Activation::Relu n_active5 = Activation::Relu();
	Multiplication n_node7{ n_weight7, &NAG_Optimizer_classify3 };
	Softmax n_softmax = Softmax();

	n_study_model_classify.addFunc(n_node5);
	n_study_model_classify.addFunc(n_active4);
	n_study_model_classify.addFunc(n_node6);
	n_study_model_classify.addFunc(n_active5);
	n_study_model_classify.addFunc(n_node7);
	n_study_model_classify.addFunc(n_softmax);

	std::cout << "Test of Classification Model with NAG Optimizer Quadrant" << "\n";
	n_study_model_classify.train(iters, input_class, quadrant_label);
	std::cout << "----------------------------------------------------------" << "\n";
	std::cout << "\n";


	std::cout << "============================================================" << "\n";
	std::cout << "============================================================" << "\n";

	Variable weight1, weight2, weight3;
	weight1.Xavier_initialization(CLASS_NUM_NODE1, CLASS_NUM_INPUT_DATA);
	weight2.Xavier_initialization(CLASS_NUM_NODE2, CLASS_NUM_NODE1);
	weight3.Xavier_initialization(CLASS_NUM_LABEL, CLASS_NUM_NODE2);

	Model study_model_classify;

	//Adagrad Adagrad_Optimizer_classify1(1E-8, lr_3);
	//Adagrad Adagrad_Optimizer_classify2(1E-8, lr_3);
	//Adagrad Adagrad_Optimizer_classify3(1E-8, lr_3);

	//RMSprop RMSprop_Optimizer_classify1(1E-8, 0.9, lr_2);
	//RMSprop RMSprop_Optimizer_classify2(1E-8, 0.9, lr_2);
	//RMSprop RMSprop_Optimizer_classify3(1E-8, 0.9, lr_2);

	Adam Adam_Optimizer_classify1(1E-8, 0.9, 0.999, lr_2);
	Adam Adam_Optimizer_classify2(1E-8, 0.9, 0.999, lr_2);
	Adam Adam_Optimizer_classify3(1E-8, 0.9, 0.999, lr_2);

	//Multiplication node1{ weight1, &Adagrad_Optimizer_classify1 };
	//Activation::Sigmoid active1 = Activation::Sigmoid();
	//Multiplication node2{ weight2, &Adagrad_Optimizer_classify2 };
	//Activation::Sigmoid active2 = Activation::Sigmoid();
	//Multiplication node3{ weight3, &Adagrad_Optimizer_classify3 };
	//Softmax softmax = Softmax();

	//Multiplication node1{ weight1, &RMSprop_Optimizer_classify1 };
	//Activation::Sigmoid active1 = Activation::Sigmoid();
	//Multiplication node2{ weight2, &RMSprop_Optimizer_classify2 };
	//Activation::Sigmoid active2 = Activation::Sigmoid();
	//Multiplication node3{ weight3, &RMSprop_Optimizer_classify3 };
	//Softmax softmax = Softmax();

	Multiplication node1{ weight1, &Adam_Optimizer_classify1 };
	Activation::Sigmoid active1 = Activation::Sigmoid();
	Multiplication node2{ weight2, &Adam_Optimizer_classify2 };
	Activation::Sigmoid active2 = Activation::Sigmoid();
	Multiplication node3{ weight3, &Adam_Optimizer_classify3 };
	Softmax softmax = Softmax();

	study_model_classify.addFunc(node1);
	study_model_classify.addFunc(active1);
	study_model_classify.addFunc(node2);
	study_model_classify.addFunc(active2);
	study_model_classify.addFunc(node3);
	study_model_classify.addFunc(softmax);


	//std::cout << "Test of Classification Model with Adagrad Optimizer Quadrant" << "\n";
	//std::cout << "Test of Classification Model with RMSprop Optimizer Quadrant" << "\n";
	std::cout << "Test of Classification Model with Adam Optimizer Quadrant" << "\n";

	study_model_classify.train(iters, input_class, quadrant_label);
	std::cout << "----------------------------------------------------------" << "\n";
	std::cout << "\n";

}