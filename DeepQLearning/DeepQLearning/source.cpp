#include <pch.h>
#include "pch.h"
int main()
{
	CMatrix* a = new CMatrix(128, 64);
	CMatrix* b = new CMatrix(64, 128);
	a->XavierNormalInitialize(128, 128);
	b->XavierNormalInitialize(128, 128);
	
	CLayer2D layer;
	layer.SetInput(a);
	layer.SetWeight(b);

	CLayer2D layer2;
	layer2.SetInput(new CMatrix(128, 128, new double[128 * 128]));

	CTimer timer;
	timer.StartCheck();

	layer2.GetInput()->GetMatMul(layer.GetInput(), layer.GetWeight());
	timer.EndCheck();

	timer.PrintElapsedTime("Get Result");
//
//	std::cout << std::endl;
//	d->PrintData();
//	std::cout << std::endl;
//	e->PrintData();
//	std::cout << std::endl;
//	f->PrintData();
//	std::cout << std::endl;
}