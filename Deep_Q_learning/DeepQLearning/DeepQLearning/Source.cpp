#include "pch.h"
#include <iostream>
#include <pch.h>

int main()
{	
	NeuralNet network;
	network.SetLayer(3, Dim2D(1, 3), Dim2D(1, 6), Dim2D(1, 5));
	bool check = true;



	Layer2D layer(Dim2D(1, 3));
	layer.actFunc = new Relu;
}