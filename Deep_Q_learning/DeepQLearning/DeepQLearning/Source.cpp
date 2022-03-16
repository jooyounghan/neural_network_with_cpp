#include "pch.h"
#include <iostream>
#include <string>


int main()
{	

	Nodea* a = new Nodea();
	NodeTwo* b = TypeCast<NodeTwo*>(a);
	bool c = CanCast<NodeTwo*>(a);

	NodeTwo* d = new NodeTwo();
	Nodea* e = TypeCast<Nodea*>(d);
	bool f = CanCast<Nodea*>(d);
	bool g;
}