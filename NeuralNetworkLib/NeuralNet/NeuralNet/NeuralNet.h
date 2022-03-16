#pragma once
#include "Layer.h"

using TL = TypeList<class Nodea, class NodeTwo, class Layera, class LayeraTwo>;

class Nodea
{
public:
	DECLARE_TL
	Nodea() { INIT_TL(Nodea) }

public:
	int a;
	std::string t;
};

class NodeTwo : public Nodea
{
public:
	NodeTwo() { INIT_TL(NodeTwo) }
	int b;
};


class Layera
{
public:
	DECLARE_TL
	Layera() { INIT_TL(Layera) }

public:
	int a;
	std::string t;
	std::vector<int> v;
};

class LayeraTwo : public Layera
{
public:
	LayeraTwo() { INIT_TL(LayeraTwo) }
	int b;
	std::vector<int> v2;
};
