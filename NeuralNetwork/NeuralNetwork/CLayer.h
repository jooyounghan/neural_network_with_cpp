#pragma once

#include "CMatrix.h"

class CMatrix;

class CLayer
{
friend class CInitializer;

protected:
	std::shared_ptr<CMatrix> input;
	std::shared_ptr<CMatrix> output;
	std::shared_ptr<CMatrix> gradient;

public:
	CLayer();
	virtual ~CLayer();

public:
	void SetInput(std::shared_ptr<CMatrix> newInput);
	virtual void SetOutput() = 0;
	void SetGradient(std::shared_ptr<CMatrix> newGradient);

public:
	void Connect(CLayer* layer);

public:
	virtual void ForwardProp() = 0;
	virtual CMatrix BackwardProp(CLayer* layer) = 0;

public:
	std::shared_ptr<CMatrix> GetInput();
	std::shared_ptr<CMatrix> GetOutput();
	std::shared_ptr<CMatrix> GetGradient();
};

