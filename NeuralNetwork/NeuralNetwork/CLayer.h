#pragma once
class CMatrix;

class CLayer
{
friend class CInitializer;

protected:
	CMatrix* input;
	CMatrix* output;
	CMatrix* gradient;

protected:
	CLayer* frontLayer;
	CLayer* rearLayer;

public:
	CLayer();
	~CLayer();
public:
	void SetInput(CMatrix* newInput);
	virtual void SetOutput() = 0;
	void SetGradient(CMatrix* newGradient);

	void Connect(CLayer* layer);

	virtual void ForwardProp() = 0;
	virtual CMatrix BackwardProp(CLayer* layer) = 0;

	CMatrix* GetGradient();
};

