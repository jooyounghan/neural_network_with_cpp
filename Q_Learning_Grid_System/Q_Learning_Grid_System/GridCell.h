#pragma once
#include <assert.h>
class GridCell {
private:
	int direction[4]{ 0 };
	// �ش� �������� �̵��� ���� �̵� ��
	int profit;
	// �ش� ���� �����Ͽ��� �� ��� �Ǵ� ����ġ


public:
	GridCell() {};
	

	/*
	TODO
	
	update : � �������� �̵��Ͽ��� ��, (�� ���� ���Ͽ� ���� �� �ִ� ����ġ + �ش� �������� �ִ� �̵�)��
	����Ͽ� �̸� �̵� �� ���� �̵����⿡ ������Ʈ ���ش�.

	maxprofit : �� ���� �� �ִ� �̵��� return

	*/
	void update() {}


};