#pragma once
#include <assert.h>
class GridCell {
public:
	// �ش� �������� �̵��� ���� �̵� ���� ��Ÿ���� ����
	int direction[4]{ 0 };
	// �ش� ���� �����Ͽ��� �� ��� �Ǵ� ����ġ�� ��Ÿ���� ����
	int profit;
	// �ش� ���� ������������ ��Ÿ���� ����
	bool goal;

public:
	GridCell() : profit(-0.1), goal(false) {};

	/*
	dir = 0(��), 1(��), 2(��), 3(��)
	� �������� �̵��Ͽ��� ��, (�� ���� ���Ͽ� ���� �� �ִ� ����ġ + �ش� �������� �ִ� �̵�)��
	����Ͽ� �̸� �̵� �� ���� �̵����⿡ ������Ʈ ���ش�.
	*/
	void update(const int& dir, GridCell* moved_cell) {
		direction[dir] = moved_cell->max_profit() + moved_cell->profit;
	}

	// �� ���� �� �ִ� �̵��� ��ȯ�Ѵ�.
	int max_profit() {
		int temp_max = direction[0];
		for (int i = 0; i < 4; ++i) {
			if (temp_max > direction[i]) {
				temp_max = direction[i];
			}
		}
		return temp_max;
	}
};