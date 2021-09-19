#pragma once
#include <assert.h>
class GridCell {
public:
	// �ش� �������� �̵��� ���� �̵� ���� ��Ÿ���� ����
	float direction[4]{ 0 };
	// �ش� ���� �����Ͽ��� �� ��� �Ǵ� ����ġ�� ��Ÿ���� ����
	float profit;
	// �ش� ���� ������������ ��Ÿ���� ����
	bool goal;
	// �ش� ���� ��ֹ������� ��Ÿ���� ����
	bool obstacle;


public:
	GridCell() : profit(-0.1), goal(false), obstacle(false) {};

	/*
	dir = 0(��), 1(��), 2(��), 3(��)
	� �������� �̵��Ͽ��� ��, (�� ���� ���Ͽ� ���� �� �ִ� ����ġ + �ش� �������� �ִ� �̵�)��
	����Ͽ� �̸� �̵� �� ���� �̵����⿡ ������Ʈ ���ش�.
	*/
	void update(const int& dir, GridCell* moved_cell) {
		direction[dir] = moved_cell->max_profit() + moved_cell->profit;
	}

	// �� ���� �� �ִ� �̵��� ��ȯ�Ѵ�.
	float max_profit() {
		float temp_max = direction[0];
		for (int i = 0; i < 4; ++i) {
			if (temp_max < direction[i]) {
				temp_max = direction[i];
			}
		}
		return temp_max;
	}

	int max_profit_dir() {
		float temp_max = direction[0];
		int temp_dir = 0;
		for (int i = 1; i < 4; ++i) {
			if (temp_max < direction[i]) {
				temp_max = direction[i];
				temp_dir = i;
			}
		}
		return temp_dir;
	}
};