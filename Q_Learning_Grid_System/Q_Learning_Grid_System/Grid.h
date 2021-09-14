#pragma once
#include "GridCell.h"
#include "Agent.h"

class Grid {
public:
	int row, col;
	int size;
	GridCell* gridmap = nullptr;

	

public:
	Grid(const int& row_in, const int& col_in)
		: row(row_in), col(col_in), size(row_in* col_in) {
		gridmap = new GridCell[size];
	}

	/*
	TODO
	
	move : Agent�� �̵���Ű�� �Լ�(������ ���� �б�ó�� �ʿ�)

	setprofit : ���� �̵��� �����ϴ� �Լ�
	


	*/

	~Grid(){
		if (gridmap != nullptr) {
			delete[] gridmap;
		}
	}
};