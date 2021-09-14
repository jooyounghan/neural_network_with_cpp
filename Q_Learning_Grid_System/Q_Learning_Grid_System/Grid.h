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
	
	move : Agent를 이동시키는 함수(범위에 대한 분기처리 필요)

	setprofit : 셀의 이득을 설정하는 함수
	


	*/

	~Grid(){
		if (gridmap != nullptr) {
			delete[] gridmap;
		}
	}
};