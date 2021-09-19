#include "Grid.h"

int main() {
	Grid GridSystem{ 4, 6 };
	Agent agent{ 0, 0 };

	/*
	아래와 같은 Grid System을 생성한다.
	* : 출발지점
	O : 이동 가능 경로
	/ : 장애물
	@ : 도착지점
		*  /  O  O  O  O
		O  /  O  /  O  O
		O  /  O  /  O  O
		O  O  O  /  O  @
	*/

	GridSystem.set_agent(&agent);
	GridSystem.set_goal(3, 5);

	GridSystem.set_obstacle(0, 1);
	GridSystem.set_obstacle(1, 1);
	GridSystem.set_obstacle(2, 1);
	GridSystem.set_obstacle(1, 3);
	GridSystem.set_obstacle(2, 3);
	GridSystem.set_obstacle(3, 3);

	GridSystem.initialize_profit(-1.0 / float(GridSystem.size));

	GridSystem.train(10000000);
	std::cout << "Grid System Train Finished!" << std::endl;

	GridSystem.go();
}