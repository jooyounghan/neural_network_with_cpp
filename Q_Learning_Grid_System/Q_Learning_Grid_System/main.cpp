#include "Grid.h"

int main() {
	Grid GridSystem{ 2, 3 };
	Agent agent{ 0, 0 };

	GridSystem.set_agent(&agent);
	GridSystem.set_goal(1, 2);

	GridSystem.initialize_profit(-0.1);

	GridSystem.train(100000);
	std::cout << "Grid System Train Finished!" << std::endl;

	GridSystem.go();
}