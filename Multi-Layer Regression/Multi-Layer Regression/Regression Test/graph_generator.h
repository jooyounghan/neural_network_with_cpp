#pragma once
#include "gl/glew.h"
#include "GLFW/glfw3.h"

#include "function_generator.h"


class graph_generator {
public:
	function_generator* label_data;
	function_generator* noised_data;
	function_generator* nn_output_data;

public:
	graph_generator() : label_data(nullptr), noised_data(nullptr), nn_output_data(nullptr) {};

	void setLabelData(function_generator& label_func_in) {
		if (label_data != nullptr) {
			delete label_data;
			label_data = nullptr;
		}
		label_data = &label_func_in;
	}

	void setNoisedData(function_generator& noised_func_in) {
		if (noised_data != nullptr) {
			delete noised_data;
			noised_data = nullptr;
		}
		noised_data = &noised_func_in;
	}

	void draw() {

		if (!glfwInit()) {
			std::cout << "glfw intialization failed" << std::endl;
			glfwTerminate();
			return;
		}

		GLFWwindow* graph_window = glfwCreateWindow(1280, 960, "Graph Generator", NULL, NULL);

		glfwMakeContextCurrent(graph_window);
		glClearColor(0, 0, 0, 0);

		int buf_width, buf_height;
		glfwGetFramebufferSize(graph_window, &buf_width, &buf_height);
		float aspect_ratio = float(buf_width) / float(buf_height);
		glViewport(0, 0, buf_width, buf_height);
		float& window_start = label_data->start;
		float& window_end = label_data->end;
		glOrtho(window_start, window_end, window_start / aspect_ratio, window_end / aspect_ratio, -1, 1);

		while (!glfwWindowShouldClose(graph_window)) {
			glClear(GL_COLOR_BUFFER_BIT);
			
			glPointSize(10);
			glBegin(GL_LINE_STRIP);
			glColor3f(1, 0, 0);
			for (int idx = 0; idx < label_data->num_data; ++idx) {
				glVertex3f(label_data->x_data[idx], label_data->y_data[idx], 0.0);
			}
			glEnd();

			glPointSize(10);
			glBegin(GL_POINTS);
			glColor3f(0, 1, 0);
			for (int idx = 0; idx < noised_data->num_data; ++idx) {
				glVertex3f(noised_data->x_data[idx], noised_data->y_data[idx], 0.0);
			}
			glEnd();

			glfwSwapBuffers(graph_window);
			glfwPollEvents();
		}
		glfwTerminate();
		return;
	}


};