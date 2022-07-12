#pragma once
#include <Windows.h>
// Crash Macro

#define CRASH(cause)						\
{											\
	unsigned int*	crash = nullptr;		\
	__analysis_assume(crash != nullptr);	\
	*crash = 0xDEADBEEF;					\
}											\

#define ASSERT_CRASH(expr)					\
{											\
	if(!(expr)){							\
		CRASH("ASSERT_CRASH");				\
		__analysis_assume(expr);			\
	}										\
}											\