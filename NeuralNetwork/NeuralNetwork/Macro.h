#pragma once

// Crash Macro -----------------------------------
#ifdef _DEBUG
#define CRASH(cause)						\
{											\
	UINT*	crash = nullptr;				\
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
// ------------------------------------Crash Macro
#endif // DEBUG




// Delete Macro -----------------------------------
#define DELETEPTR(ptr)							\
{												\
	if (ptr != nullptr)							\
	{											\
		delete ptr;								\
		ptr = nullptr;							\
	}											\
}												\

#define DELETEARRAYPTR(ptr)						\
{												\
	if (ptr != nullptr)							\
	{											\
		delete[] ptr;							\
		ptr = nullptr;							\
	}											\
}												\
// ------------------------------------Delete Macro