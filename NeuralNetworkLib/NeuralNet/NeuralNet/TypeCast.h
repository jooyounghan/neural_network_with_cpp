#pragma once
#include "IncludePart.h"

#pragma region Interface Defining
#define DECLARE_TL		using TL = TL;	int _typeId;
#define INIT_TL(Type)	_typeId = IndexOf<TL, Type>::Result;
#pragma endregion

#pragma region TypeList
template<typename... V>
struct TypeList;

template<typename U, typename V>
struct TypeList<U, V>
{
	using Head = U;
	using Tail = V;
};

template<typename U, typename... V>
struct TypeList<U, V...>
{
	using Head = U;
	using Tail = TypeList<V...>;
};
#pragma endregion

#pragma region Length
template<typename U>
struct Length;

template<>
struct Length<TypeList<>>
{
	enum { value = 0 };
};

template <typename U, typename... V>
struct Length<TypeList<U, V...>>
{
	enum { value = 1 + Length<TypeList<V...>>::value };
};
#pragma endregion

#pragma region TypeAt
template <typename U, int idx>
struct TypeAt;

template<int idx>
struct TypeAt<TypeList<>, idx>
{
	using Result = typename void;
};

template <typename U, typename... V>
struct TypeAt<TypeList<U, V...>, 0>
{
	using Result = typename U;
};

template <typename U, typename... V, int idx>
struct TypeAt<TypeList<U, V...>, idx>
{
	using Result = typename TypeAt<TypeList<V...>, idx - 1>::Result;
};
#pragma endregion

#pragma region IndexOf
template<typename U, typename V>
struct IndexOf;

template<typename U, typename... V>
struct IndexOf<TypeList<U, V...>, U>
{
	enum { Result = 0 };
};

template <typename U>
struct IndexOf<TypeList<>, U>
{
	enum { Result = -1 };
};

template<typename U, typename... V, typename X>
struct IndexOf<TypeList<U, V...>, X>
{
private:
	enum { temp = IndexOf<TypeList<V...>, X>::Result };
public:
	enum { Result = (temp == -1) ? -1 : temp + 1 };
};
#pragma endregion

#pragma region Conversion
template<typename U, typename V>
class Conversion
{
private:
	using Small = __int8;
	using Big	= __int32;

private:
	static Small	Test(const V&) { return 0; }
	static Big		Test(...) { return 0; }
	static U		MakeFrom() { return 0; }

public:
	enum { exists = sizeof(Test(MakeFrom())) == sizeof(Small) };
};
#pragma endregion

#pragma region Int2Type
template<int v>
struct Int2Type
{
	enum { Result = v };
};
#pragma endregion

#pragma region TypeConversion
template<typename TL>
class TypeConversion
{
private:
	enum { length = Length<TL>::value };
	static bool convertTable[length][length];

	TypeConversion() { MakeTable(Int2Type<0>(), Int2Type<0>()); };

public:
	template<int U, int V>
	static void MakeTable(Int2Type<U>, Int2Type<V>)
	{
		using From = typename TypeAt<TL, U>::Result;
		using To = typename TypeAt<TL, V>::Result;
		if (Conversion<From, To>::exists) { convertTable[U][V] = true; }
		else { convertTable[U][V] = false; }
		 
		MakeTable(Int2Type<U>(), Int2Type<V + 1>());
	}

	template<int U>
	static void MakeTable(Int2Type<U>, Int2Type<length>)
	{
		MakeTable(Int2Type<U + 1>(), Int2Type<0>());
	}

	template<int U>
	static void MakeTable(Int2Type<length>, Int2Type<U>) { }

	static bool CanConvert(int from, int to)
	{
		static TypeConversion<TL> typeconversion;
		return convertTable[from][to];
	}
};

template<typename TL>
bool TypeConversion<TL>::convertTable[length][length];

template<typename To, typename From>
To TypeCast(From* ptr)
{
	if (ptr == nullptr) return nullptr;
	if (TypeConversion<From::TL>::CanConvert(ptr->_typeId, IndexOf<From::TL, std::remove_pointer_t<To>>::Result))
		return static_cast<To>(ptr);
	return nullptr;
}

template<typename To, typename From>
std::shared_ptr<To> TypeCast(std::shared_ptr<From> ptr)
{
	if (ptr == nullptr) return nullptr;
	if (TypeConversion<From::TL>::CanConvert(ptr->_typeId, IndexOf<From::TL, std::remove_pointer_t<To>>::Result))
		return std::static_pointer_cast<To>(ptr);
	return nullptr;
}

template<typename To, typename From>
bool CanCast(From* ptr)
{
	if (ptr == nullptr) return false;
	return TypeConversion<From::TL>::CanConvert(ptr->_typeId, IndexOf<From::TL, std::remove_pointer_t<To>>::Result);
}

template<typename To, typename From>
bool CanCast(std::shared_ptr<From> ptr)
{
	if (ptr == nullptr) return false;
	return TypeConversion<From::TL>::CanConvert(ptr->_typeId, IndexOf<From::TL, std::remove_pointer_t<To>>::Result);
}
#pragma endregion