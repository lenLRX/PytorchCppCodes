#ifndef __UTIL_H__
#define __UTIL_H__

#include <string>
#include <algorithm>

#ifdef WIN32
#ifdef SIMDOTA2_DLLEXPORT
#define SIMDOTA2_API __declspec(dllexport)
#else
#define SIMDOTA2_API __declspec(dllimport)
#endif
#else
#define SIMDOTA2_API
#endif

//
std::string DirName(std::string source);

template<typename T>
void upperBound(T& a, const T& bound)
{
    if (a > bound)
    {
        a = bound;
    }
}

template<typename T>
void lowerBound(T& a, const T& bound)
{
    if (a < bound)
    {
        a = bound;
    }
}

#endif//__UTIL_H__