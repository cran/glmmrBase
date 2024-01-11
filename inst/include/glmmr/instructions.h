#pragma once

#include "general.h"

// different Dos for the calculator
enum class Do{
  PushUserNumber0 = 0,
  PushUserNumber1 = 1,
  PushUserNumber2 = 2,
  PushUserNumber3 = 3,
  PushUserNumber4 = 4,
  PushUserNumber5 = 5,
  PushUserNumber6 = 6,
  PushUserNumber7 = 7,
  PushUserNumber8 = 8,
  PushUserNumber9 = 9,
  PushUserNumber10 = 10,
  PushUserNumber11 = 11,
  PushUserNumber12 = 12,
  PushUserNumber13 = 13,
  PushUserNumber14 = 14,
  PushUserNumber15 = 15,
  PushUserNumber16 = 16,
  PushUserNumber17 = 17,
  PushUserNumber18 = 18,
  PushUserNumber19 = 19,
  Add,
  Subtract,
  Multiply,
  Divide,
  PushData,
  PushCovData,
  PushParameter,
  PushY,
  Power,
  Exp,
  Sqrt,
  Square,
  Negate,
  Bessel,
  BesselK,
  Log,
  Gamma,
  Sin,
  Cos,
  PushExtraData,
  Int1,
  Int2,
  Int3,
  Int4,
  Int5,
  Int6,
  Int7,
  Int8,
  Int9,
  Int10,
  Pi,
  Constant1,
  Constant2,
  Constant3,
  Constant4,
  Constant5,
  Constant6,
  LogFactorialApprox,
  PushVariance
};

const static std::vector<Do> xvar_rpn = {Do::PushData,Do::PushCovData,Do::Subtract,Do::Square};

// for printing for debugging
const std::map<Do,std::string> instruction_str{{Do::Add, "Add"}, 
{Do::Subtract, "Subtract"}, 
{Do::Multiply, "Multiply"},
{Do::Divide, "Divide"},
{Do::PushData, "Push data"},
{Do::PushCovData, "Push cov data"},
{Do::PushParameter, "Push parameter"},
{Do::PushY, "Push y"},
{Do::Power, "Power"},
{Do::Exp, "Exp"},
{Do::Sqrt, "Sqrt"},
{Do::Square, "Square"},
{Do::Negate, "Negate"},
{Do::Bessel, "Bessel"},
{Do::BesselK, "BesselK"},
{Do::Log, "Log"},
{Do::Gamma, "Gamma"},
{Do::Sin, "Sin"},
{Do::Cos, "Cos"},
{Do::PushExtraData, "Push extra data"},
{Do::Int1, "Int 1"},
{Do::Int2, "Int 2"},
{Do::Int3, "Int 3"},
{Do::Int4, "Int 4"},
{Do::Int5, "Int 5"},
{Do::Int6, "Int 6"},
{Do::Int7, "Int 7"},
{Do::Int8, "Int 8"},
{Do::Int9, "Int 9"},
{Do::Int10, "Int 10"},
{Do::Pi, "Pi"},
{Do::Constant1, "Constant 1"},
{Do::Constant2, "Constant 2"},
{Do::Constant3, "Constant 3"},
{Do::Constant4, "Constant 4"},
{Do::Constant5, "Constant 5"},
{Do::Constant6, "Constant 6"},
{Do::LogFactorialApprox, "Log factorial approx"},
{Do::PushVariance, "Push variance"},
{Do::PushUserNumber0, "Push user number 0"},
{Do::PushUserNumber1, "Push user number 1"},
{Do::PushUserNumber2, "Push user number 2"},
{Do::PushUserNumber3, "Push user number 3"},
{Do::PushUserNumber4, "Push user number 4"},
{Do::PushUserNumber5, "Push user number 5"},
{Do::PushUserNumber6, "Push user number 6"},
{Do::PushUserNumber7, "Push user number 7"},
{Do::PushUserNumber8, "Push user number 8"},
{Do::PushUserNumber9, "Push user number 9"}
};
