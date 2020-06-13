// Copyright 2010-2018 Google LLC
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// [START program]
#include "ortools/sat/cp_model.h"

namespace operations_research {
namespace sat {

void SimpleSatProgram() {
  // [START model]
  CpModelBuilder cp_model;
  // [END model]

  // [START variables]
  const Domain domain(-1, 1);
  const Domain second_domain(Domain::FromValues({-1,1}));

  const IntVar x = cp_model.NewIntVar(domain).WithName("x");
  const IntVar y = cp_model.NewIntVar(second_domain).WithName("y");

  const IntVar z = cp_model.NewIntVar(domain).WithName("z");
  // [END variables]

  // [START constraints]
  IntVar sum_weights_activation = cp_model.NewIntVar(Domain(-2,2));
  IntVar sum_temp_1 = cp_model.NewIntVar(Domain(0, 2));
  cp_model.AddEquality(sum_weights_activation, LinearExpr::Sum({x, y}));
  cp_model.AddEquality(sum_temp_1, z.AddConstant(1));
  cp_model.AddAbsEquality(sum_temp_1, sum_weights_activation);
  // [END constraints]


  // [START variables]
  IntVar a = cp_model.NewIntVar(Domain(-1)).WithName("a");
  IntVar b = cp_model.NewIntVar(Domain(-1)).WithName("b");

  IntVar c = cp_model.NewIntVar(domain).WithName("c");
  // [END variables]


  /*
    (C == 0) ssi (weights == 0)
      (C == 0) => (weights == 0) et (weights == 0) => (C == 0)
      Not(weights == 0) => Not(C == 0) et Not(C == 0) => (Not weights == 0)
    (C == 1) ssi (a == b)
      (C == 1) => (a == b) et (a == b) => (C == 1)
      Not(a == b) => Not(C == 1) et Not(C == 1) => Not(a == b)

  */

  BoolVar b1 = cp_model.NewBoolVar();
  BoolVar b2 = cp_model.NewBoolVar();

  IntVar abs_one = cp_model.NewIntVar(second_domain);
  IntVar zero_or_minus_one = cp_model.NewIntVar(Domain(-1,0));

  // Implement b1 == (C == 0)
  cp_model.AddEquality(c, 0).OnlyEnforceIf(b1);
  cp_model.AddNotEqual(c, LinearExpr(0)).OnlyEnforceIf(Not(b1));
  //Implement b2 == (a == 0)
  cp_model.AddEquality(a, 0).OnlyEnforceIf(b2);
  cp_model.AddNotEqual(a, LinearExpr(0)).OnlyEnforceIf(Not(b2));

  // b1 implies b2 and b2 implies b1
  cp_model.AddImplication(b2, b1);
  cp_model.AddImplication(b1, b2);

  BoolVar b3 = cp_model.NewBoolVar();
  BoolVar b4 = cp_model.NewBoolVar();

  // Implement b3 == (c == 1)
  cp_model.AddEquality(c, 1).OnlyEnforceIf(b3);
  cp_model.AddNotEqual(c, LinearExpr(1)).OnlyEnforceIf(Not(b3));
  //Implement b4 == (a == b)
  cp_model.AddEquality(a, b).OnlyEnforceIf(b4);
  cp_model.AddNotEqual(a, b).OnlyEnforceIf(Not(b4));


  // b3 implies b4 and b4 implies b3
  cp_model.AddImplication(b3, b4);
  cp_model.AddImplication(b4, b3);


  // [START constraints]

  // [END constraints]

  // Solving part.
  // [START solve]
  const CpSolverResponse response = Solve(cp_model.Build());
  LOG(INFO) << CpSolverResponseStats(response);
  // [END solve]

  if (response.status() == CpSolverStatus::FEASIBLE) {
    // Get the value of x in the solution.
    LOG(INFO) << "x = " << SolutionIntegerValue(response, x);
    LOG(INFO) << "y = " << SolutionIntegerValue(response, y);
    LOG(INFO) << "z = " << SolutionIntegerValue(response, z);

    LOG(INFO) << "a = " << SolutionIntegerValue(response, a);
    LOG(INFO) << "b = " << SolutionIntegerValue(response, b);
    LOG(INFO) << "c = " << SolutionIntegerValue(response, c);

    LOG(INFO) << "b1 = " << SolutionIntegerValue(response, b1);
    LOG(INFO) << "b2 = " << SolutionIntegerValue(response, b2);
    LOG(INFO) << "b3 = " << SolutionIntegerValue(response, b3);
    LOG(INFO) << "b4 = " << SolutionIntegerValue(response, b4);
  }
}

}  // namespace sat
}  // namespace operations_research

int main() {
  operations_research::sat::SimpleSatProgram();

  return EXIT_SUCCESS;
}
// [END program]
