#include "ortools/sat/cp_model.h"

namespace operations_research {
namespace sat {

  CpModelBuilder cp_model;
  const Domain domain(-1, 1);
  const Domain second_domain(Domain::FromValues({-1,1}));
  BoolVar b1 = cp_model.NewBoolVar();
  BoolVar b2 = cp_model.NewBoolVar();
  BoolVar b3 = cp_model.NewBoolVar();
  BoolVar b4 = cp_model.NewBoolVar();

  void test_zero (){

    IntVar a = cp_model.NewIntVar(Domain(0)).WithName("a");
    IntVar b = cp_model.NewIntVar(second_domain).WithName("b");

    IntVar c = cp_model.NewIntVar(domain).WithName("c");

    BoolVar b1 = cp_model.NewBoolVar();
    BoolVar b2 = cp_model.NewBoolVar();

    // Implement b1 == (C == 0)
    cp_model.AddEquality(c, 0).OnlyEnforceIf(b1);
    cp_model.AddNotEqual(c, 0).OnlyEnforceIf(Not(b1));
    //Implement b2 == (a == 0)
    cp_model.AddEquality(a, 0).OnlyEnforceIf(b2);
    cp_model.AddNotEqual(a, 0).OnlyEnforceIf(Not(b2));

    // b1 implies b2 and b2 implies b1
    cp_model.AddImplication(b2, b1);
    cp_model.AddImplication(b1, b2);

    BoolVar b3 = cp_model.NewBoolVar();
    BoolVar b4 = cp_model.NewBoolVar();

    // Implement b3 == (c == 1)
    cp_model.AddEquality(c, 1).OnlyEnforceIf(b3);
    cp_model.AddNotEqual(c, 1).OnlyEnforceIf(Not(b3));
    //Implement b4 == (a == b)
    cp_model.AddEquality(a, b).OnlyEnforceIf(b4);
    cp_model.AddNotEqual(a, b).OnlyEnforceIf(Not(b4));


    // b3 implies b4 and b4 implies b3
    cp_model.AddImplication(b3, b4);
    cp_model.AddImplication(b4, b3);

    const CpSolverResponse response = Solve(cp_model.Build());
    LOG(INFO) << CpSolverResponseStats(response);
    // [END solve]

    if (response.status() == CpSolverStatus::FEASIBLE) {

      LOG(INFO) << "a = " << SolutionIntegerValue(response, a);
      LOG(INFO) << "b = " << SolutionIntegerValue(response, b);
      LOG(INFO) << "c = " << SolutionIntegerValue(response, c);

      LOG(INFO) << "b1 = " << SolutionIntegerValue(response, b1);
      LOG(INFO) << "b2 = " << SolutionIntegerValue(response, b2);
      LOG(INFO) << "b3 = " << SolutionIntegerValue(response, b3);
      LOG(INFO) << "b4 = " << SolutionIntegerValue(response, b4);
    }
  }


  void test_minus_one_and_minus_one (){

    IntVar a = cp_model.NewIntVar(Domain(-1)).WithName("a");
    IntVar b = cp_model.NewIntVar(Domain(-1)).WithName("b");

    IntVar c = cp_model.NewIntVar(domain).WithName("c");

    BoolVar b1 = cp_model.NewBoolVar();
    BoolVar b2 = cp_model.NewBoolVar();

    // Implement b1 == (C == 0)
    cp_model.AddEquality(c, 0).OnlyEnforceIf(b1);
    cp_model.AddNotEqual(c, 0).OnlyEnforceIf(Not(b1));
    //Implement b2 == (a == 0)
    cp_model.AddEquality(a, 0).OnlyEnforceIf(b2);
    cp_model.AddNotEqual(a, 0).OnlyEnforceIf(Not(b2));

    // b1 implies b2 and b2 implies b1
    cp_model.AddImplication(b2, b1);
    cp_model.AddImplication(b1, b2);

    BoolVar b3 = cp_model.NewBoolVar();
    BoolVar b4 = cp_model.NewBoolVar();

    // Implement b3 == (c == 1)
    cp_model.AddEquality(c, 1).OnlyEnforceIf(b3);
    cp_model.AddNotEqual(c, 1).OnlyEnforceIf(Not(b3));
    //Implement b4 == (a == b)
    cp_model.AddEquality(a, b).OnlyEnforceIf(b4);
    cp_model.AddNotEqual(a, b).OnlyEnforceIf(Not(b4));


    // b3 implies b4 and b4 implies b3
    cp_model.AddImplication(b3, b4);
    cp_model.AddImplication(b4, b3);

    const CpSolverResponse response = Solve(cp_model.Build());
    LOG(INFO) << CpSolverResponseStats(response);
    // [END solve]

    if (response.status() == CpSolverStatus::FEASIBLE) {

      LOG(INFO) << "a = " << SolutionIntegerValue(response, a);
      LOG(INFO) << "b = " << SolutionIntegerValue(response, b);
      LOG(INFO) << "c = " << SolutionIntegerValue(response, c);

      LOG(INFO) << "b1 = " << SolutionIntegerValue(response, b1);
      LOG(INFO) << "b2 = " << SolutionIntegerValue(response, b2);
      LOG(INFO) << "b3 = " << SolutionIntegerValue(response, b3);
      LOG(INFO) << "b4 = " << SolutionIntegerValue(response, b4);
    }
  }


  void test_one_and_one (){

    IntVar a = cp_model.NewIntVar(Domain(1)).WithName("a");
    IntVar b = cp_model.NewIntVar(Domain(1)).WithName("b");

    IntVar c = cp_model.NewIntVar(domain).WithName("c");

    BoolVar b1 = cp_model.NewBoolVar();
    BoolVar b2 = cp_model.NewBoolVar();

    // Implement b1 == (C == 0)
    cp_model.AddEquality(c, 0).OnlyEnforceIf(b1);
    cp_model.AddNotEqual(c, 0).OnlyEnforceIf(Not(b1));
    //Implement b2 == (a == 0)
    cp_model.AddEquality(a, 0).OnlyEnforceIf(b2);
    cp_model.AddNotEqual(a, 0).OnlyEnforceIf(Not(b2));

    // b1 implies b2 and b2 implies b1
    cp_model.AddImplication(b2, b1);
    cp_model.AddImplication(b1, b2);

    BoolVar b3 = cp_model.NewBoolVar();
    BoolVar b4 = cp_model.NewBoolVar();

    // Implement b3 == (c == 1)
    cp_model.AddEquality(c, 1).OnlyEnforceIf(b3);
    cp_model.AddNotEqual(c, 1).OnlyEnforceIf(Not(b3));
    //Implement b4 == (a == b)
    cp_model.AddEquality(a, b).OnlyEnforceIf(b4);
    cp_model.AddNotEqual(a, b).OnlyEnforceIf(Not(b4));


    // b3 implies b4 and b4 implies b3
    cp_model.AddImplication(b3, b4);
    cp_model.AddImplication(b4, b3);

    const CpSolverResponse response = Solve(cp_model.Build());
    LOG(INFO) << CpSolverResponseStats(response);
    // [END solve]

    if (response.status() == CpSolverStatus::FEASIBLE) {

      LOG(INFO) << "a = " << SolutionIntegerValue(response, a);
      LOG(INFO) << "b = " << SolutionIntegerValue(response, b);
      LOG(INFO) << "c = " << SolutionIntegerValue(response, c);

      LOG(INFO) << "b1 = " << SolutionIntegerValue(response, b1);
      LOG(INFO) << "b2 = " << SolutionIntegerValue(response, b2);
      LOG(INFO) << "b3 = " << SolutionIntegerValue(response, b3);
      LOG(INFO) << "b4 = " << SolutionIntegerValue(response, b4);
    }
  }



  void test_minus_one_and_one(){
    // [START variables]
    IntVar a = cp_model.NewIntVar(Domain(-1)).WithName("a");
    IntVar b = cp_model.NewIntVar(Domain(1)).WithName("b");

    IntVar c = cp_model.NewIntVar(domain).WithName("c");

    BoolVar b1 = cp_model.NewBoolVar();
    BoolVar b2 = cp_model.NewBoolVar();

    // Implement b1 == (C == 0)
    cp_model.AddEquality(c, 0).OnlyEnforceIf(b1);
    cp_model.AddNotEqual(c, 0).OnlyEnforceIf(Not(b1));
    //Implement b2 == (a == 0)
    cp_model.AddEquality(a, 0).OnlyEnforceIf(b2);
    cp_model.AddNotEqual(a, 0).OnlyEnforceIf(Not(b2));

    // b1 implies b2 and b2 implies b1
    cp_model.AddImplication(b2, b1);
    cp_model.AddImplication(b1, b2);

    BoolVar b3 = cp_model.NewBoolVar();
    BoolVar b4 = cp_model.NewBoolVar();

    // Implement b3 == (c == 1)
    cp_model.AddEquality(c, 1).OnlyEnforceIf(b3);
    cp_model.AddNotEqual(c, 1).OnlyEnforceIf(Not(b3));
    //Implement b4 == (a == b)
    cp_model.AddEquality(a, b).OnlyEnforceIf(b4);
    cp_model.AddNotEqual(a, b).OnlyEnforceIf(Not(b4));


    // b3 implies b4 and b4 implies b3
    cp_model.AddImplication(b3, b4);
    cp_model.AddImplication(b4, b3);

    const CpSolverResponse response = Solve(cp_model.Build());
    LOG(INFO) << CpSolverResponseStats(response);
    // [END solve]

    if (response.status() == CpSolverStatus::FEASIBLE) {

      LOG(INFO) << "a = " << SolutionIntegerValue(response, a);
      LOG(INFO) << "b = " << SolutionIntegerValue(response, b);
      LOG(INFO) << "c = " << SolutionIntegerValue(response, c);

      LOG(INFO) << "b1 = " << SolutionIntegerValue(response, b1);
      LOG(INFO) << "b2 = " << SolutionIntegerValue(response, b2);
      LOG(INFO) << "b3 = " << SolutionIntegerValue(response, b3);
      LOG(INFO) << "b4 = " << SolutionIntegerValue(response, b4);
    }
  }

  void test_one_and_minus_one(){
    // [START variables]
    IntVar a = cp_model.NewIntVar(Domain(1)).WithName("a");
    IntVar b = cp_model.NewIntVar(Domain(-1)).WithName("b");

    IntVar c = cp_model.NewIntVar(domain).WithName("c");

    BoolVar b1 = cp_model.NewBoolVar();
    BoolVar b2 = cp_model.NewBoolVar();

    // Implement b1 == (C == 0)
    cp_model.AddEquality(c, 0).OnlyEnforceIf(b1);
    cp_model.AddNotEqual(c, 0).OnlyEnforceIf(Not(b1));
    //Implement b2 == (a == 0)
    cp_model.AddEquality(a, 0).OnlyEnforceIf(b2);
    cp_model.AddNotEqual(a, 0).OnlyEnforceIf(Not(b2));

    // b1 implies b2 and b2 implies b1
    cp_model.AddImplication(b2, b1);
    cp_model.AddImplication(b1, b2);

    BoolVar b3 = cp_model.NewBoolVar();
    BoolVar b4 = cp_model.NewBoolVar();

    // Implement b3 == (c == 1)
    cp_model.AddEquality(c, 1).OnlyEnforceIf(b3);
    cp_model.AddNotEqual(c, 1).OnlyEnforceIf(Not(b3));
    //Implement b4 == (a == b)
    cp_model.AddEquality(a, b).OnlyEnforceIf(b4);
    cp_model.AddNotEqual(a, b).OnlyEnforceIf(Not(b4));


    // b3 implies b4 and b4 implies b3
    cp_model.AddImplication(b3, b4);
    cp_model.AddImplication(b4, b3);

    const CpSolverResponse response = Solve(cp_model.Build());
    LOG(INFO) << CpSolverResponseStats(response);
    // [END solve]

    if (response.status() == CpSolverStatus::FEASIBLE) {

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
    //operations_research::sat::test_zero();
    //operations_research::sat::test_one_and_one();
    //operations_research::sat::test_minus_one_and_minus_one();
    //operations_research::sat::test_minus_one_and_one();
    operations_research::sat::test_one_and_minus_one();

    return EXIT_SUCCESS;
  }
  // [END program]
