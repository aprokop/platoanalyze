/*!
  These unit tests are for the transient dynamics formulation
*/

#include "PlatoTestHelpers.hpp"
#include "PlatoTypes.hpp"
#include "Teuchos_UnitTestHarness.hpp"


#include "Simp.hpp"
#include "Ramp.hpp"
#include "Strain.hpp"
#include "Solutions.hpp"
#include "Heaviside.hpp"
#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "Plato_Solve.hpp"
#include "LinearStress.hpp"
#include "ProjectToNode.hpp"
#include "ApplyWeighting.hpp"
#include "StressDivergence.hpp"
#include "SimplexMechanics.hpp"
#include "ApplyConstraints.hpp"
#include "PlatoMathHelpers.hpp"
#include "InterpolateFromNodal.hpp"
#include "PlatoAbstractProblem.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "hyperbolic/HyperbolicVectorFunction.hpp"
#include "hyperbolic/HyperbolicPhysicsScalarFunction.hpp"
#include "hyperbolic/Newmark.hpp"
#include "hyperbolic/InertialContent.hpp"
#include "hyperbolic/ElastomechanicsResidual.hpp"
#include "hyperbolic/HyperbolicMechanics.hpp"
#include "hyperbolic/HyperbolicProblem.hpp"
#include <Teuchos_MathExpr.hpp>

template <class VectorFunctionT, class VectorT, class ControlT>
Plato::Scalar
testVectorFunction_Partial_z(
    VectorFunctionT& aVectorFunction,
    VectorT aU,
    VectorT aV,
    VectorT aA,
    ControlT aControl,
    Plato::Scalar aTimeStep)
{
    // compute initial R and dRdz
    auto tResidual = aVectorFunction.value(aU, aV, aA, aControl, aTimeStep);
    auto t_dRdz = aVectorFunction.gradient_z(aU, aV, aA, aControl, aTimeStep);

    Plato::ScalarVector tStep = Plato::ScalarVector("Step", aControl.extent(0));
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.025, 0.05, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);

    // compute F at z - step
    Plato::blas1::axpy(-1.0, tStep, aControl);
    auto tResidualNeg = aVectorFunction.value(aU, aV, aA, aControl, aTimeStep);

    // compute F at z + step
    Plato::blas1::axpy(2.0, tStep, aControl);
    auto tResidualPos = aVectorFunction.value(aU, aV, aA, aControl, aTimeStep);
    Plato::blas1::axpy(-1.0, tStep, aControl);

    // compute actual change in F over 2 * deltaZ
    Plato::blas1::axpy(-1.0, tResidualPos, tResidualNeg);
    auto tDeltaFD = Plato::blas1::norm(tResidualNeg);

    Plato::ScalarVector tDeltaR = Plato::ScalarVector("delta R", tResidual.extent(0));
    Plato::blas1::scale(2.0, tStep);
    Plato::VectorTimesMatrixPlusVector(tStep, t_dRdz, tDeltaR);
    auto tDeltaAD = Plato::blas1::norm(tDeltaR);

    // return error
    Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
    return std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
}

TEUCHOS_UNIT_TEST( TransientDynamicsProblemTests, 3D )
{
  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", cMeshWidth);

  // create input for transient mechanics problem
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>     \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>            \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>              \n"
    "  <ParameterList name='Hyperbolic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>         \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='Alyoominium'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Criteria'>                                         \n"
    "    <ParameterList name='Internal Energy'>                                \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>      \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/> \n"
    "      <ParameterList name='Penalty Function'>                             \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>               \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>            \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>       \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Alyoominium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
    "        <Parameter name='Mass Density' type='double' value='2.7'/>          \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>      \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>   \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Time Integration'>                                 \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>           \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>           \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>            \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>            \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                     \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>            \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>     \n"
    "      <Parameter name='Values' type='Array(double)' value='{1e3, 0, 0}'/> \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>          \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                          \n"
  );

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  auto* tHyperbolicProblem =
    new Plato::HyperbolicProblem<Plato::Hyperbolic::Mechanics<cSpaceDim>>
    (tMesh, *tInputParams, tMachine);

  TEST_ASSERT(tHyperbolicProblem != nullptr);

  int tNumDofs = cSpaceDim*tMesh->NumNodes();
  Plato::ScalarVector tControl("control", tNumDofs);
  Plato::blas1::fill(1.0, tControl);

  /*****************************************************
   Test HyperbolicProblem::solution(aControl);
   *****************************************************/

  auto tSolution = tHyperbolicProblem->solution(tControl);
  auto tDisplacements = tSolution.get("State");
  auto tDisplacement = Kokkos::subview(tDisplacements, /*tStepIndex*/1, Kokkos::ALL());

  auto tDisplacement_Host = Kokkos::create_mirror_view( tDisplacement );
  Kokkos::deep_copy( tDisplacement_Host, tDisplacement );

  std::vector<Plato::Scalar> tDisplacement_gold = {
    3.61863560324768500405784e-11,  3.83849353315283790119312e-12,
    3.83849353315283790119312e-12,  1.48267383576000436797713e-12,
   -2.25624866593761909787027e-13, -2.73367053814262282760693e-13,
   -2.42591058524479259361185e-11, -4.09157004346611346509595e-12,
   -6.54292514491087131336281e-12,  1.48267383576000436797713e-12,
   -2.73367053814262282760693e-13, -2.25624866593761909787027e-13
  };

  for(int iNode=0; iNode<int(tDisplacement_gold.size()); iNode++){
    if(tDisplacement_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(tDisplacement_Host[iNode]) < 1e-3);
    } else {
      TEST_FLOATING_EQUALITY(
        tDisplacement_Host[iNode],
        tDisplacement_gold[iNode], 1e-3);
    }
  }


  /*****************************************************
   Test HyperbolicProblem::criterionValue(aControl);
   *****************************************************/

  auto tCriterionValue = tHyperbolicProblem->criterionValue(tControl, "Internal Energy");
  Plato::Scalar tCriterionValue_gold = 5.43649521380863686677761e-9;

  TEST_FLOATING_EQUALITY( tCriterionValue, tCriterionValue_gold, 1e-4);


  /*********************************************************
   Test HyperbolicProblem::criterionValue(aControl, aState);
   *********************************************************/

  tCriterionValue = tHyperbolicProblem->criterionValue(tControl, tSolution, "Internal Energy");
  TEST_FLOATING_EQUALITY( tCriterionValue, tCriterionValue_gold, 1e-4);


  /*****************************************************
   Test HyperbolicProblem::criterionGradient(aControl);
   *****************************************************/

  auto tCriterionGradient = tHyperbolicProblem->criterionGradient(tControl, "Internal Energy");


  /**************************************************************
   The gradients below are verified with FD check elsewhere. The
   calls below are to catch any signals that may be thrown.
   **************************************************************/

  /************************************************************
   Call HyperbolicProblem::criterionGradient(aControl, aState);
   ************************************************************/

  tCriterionGradient = tHyperbolicProblem->criterionGradient(tControl, tSolution, "Internal Energy");


  /*****************************************************
   Call HyperbolicProblem::criterionGradientX(aControl);
   *****************************************************/

  auto tCriterionGradientX = tHyperbolicProblem->criterionGradientX(tControl, "Internal Energy");


  /************************************************************
   Call HyperbolicProblem::criterionGradientX(aControl, aState);
   ************************************************************/

  tCriterionGradientX = tHyperbolicProblem->criterionGradientX(tControl, tSolution, "Internal Energy");



  // test criterionValue
  //
  auto tConstraintValue = tHyperbolicProblem->criterionValue(tControl, "Internal Energy");


  // test criterionValue
  //
  tConstraintValue = tHyperbolicProblem->criterionValue(tControl, tSolution, "Internal Energy");


  // test criterionGradient
  //
  auto tConstraintGradient = tHyperbolicProblem->criterionGradient(tControl, "Internal Energy");


  // test criterionGradient
  //
  tConstraintGradient = tHyperbolicProblem->criterionGradient(tControl, tSolution, "Internal Energy");


  // test criterionGradientX
  //
  auto tConstraintGradientX = tHyperbolicProblem->criterionGradientX(tControl, "Internal Energy");


  // test criterionGradientX
  //
  tConstraintGradientX = tHyperbolicProblem->criterionGradientX(tControl, tSolution, "Internal Energy");

  delete tHyperbolicProblem;
}

TEUCHOS_UNIT_TEST( TransientMechanicsResidualTests, 3D_NoMass )
{
  // create input for transient mechanics residual
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                     \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>    \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>             \n"
    "  <ParameterList name='Hyperbolic'>                                      \n"
    "    <ParameterList name='Penalty Function'>                              \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>             \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>        \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                       \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='Soylent Green'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Material Models'>                                   \n"
    "    <ParameterList name='Soylent Green'>                                   \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                      \n"
    "        <Parameter name='Mass Density' type='double' value='0.0'/>         \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>  \n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                       \n"
    "  <ParameterList name='Time Integration'>                                \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>           \n"
    "    <Parameter name='Time Step' type='double' value='1.0'/>              \n"
    "  </ParameterList>                                                       \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                    \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>           \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>    \n"
    "      <Parameter name='Values' type='Array(double)' value='{1e3, 0, 0}'/>\n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>         \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                       \n"
    "</ParameterList>                                                         \n"
  );

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", cMeshWidth);

  Plato::SpatialModel tSpatialModel(tMesh, *tInputParams);

  Plato::DataMap tDataMap;
  Plato::Hyperbolic::VectorFunction<::Plato::Hyperbolic::Mechanics<cSpaceDim>>
    tVectorFunction(tSpatialModel, tDataMap, *tInputParams, tInputParams->get<std::string>("PDE Constraint"));

  int tNumDofs = cSpaceDim*tMesh->NumNodes();
  Plato::ScalarVector tControl("control", tMesh->NumNodes());
  Plato::blas1::fill(1.0, tControl);

  // create mesh based displacement from host data
  //
  std::vector<Plato::Scalar> u_host( cSpaceDim*tMesh->NumNodes() );
  Plato::Scalar disp = 0.0, dval = 1.0e-7;
  for( auto& val : u_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    u_host_view(u_host.data(),u_host.size());
  Plato::ScalarVector tU = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

  Plato::ScalarVector tV("Velocity", tNumDofs);
  Plato::blas1::fill(0.0, tV);

  Plato::ScalarVector tA("Acceleration", tNumDofs);
  Plato::blas1::fill(1000.0, tA);

  auto tTimeStep = tInputParams->sublist("Time Integration").get<Plato::Scalar>("Time Step");

  /**************************************
   Test VectorFunction ref value (Fext)
   **************************************/

  auto tResidualZero = tVectorFunction.value(tV, tV, tA, tControl, tTimeStep);
  auto tResidualZero_Host = Kokkos::create_mirror_view( tResidualZero );
  Kokkos::deep_copy( tResidualZero_Host, tResidualZero );

  std::vector<Plato::Scalar> tResidualZero_gold = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 83.33333333333333, 0, 0, 125.0000000000000,
    0, 0, 41.66666666666666, 0, 0, 125.0000000000000, 0, 0,
    250.0000000000000, 0, 0, 125.0000000000000, 0, 0, 41.66666666666666,
    0, 0, 125.0000000000000, 0, 0, 83.33333333333333, 0, 0
  };

  for(int iNode=0; iNode<int(tResidualZero_gold.size()); iNode++){
    if(tResidualZero_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(tResidualZero_Host[iNode]) < 1e-12);
    } else {
      // R(u) = Fint - Fext;
      TEST_FLOATING_EQUALITY(-tResidualZero_Host[iNode], tResidualZero_gold[iNode], 1e-6);
    }
  }

  /**************************************
   Test VectorFunction value
   **************************************/

  auto tResidual = tVectorFunction.value(tU, tV, tA, tControl, tTimeStep);
  auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
  Kokkos::deep_copy( tResidual_Host, tResidual );

  std::vector<Plato::Scalar> tResidual_gold = {
  -917857.142857142724,   -692857.142857142841, -617857.142857142724,
  -1.18928571428571409e6, -964285.714285714319, -262500.000000000116,
  -271428.571428571420,   -271428.571428571420,  355357.142857142782,
  -1.15178571428571432e6, -299999.999999999942, -851785.714285714086
  };

  for(int iNode=0; iNode<int(tResidual_gold.size()); iNode++){
    if(tResidual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(tResidual_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(tResidual_Host[iNode], tResidual_gold[iNode], 1e-13);
    }
  }

  /**************************************
   Test VectorFunction gradient wrt U
   **************************************/

  auto tJacobian = tVectorFunction.gradient_u(tU, tV, tA, tControl, tTimeStep);
  auto jac_entries = tJacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
    2.73809523809523804e11, 0.000000000000000000,    0.000000000000000000,
    0.000000000000000000,   2.73809523809523834e11,  0.000000000000000000,
    0.000000000000000000,   0.000000000000000000,    2.73809523809523804e11,
   -4.16666666666666718e10, 0.000000000000000000,    2.08333333333333359e10,
    0.000000000000000000,  -4.16666666666666718e10,  2.08333333333333359e10,
    5.35714285714285660e10, 5.35714285714285660e10, -1.90476190476190460e11
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }

  /**************************************
   Test VectorFunction gradient wrt V
   **************************************/

  auto tJacobianV = tVectorFunction.gradient_v(tU, tV, tA, tControl, tTimeStep);
  auto jacV_entries = tJacobianV->entries();
  auto jacV_entriesHost = Kokkos::create_mirror_view( jacV_entries );
  Kokkos::deep_copy(jacV_entriesHost, jacV_entries);

  std::vector<Plato::Scalar> gold_jacV_entries = {
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00
  };

  int jacV_entriesSize = gold_jacV_entries.size();
  for(int i=0; i<jacV_entriesSize; i++){
    if(gold_jacV_entries[i] == 0.0){
      TEST_ASSERT(fabs(jacV_entriesHost[i]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(jacV_entriesHost(i), gold_jacV_entries[i], 1.0e-15);
    }
  }

  /**************************************
   Test VectorFunction gradient wrt A
   **************************************/

  auto tJacobianA = tVectorFunction.gradient_a(tU, tV, tA, tControl, tTimeStep);
  auto jacA_entries = tJacobianA->entries();
  auto jacA_entriesHost = Kokkos::create_mirror_view( jacA_entries );
  Kokkos::deep_copy(jacA_entriesHost, jacA_entries);

  // density is zero, so mass matrix should be zeros
  std::vector<Plato::Scalar> gold_jacA_entries = {
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00
  };

  int jacA_entriesSize = gold_jacA_entries.size();
  for(int i=0; i<jacA_entriesSize; i++){
    if(gold_jacA_entries[i] == 0.0){
      TEST_ASSERT(fabs(jacA_entriesHost[i]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(jacA_entriesHost(i), gold_jacA_entries[i], 1.0e-15);
    }
  }

  /**************************************
   Test VectorFunction gradient wrt X
   **************************************/

  auto tGradientX = tVectorFunction.gradient_x(tU, tV, tA, tControl, tTimeStep);
  auto tGradientX_entries = tGradientX->entries();
  auto tGradientX_entriesHost = Kokkos::create_mirror_view( tGradientX_entries );
  Kokkos::deep_copy(tGradientX_entriesHost, tGradientX_entries);

  std::vector<Plato::Scalar> gold_tGradientX_entries = {
   -1.47857142857142817229033e6, -1.47857142857142840512097e6,
   -1.47857142857142840512097e6, -492857.142857142898719758,
   -492857.142857142956927419,   -492857.142857142724096775,
   -164285.714285714173456654,   -164285.714285714260768145,
   -164285.714285714202560484,   -189285.714285714377183467,
   -114285.714285714202560484,    360714.285714285622816533
  };

  int tGradientX_entriesSize = gold_tGradientX_entries.size();
  for(int i=0; i<tGradientX_entriesSize; i++){
    TEST_FLOATING_EQUALITY(tGradientX_entriesHost(i), gold_tGradientX_entries[i], 1.0e-14);
  }

  /**************************************
   Test VectorFunction gradient wrt Z
   **************************************/

  auto t_dRdz_error = testVectorFunction_Partial_z(tVectorFunction, tU, tV, tA, tControl, tTimeStep);
  TEST_ASSERT(t_dRdz_error < 1.0e-6);

  auto tGradientZ = tVectorFunction.gradient_z(tU, tV, tA, tControl, tTimeStep);
  auto tGradientZ_entries = tGradientZ->entries();
  auto tGradientZ_entriesHost = Kokkos::create_mirror_view( tGradientZ_entries );
  Kokkos::deep_copy(tGradientZ_entriesHost, tGradientZ_entries);

  std::vector<Plato::Scalar> gold_tGradientZ_entries = {
    -229464.285714285681024194, -173214.285714285710128024,
    -154464.285714285681024194, -67857.1428571428550640121,
    -67857.1428571428550640121,  88839.2857142856810241938,
    -58482.1428571428550640121,  98214.2857142857101280242,
    -58482.1428571428405120969, -126339.285714285695576109,
     30357.1428571428441500757,  30357.1428571428405120969
  };

  int tGradientZ_entriesSize = gold_tGradientZ_entries.size();
  for(int i=0; i<tGradientZ_entriesSize; i++){
    TEST_FLOATING_EQUALITY(tGradientZ_entriesHost(i), gold_tGradientZ_entries[i], 1.0e-14);
  }

}

TEUCHOS_UNIT_TEST( TransientMechanicsResidualTests, 3D_WithMass )
{
  // create input for transient mechanics residual
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                       \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>      \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>               \n"
    "  <ParameterList name='Hyperbolic'>                                        \n"
    "    <ParameterList name='Penalty Function'>                                \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>               \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>          \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                  \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='6061-T6 Aluminum'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Material Models'>                                   \n"
    "    <ParameterList name='6061-T6 Aluminum'>                                \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                      \n"
    "        <Parameter name='Mass Density' type='double' value='2.7'/>         \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>  \n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Time Integration'>                                  \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>            \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>            \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>             \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>             \n"
    "  </ParameterList>                                                         \n"
    "</ParameterList>                                                           \n"
  );

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", cMeshWidth);

  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tInputParams);
  Plato::Hyperbolic::VectorFunction<::Plato::Hyperbolic::Mechanics<cSpaceDim>>
    tVectorFunction(tSpatialModel, tDataMap, *tInputParams, tInputParams->get<std::string>("PDE Constraint"));

  int tNumDofs = cSpaceDim*tMesh->NumNodes();
  Plato::ScalarVector tControl("control", tNumDofs);
  Plato::blas1::fill(1.0, tControl);

  // create mesh based displacement from host data
  //
  std::vector<Plato::Scalar> u_host( cSpaceDim*tMesh->NumNodes() );
  Plato::Scalar disp = 0.0, dval = 1.0e-7;
  for( auto& val : u_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    u_host_view(u_host.data(),u_host.size());
  auto tU = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

  Plato::ScalarVector tV("Velocity", tNumDofs);
  Plato::blas1::fill(0.0, tV);

  Plato::ScalarVector tA("Acceleration", tNumDofs);
  Plato::blas1::fill(1000.0, tA);

  auto tTimeStep = tInputParams->sublist("Time Integration").get<Plato::Scalar>("Time Step");


  /**************************************
   Test VectorFunction value
   **************************************/

  auto tResidual = tVectorFunction.value(tU, tV, tA, tControl, tTimeStep);

  // TODO: test Residual
  auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
  Kokkos::deep_copy( tResidual_Host, tResidual );

  std::vector<Plato::Scalar> tResidual_gold = {
   -917772.767857142724,   -692772.767857142841,  -617772.767857142724,
   -1.18917321428571409e6, -964173.214285714319,  -262387.500000000116,
   -271400.446428571420,   -271400.446428571420,   355385.267857142782,
   -1.15167321428571432e6, -299887.499999999942,  -851673.214285714086
  };

  for(int iNode=0; iNode<int(tResidual_gold.size()); iNode++){
    if(tResidual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(tResidual_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(tResidual_Host[iNode], tResidual_gold[iNode], 1e-13);
    }
  }


  /**************************************
   Test VectorFunction gradient wrt U
   **************************************/

  auto tJacobian = tVectorFunction.gradient_u(tU, tV, tA, tControl, tTimeStep);
  auto jac_entries = tJacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  // mass is non-zero, but this shouldn't affect stiffness:
  std::vector<Plato::Scalar> gold_jac_entries = {
    2.73809523809523804e11, 0.000000000000000000,    0.000000000000000000,
    0.000000000000000000,   2.73809523809523834e11,  0.000000000000000000,
    0.000000000000000000,   0.000000000000000000,    2.73809523809523804e11,
   -4.16666666666666718e10, 0.000000000000000000,    2.08333333333333359e10,
    0.000000000000000000,  -4.16666666666666718e10,  2.08333333333333359e10,
    5.35714285714285660e10, 5.35714285714285660e10, -1.90476190476190460e11
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }


  /**************************************
   Test VectorFunction gradient wrt V
   **************************************/

  auto tJacobianV = tVectorFunction.gradient_v(tU, tV, tA, tControl, tTimeStep);
  auto jacV_entries = tJacobianV->entries();
  auto jacV_entriesHost = Kokkos::create_mirror_view( jacV_entries );
  Kokkos::deep_copy(jacV_entriesHost, jacV_entries);

  std::vector<Plato::Scalar> gold_jacV_entries = {
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00
  };

  int jacV_entriesSize = gold_jacV_entries.size();
  for(int i=0; i<jacV_entriesSize; i++){
    if(gold_jacV_entries[i] == 0.0){
      TEST_ASSERT(fabs(jacV_entriesHost[i]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(jacV_entriesHost(i), gold_jacV_entries[i], 1.0e-15);
    }
  }


  /**************************************
   Test VectorFunction gradient wrt A
   **************************************/

  auto tJacobianA = tVectorFunction.gradient_a(tU, tV, tA, tControl, tTimeStep);
  auto jacA_entries = tJacobianA->entries();
  auto jacA_entriesHost = Kokkos::create_mirror_view( jacA_entries );
  Kokkos::deep_copy(jacA_entriesHost, jacA_entries);

  std::vector<Plato::Scalar> gold_jacA_entries = {
    0.0210937500000000014, 0.00000000000000000,   0.00000000000000000,
      0.00000000000000000, 0.0210937500000000014, 0.00000000000000000,
      0.00000000000000000, 0.00000000000000000,   0.0210937500000000014
  };

  int jacA_entriesSize = gold_jacA_entries.size();
  for(int i=0; i<jacA_entriesSize; i++){
    if(gold_jacA_entries[i] == 0.0){
      TEST_ASSERT(fabs(jacA_entriesHost[i]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(jacA_entriesHost(i), gold_jacA_entries[i], 1.0e-15);
    }
  }


  /**************************************
   Test VectorFunction gradient wrt X
   **************************************/

  auto tGradientX = tVectorFunction.gradient_x(tU, tV, tA, tControl, tTimeStep);
  auto tGradientX_entries = tGradientX->entries();
  auto tGradientX_entriesHost = Kokkos::create_mirror_view( tGradientX_entries );
  Kokkos::deep_copy(tGradientX_entriesHost, tGradientX_entries);

  std::vector<Plato::Scalar> gold_tGradientX_entries = {
    -1.47862767857142817229033e6, -1.47862767857142840512097e6,
    -1.47862767857142840512097e6, -492913.392857142898719758,
    -492913.392857142956927419,   -492913.392857142724096775,
    -164341.964285714173456654,   -164341.964285714260768145,
    -164341.964285714202560484,   -189285.714285714377183467,
    -114285.714285714202560484,    360714.285714285622816533,
  };

  int tGradientX_entriesSize = gold_tGradientX_entries.size();
  for(int i=0; i<tGradientX_entriesSize; i++){
    TEST_FLOATING_EQUALITY(tGradientX_entriesHost(i), gold_tGradientX_entries[i], 1.0e-14);
  }


  /**************************************
   Test VectorFunction gradient wrt Z
   **************************************/

  auto tGradientZ = tVectorFunction.gradient_z(tU, tV, tA, tControl, tTimeStep);
  auto tGradientZ_entries = tGradientZ->entries();
  auto tGradientZ_entriesHost = Kokkos::create_mirror_view( tGradientZ_entries );
  Kokkos::deep_copy(tGradientZ_entriesHost, tGradientZ_entries);

  std::vector<Plato::Scalar> gold_tGradientZ_entries = {
   -229443.191964285681024194, -173193.191964285710128024,
   -154443.191964285681024194, -67850.1116071428550640121,
   -67850.1116071428550640121,  88846.3169642856810241938,
   -58475.1116071428550640121,  98221.3169642857101280242,
   -58475.1116071428405120969, -126332.254464285695576109,
    30364.1741071428441500757,  30364.1741071428405120969
  };

  int tGradientZ_entriesSize = gold_tGradientZ_entries.size();
  for(int i=0; i<tGradientZ_entriesSize; i++){
    TEST_FLOATING_EQUALITY(tGradientZ_entriesHost(i), gold_tGradientZ_entries[i], 1.0e-14);
  }

}

TEUCHOS_UNIT_TEST( TransientMechanicsResidualTests, NewmarkIntegratorUForm )
{
  // create input for transient mechanics residual
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                   \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>  \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>           \n"
    "  <ParameterList name='Hyperbolic'>                                    \n"
    "    <ParameterList name='Penalty Function'>                            \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>           \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>      \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>              \n"
    "    </ParameterList>                                                   \n"
    "  </ParameterList>                                                     \n"
    "  <ParameterList name='Material Models'>                               \n"
    "    <ParameterList name='Isotropic Linear Elastic'>                    \n"
    "      <Parameter name='Mass Density' type='double' value='2.7'/>       \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.36'/>   \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>\n"
    "    </ParameterList>                                                   \n"
    "  </ParameterList>                                                     \n"
    "  <ParameterList name='Time Integration'>                              \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>        \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>        \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>         \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>         \n"
    "  </ParameterList>                                                     \n"
    "</ParameterList>                                                       \n"
  );

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", cMeshWidth);
  int tNumDofs = cSpaceDim*tMesh->NumNodes();

  auto tIntegratorParams = tInputParams->sublist("Time Integration");
  Plato::NewmarkIntegratorUForm<Plato::Hyperbolic::Mechanics<cSpaceDim>> tIntegrator(tIntegratorParams, 1.0);

  Plato::Scalar tU_val = 1.0, tV_val = 2.0, tA_val = 3.0;
  Plato::Scalar tU_Prev_val = 3.0, tV_Prev_val = 2.0, tA_Prev_val = 1.0;

  Plato::ScalarVector tU("Displacement", tNumDofs);
  Plato::blas1::fill(tU_val, tU);

  Plato::ScalarVector tV("Velocity", tNumDofs);
  Plato::blas1::fill(tV_val, tV);

  Plato::ScalarVector tA("Acceleration", tNumDofs);
  Plato::blas1::fill(tA_val, tA);

  Plato::ScalarVector tU_Prev("Previous Displacement", tNumDofs);
  Plato::blas1::fill(tU_Prev_val, tU_Prev);

  Plato::ScalarVector tV_Prev("Previous Velocity", tNumDofs);
  Plato::blas1::fill(tV_Prev_val, tV_Prev);

  Plato::ScalarVector tA_Prev("Previous Acceleration", tNumDofs);
  Plato::blas1::fill(tA_Prev_val, tA_Prev);

  auto tTimeStep = tIntegratorParams.get<Plato::Scalar>("Time Step");
  auto tGamma    = tIntegratorParams.get<Plato::Scalar>("Newmark Gamma");
  auto tBeta     = tIntegratorParams.get<Plato::Scalar>("Newmark Beta");

  Plato::Scalar
    tU_pred_val = tU_Prev_val
                + tTimeStep*tV_Prev_val
                + tTimeStep*tTimeStep/2.0 * (1.0-2.0*tBeta)*tA_Prev_val;

  Plato::Scalar
    tV_pred_val = tV_Prev_val
                + (1.0-tGamma)*tTimeStep * tA_Prev_val;


  /**************************************
   Test Newmark integrator v_value
   **************************************/

  Plato::Scalar tTestVal_V = tV_val - tV_pred_val - tGamma/(tBeta*tTimeStep)*(tU_val - tU_pred_val);

  auto tResidualV = tIntegrator.v_value(tU, tU_Prev,
                                        tV, tV_Prev,
                                        tA, tA_Prev, tTimeStep);

  auto tResidualV_host = Kokkos::create_mirror_view( tResidualV );
  Kokkos::deep_copy(tResidualV_host, tResidualV);

  for( int iVal=0; iVal<tNumDofs; iVal++)
  {
      TEST_FLOATING_EQUALITY(tResidualV_host(iVal), tTestVal_V, 1.0e-15);
  }


  /**************************************
   Test Newmark integrator v_grad_u
   **************************************/

  auto tR_vu = tIntegrator.v_grad_u(tTimeStep);
  auto tTestVal_R_vu = -tGamma/(tBeta*tTimeStep);
  TEST_FLOATING_EQUALITY(tR_vu, tTestVal_R_vu, 1.0e-15);


  /**************************************
   Test Newmark integrator v_grad_u_prev
   **************************************/

  auto tR_vu_prev = tIntegrator.v_grad_u_prev(tTimeStep);
  auto tTestVal_R_vu_prev = tGamma/(tBeta*tTimeStep);
  TEST_FLOATING_EQUALITY(tR_vu_prev, tTestVal_R_vu_prev, 1.0e-15);


  /**************************************
   Test Newmark integrator v_grad_v_prev
   **************************************/

  auto tR_vv_prev = tIntegrator.v_grad_v_prev(tTimeStep);
  auto tTestVal_R_vv_prev = tGamma/tBeta - 1.0;
  TEST_FLOATING_EQUALITY(tR_vv_prev, tTestVal_R_vv_prev, 1.0e-15);


  /**************************************
   Test Newmark integrator v_grad_a_prev
   **************************************/

  auto tR_va_prev = tIntegrator.v_grad_a_prev(tTimeStep);
  auto tTestVal_R_va_prev = (tGamma/(2.0*tBeta) - 1.0) * tTimeStep;
  TEST_FLOATING_EQUALITY(tR_va_prev, tTestVal_R_va_prev, 1.0e-15);




  /**************************************
   Test Newmark integrator a_value
   **************************************/

  Plato::Scalar tTestVal_A = tA_val - 1.0/(tBeta*tTimeStep*tTimeStep)*(tU_val - tU_pred_val);

  auto tResidualA = tIntegrator.a_value(tU, tU_Prev,
                                        tV, tV_Prev,
                                        tA, tA_Prev, tTimeStep);

  auto tResidualA_host = Kokkos::create_mirror_view( tResidualA );
  Kokkos::deep_copy(tResidualA_host, tResidualA);

  for( int iVal=0; iVal<tNumDofs; iVal++)
  {
      TEST_FLOATING_EQUALITY(tResidualA_host(iVal), tTestVal_A, 1.0e-15);
  }


  /**************************************
   Test Newmark integrator a_grad_u
   **************************************/

  auto tR_au = tIntegrator.a_grad_u(tTimeStep);
  auto tTestVal_R_au = -1.0/(tBeta*tTimeStep*tTimeStep);
  TEST_FLOATING_EQUALITY(tR_au, tTestVal_R_au, 1.0e-15);


  /**************************************
   Test Newmark integrator a_grad_u_prev
   **************************************/

  auto tR_au_prev = tIntegrator.a_grad_u_prev(tTimeStep);
  auto tTestVal_R_au_prev = 1.0/(tBeta*tTimeStep*tTimeStep);
  TEST_FLOATING_EQUALITY(tR_au_prev, tTestVal_R_au_prev, 1.0e-15);


  /**************************************
   Test Newmark integrator a_grad_v_prev
   **************************************/

  auto tR_av_prev = tIntegrator.a_grad_v_prev(tTimeStep);
  auto tTestVal_R_av_prev = 1.0/(tBeta*tTimeStep);
  TEST_FLOATING_EQUALITY(tR_av_prev, tTestVal_R_av_prev, 1.0e-15);


  /**************************************
   Test Newmark integrator a_grad_a_prev
   **************************************/

  auto tR_aa_prev = tIntegrator.a_grad_a_prev(tTimeStep);
  auto tTestVal_R_aa_prev = 1.0/(2.0*tBeta) - 1.0;
  TEST_FLOATING_EQUALITY(tR_aa_prev, tTestVal_R_aa_prev, 1.0e-15);


}

TEUCHOS_UNIT_TEST( TransientMechanicsResidualTests, NewmarkIntegratorAForm )
{
  // create input for transient mechanics residual
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                   \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>  \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>           \n"
    "  <ParameterList name='Hyperbolic'>                                    \n"
    "    <ParameterList name='Penalty Function'>                            \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>           \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>      \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>              \n"
    "    </ParameterList>                                                   \n"
    "  </ParameterList>                                                     \n"
    "  <ParameterList name='Material Models'>                               \n"
    "    <ParameterList name='Isotropic Linear Elastic'>                    \n"
    "      <Parameter name='Mass Density' type='double' value='2.7'/>       \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.36'/>   \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>\n"
    "    </ParameterList>                                                   \n"
    "  </ParameterList>                                                     \n"
    "  <ParameterList name='Time Integration'>                              \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>        \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>        \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>         \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>         \n"
    "    <Parameter name='A-Form' type='bool' value='true'/>           \n"
    "  </ParameterList>                                                     \n"
    "</ParameterList>                                                       \n"
  );

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", cMeshWidth);
  int tNumDofs = cSpaceDim*tMesh->NumNodes();

  auto tIntegratorParams = tInputParams->sublist("Time Integration");
  Plato::NewmarkIntegratorAForm<Plato::Hyperbolic::Mechanics<cSpaceDim>> tIntegrator(tIntegratorParams, 1.0);

  Plato::Scalar tU_val = 1.0, tV_val = 2.0, tA_val = 3.0;
  Plato::Scalar tU_Prev_val = 3.0, tV_Prev_val = 2.0, tA_Prev_val = 1.0;

  Plato::ScalarVector tU("Displacement", tNumDofs);
  Plato::blas1::fill(tU_val, tU);

  Plato::ScalarVector tV("Velocity", tNumDofs);
  Plato::blas1::fill(tV_val, tV);

  Plato::ScalarVector tA("Acceleration", tNumDofs);
  Plato::blas1::fill(tA_val, tA);

  Plato::ScalarVector tU_Prev("Previous Displacement", tNumDofs);
  Plato::blas1::fill(tU_Prev_val, tU_Prev);

  Plato::ScalarVector tV_Prev("Previous Velocity", tNumDofs);
  Plato::blas1::fill(tV_Prev_val, tV_Prev);

  Plato::ScalarVector tA_Prev("Previous Acceleration", tNumDofs);
  Plato::blas1::fill(tA_Prev_val, tA_Prev);

  auto tTimeStep = tIntegratorParams.get<Plato::Scalar>("Time Step");
  auto tGamma    = tIntegratorParams.get<Plato::Scalar>("Newmark Gamma");
  auto tBeta     = tIntegratorParams.get<Plato::Scalar>("Newmark Beta");

  Plato::Scalar
    tU_pred_val = tU_Prev_val
                + tTimeStep*tV_Prev_val
                + tTimeStep*tTimeStep/2.0 * (1.0-2.0*tBeta)*tA_Prev_val;

  Plato::Scalar
    tV_pred_val = tV_Prev_val
                + (1.0-tGamma)*tTimeStep * tA_Prev_val;

  /**************************************
   Test Newmark integrator v_value
   **************************************/
  Plato::Scalar tTestVal_V = tV_val - tV_pred_val - tGamma*tTimeStep*tA_val;

  auto tResidualV = tIntegrator.v_value(tU, tU_Prev,
                                        tV, tV_Prev,
                                        tA, tA_Prev, tTimeStep);

  auto tResidualV_host = Kokkos::create_mirror_view( tResidualV );
  Kokkos::deep_copy(tResidualV_host, tResidualV);

  for( int iVal=0; iVal<tNumDofs; iVal++)
  {
      TEST_FLOATING_EQUALITY(tResidualV_host(iVal), tTestVal_V, 1.0e-15);
  }

  /**************************************
   Test Newmark integrator v_grad_a
   **************************************/
  auto tR_va = tIntegrator.v_grad_a(tTimeStep);
  auto tTestVal_R_va = -tGamma*tTimeStep;
  TEST_FLOATING_EQUALITY(tR_va, tTestVal_R_va, 1.0e-15);

  /**************************************
   Test Newmark integrator u_value
   **************************************/
  Plato::Scalar tTestVal_U = tU_val - tU_pred_val - tTimeStep*tTimeStep*tBeta*tA_val;

  auto tResidualU = tIntegrator.u_value(tU, tU_Prev,
                                        tV, tV_Prev,
                                        tA, tA_Prev, tTimeStep);

  auto tResidualU_host = Kokkos::create_mirror_view( tResidualU );
  Kokkos::deep_copy(tResidualU_host, tResidualU);

  for( int iVal=0; iVal<tNumDofs; iVal++)
  {
      TEST_FLOATING_EQUALITY(tResidualU_host(iVal), tTestVal_U, 1.0e-15);
  }

  /**************************************
   Test Newmark integrator u_grad_a
   **************************************/
  auto tR_ua = tIntegrator.u_grad_a(tTimeStep);
  auto tTestVal_R_ua = -tBeta*tTimeStep*tTimeStep;
  TEST_FLOATING_EQUALITY(tR_ua, tTestVal_R_ua, 1.0e-15);
}

TEUCHOS_UNIT_TEST( TransientMechanicsResidualTests, 3D_ScalarFunction )
{
  // create input for transient mechanics residual
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                   \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>  \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>           \n"
    "  <ParameterList name='Hyperbolic'>                                    \n"
    "    <ParameterList name='Penalty Function'>                            \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>           \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>      \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>              \n"
    "    </ParameterList>                                                   \n"
    "  </ParameterList>                                                     \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='6061-T6 Aluminum'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Criteria'>                                      \n"
    "    <ParameterList name='Internal Energy'>                             \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>   \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/> \n"
    "      <ParameterList name='Penalty Function'>                          \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>            \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>    \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>         \n"
    "      </ParameterList>                                                 \n"
    "    </ParameterList>                                                   \n"
    "  </ParameterList>                                                     \n"
    "  <ParameterList name='Material Models'>                                \n"
    "    <ParameterList name='6061-T6 Aluminum'>                              \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                    \n"
    "        <Parameter name='Mass Density' type='double' value='2.7'/>       \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>   \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>\n"
    "      </ParameterList>                                                   \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                     \n"
    "  <ParameterList name='Time Integration'>                              \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>        \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>        \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>         \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>         \n"
    "  </ParameterList>                                                     \n"
    "</ParameterList>                                                       \n"
  );

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", cMeshWidth);
  int tNumDofs = cSpaceDim*tMesh->NumNodes();

  Plato::DataMap tDataMap;
  std::string tMyFunction("Internal Energy");
  Plato::SpatialModel tSpatialModel(tMesh, *tInputParams);
  Plato::Hyperbolic::PhysicsScalarFunction<::Plato::Hyperbolic::Mechanics<cSpaceDim>>
    tScalarFunction(tSpatialModel, tDataMap, *tInputParams, tMyFunction);

  auto tTimeStep = tInputParams->sublist("Time Integration").get<Plato::Scalar>("Time Step");
  auto tNumSteps = tInputParams->sublist("Time Integration").get<int>("Number Time Steps");

  Plato::ScalarVector tControl("control", tNumDofs);
  Plato::blas1::fill(1.0, tControl);

  Plato::ScalarMultiVector tU("Displacement", tNumSteps, tNumDofs);
  Plato::ScalarMultiVector tV("Velocity",     tNumSteps, tNumDofs);
  Plato::ScalarMultiVector tA("Acceleration", tNumSteps, tNumDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumDofs), LAMBDA_EXPRESSION(int aDofOrdinal)
  {
    for(int i=0; i<tNumSteps; i++)
    {
      tU(i, aDofOrdinal) = 1.0*i*aDofOrdinal * 1.0e-7;
      tV(i, aDofOrdinal) = 2.0*i*aDofOrdinal * 1.0e-7;
      tA(i, aDofOrdinal) = 3.0*i*aDofOrdinal * 1.0e-7;
    }
  }, "initial data");


  /**************************************
   Test ScalarFunction value
   **************************************/
  Plato::Solutions tSolution;
  tSolution.set("State", tU);
  tSolution.set("StateDot", tV);
  tSolution.set("StateDotDot", tA);
  auto tValue = tScalarFunction.value(tSolution, tControl, tTimeStep);

  TEST_FLOATING_EQUALITY(tValue, 78.8914285714285767880938, 1.0e-15);


  /**************************************
   Test ScalarFunction gradient wrt U
   **************************************/

  int tStepIndex = 1;
  auto tObjGradU = tScalarFunction.gradient_u(tSolution, tControl, tStepIndex, tTimeStep);

  auto tObjGradU_Host = Kokkos::create_mirror_view( tObjGradU );
  Kokkos::deep_copy( tObjGradU_Host, tObjGradU );

  std::vector<Plato::Scalar> tObjGradU_gold = {
   -1.83571428571428544819355e6, -1.38571428571428568102419e6,
   -1.23571428571428568102419e6, -2.37857142857142817229033e6,
   -1.92857142857142840512097e6, -525000.000000000000000000,
   -542857.142857142724096775,   -542857.142857142724096775,
    710714.285714285448193550,   -2.30357142857142817229033e6,
   -600000.000000000116415322,   -1.70357142857142840512097e6
  };

  for(int iNode=0; iNode<int(tObjGradU_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(tObjGradU_Host[iNode], tObjGradU_gold[iNode], 1e-15);
  }


  /**************************************
   Test ScalarFunction gradient wrt X
   **************************************/

  auto tObjGradX = tScalarFunction.gradient_x(tSolution, tControl, tTimeStep);

  auto tObjGradX_Host = Kokkos::create_mirror_view( tObjGradX );
  Kokkos::deep_copy( tObjGradX_Host, tObjGradX );

  std::vector<Plato::Scalar> tObjGradX_gold = {
   17.4942857142857093322164,  1.44857142857142950909122,
  -3.89999999999999902300374,  16.2321428571428540976740,
  -1.16357142857142981107188,  2.89928571428571357770920,
  -1.26214285714285767703302, -2.61214285714285665562784,
   6.79928571428571260071294,  15.0171428571428560161394,
   8.29285714285714092852686, -7.09714285714285608719365
  };

  for(int iNode=0; iNode<int(tObjGradX_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(tObjGradX_Host[iNode], tObjGradX_gold[iNode], 1e-14);
  }


  /**************************************
   Test ScalarFunction gradient wrt Z
   **************************************/

  auto tObjGradZ = tScalarFunction.gradient_z(tSolution, tControl, tTimeStep);

  auto tObjGradZ_Host = Kokkos::create_mirror_view( tObjGradZ );
  Kokkos::deep_copy( tObjGradZ_Host, tObjGradZ );

  std::vector<Plato::Scalar> tObjGradZ_gold = {
   2.46535714285714258053872,  3.28714285714285647799215,
   0.821785714285713897453434, 3.28714285714285692208136,
   4.93071428571428427289902,  1.64357142857142801695147,
   0.821785714285714119498039, 1.64357142857142823899608,
   0.821785714285714008475736, 3.28714285714285647799215,
   4.93071428571428427289902,  1.64357142857142823899608
  };

  for(int iNode=0; iNode<int(tObjGradZ_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(tObjGradZ_Host[iNode], tObjGradZ_gold[iNode], 1e-15);
  }

}

TEUCHOS_UNIT_TEST( TimeDependentBCsTests, EssentialBoundaryDofValuesMatchFunctionValues )
{
  // create comm
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", cMeshWidth);

  // create input for problem
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>     \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>            \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>              \n"
    "  <ParameterList name='Hyperbolic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>         \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='Alyoominium'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Alyoominium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
    "        <Parameter name='Mass Density' type='double' value='2.7'/>          \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>      \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>   \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Time Integration'>                                 \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>           \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>           \n"
    "    <Parameter name='Number Time Steps' type='int' value='10'/>            \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>            \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='State Essential Boundary Conditions'>                     \n"
    "    <ParameterList  name='x+ Applied Displacement'>            \n"
    "      <Parameter name='Type'   type='string'        value='Time Dependent'/>     \n"
    "      <Parameter name='Index'  type='int'           value='0'/>     \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>          \n"
    "      <Parameter name='Function' type='string' value='w=2e-6; p=0.01; pi=3.14159264; p*sin(pi*t/w)'/> \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='State Dot Essential Boundary Conditions'>                     \n"
    "    <ParameterList  name='x+ Applied Velocity'>            \n"
    "      <Parameter name='Type'   type='string'        value='Time Dependent'/>     \n"
    "      <Parameter name='Index'  type='int'           value='0'/>     \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>          \n"
    "      <Parameter name='Function' type='string' value='w=2e-6; p=0.01; pi=3.14159264; p*(pi/w)*cos(pi*t/w)'/> \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='State Dot Dot Essential Boundary Conditions'>                     \n"
    "    <ParameterList  name='x+ Applied Acceleration'>            \n"
    "      <Parameter name='Type'   type='string'        value='Time Dependent'/>     \n"
    "      <Parameter name='Index'  type='int'           value='0'/>     \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>          \n"
    "      <Parameter name='Function' type='string' value='w=2e-6; p=0.01; pi=3.14159264; -p*(pi*pi/w/w)*sin(pi*t/w)'/> \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Linear Solver'>                              \n"
    "    <Parameter name='Solver Stack' type='string' value='Epetra'/>        \n"
    "    <Parameter name='Display Iterations' type='int' value='0'/>     \n"
    "    <Parameter name='Iterations' type='int' value='50'/>            \n"
    "    <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "  </ParameterList>                                                  \n"
    "</ParameterList>                                                          \n"
  );

  // construct problem
  //
  auto tHyperbolicProblem = 
    std::make_unique<Plato::HyperbolicProblem<Plato::Hyperbolic::Mechanics<cSpaceDim>>>
    (tMesh, *tInputParams, tMachine);

  //create control
  //
  int tNumDofs = cSpaceDim*tMesh->NumNodes();
  Plato::ScalarVector tControl("control", tNumDofs);
  Plato::blas1::fill(1.0, tControl);

  // initialize
  //
  std::vector<Plato::OrdinalType> tBcDofs = {54, 57, 60, 63, 66, 69, 72, 75, 78};
  Plato::ScalarVector tDisplacement("displacement", tNumDofs);
  Plato::ScalarVector tVelocity("velocity", tNumDofs);
  Plato::ScalarVector tAcceleration("acceleration", tNumDofs);

  // fill in function values at t=0
  //
  Plato::Scalar tTime = 0.0;
  tHyperbolicProblem->constrainFieldsAtBoundary(tDisplacement,tVelocity,tAcceleration,tTime);

  auto tVelocity_Host = Kokkos::create_mirror_view( tVelocity );
  Kokkos::deep_copy( tVelocity_Host, tVelocity );

  for(int iDof=0; iDof<int(tBcDofs.size()); iDof++){
      auto tDof = tBcDofs[iDof];
      TEST_FLOATING_EQUALITY(tVelocity_Host[tDof],15707.9632679, 1e-7);
  }

  auto tAcceleration_Host = Kokkos::create_mirror_view( tAcceleration );
  Kokkos::deep_copy( tAcceleration_Host, tAcceleration );

  for(int iDof=0; iDof<int(tBcDofs.size()); iDof++){
      auto tDof = tBcDofs[iDof];
      TEST_FLOATING_EQUALITY(tAcceleration_Host[tDof],0.0, 1e-7);
  }

  // fill in function values at t=1.5e-6
  //
  tTime = 1.5e-6;
  tHyperbolicProblem->constrainFieldsAtBoundary(tDisplacement,tVelocity,tAcceleration,tTime);

  Kokkos::deep_copy( tVelocity_Host, tVelocity );

  for(int iDof=0; iDof<int(tBcDofs.size()); iDof++){
      auto tDof = tBcDofs[iDof];
      TEST_FLOATING_EQUALITY(tVelocity_Host[tDof],-11107.207345395916, 1e-7);
  }

  Kokkos::deep_copy( tAcceleration_Host, tAcceleration );

  for(int iDof=0; iDof<int(tBcDofs.size()); iDof++){
      auto tDof = tBcDofs[iDof];
      TEST_FLOATING_EQUALITY(tAcceleration_Host[tDof],-17447160499.097198, 1e-7);
  }

}

TEUCHOS_UNIT_TEST( TransientMechanicsSolverTests, UFormAndAFormEquivalenceWithConstantNaturalBCs )
{
  // create comm
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", cMeshWidth);

  // create input for u-form problem
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParamsUForm =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>     \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>            \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>              \n"
    "  <ParameterList name='Hyperbolic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>         \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='Alyoominium'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Alyoominium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
    "        <Parameter name='Mass Density' type='double' value='2.7'/>          \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>      \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>   \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Time Integration'>                                 \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>           \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>           \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>            \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>            \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                     \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>            \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>     \n"
    "      <Parameter name='Values' type='Array(double)' value='{1e9, 0, 0}'/> \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>          \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Linear Solver'>                              \n"
    "    <Parameter name='Solver Stack' type='string' value='Epetra'/>        \n"
    "    <Parameter name='Display Iterations' type='int' value='0'/>     \n"
    "    <Parameter name='Iterations' type='int' value='50'/>            \n"
    "    <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "  </ParameterList>                                                  \n"
    "</ParameterList>                                                          \n"
  );

  // create input for a-form problem
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParamsAForm =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>     \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>            \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>              \n"
    "  <ParameterList name='Hyperbolic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>         \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='Alyoominium'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Alyoominium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
    "        <Parameter name='Mass Density' type='double' value='2.7'/>          \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>      \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>   \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Time Integration'>                                 \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>           \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>           \n"
    "    <Parameter name='A-Form' type='bool' value='true'/>           \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>            \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>            \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                     \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>            \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>     \n"
    "      <Parameter name='Values' type='Array(double)' value='{1e9, 0, 0}'/> \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>          \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Linear Solver'>                              \n"
    "    <Parameter name='Solver Stack' type='string' value='Epetra'/>        \n"
    "    <Parameter name='Display Iterations' type='int' value='0'/>     \n"
    "    <Parameter name='Iterations' type='int' value='50'/>            \n"
    "    <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "  </ParameterList>                                                  \n"
    "</ParameterList>                                                          \n"
  );

  // create u-form problem
  //
  auto tHyperbolicProblemUForm = 
    std::make_unique<Plato::HyperbolicProblem<Plato::Hyperbolic::Mechanics<cSpaceDim>>>
    (tMesh, *tInputParamsUForm, tMachine);

  // create a-form problem
  //
  auto tHyperbolicProblemAForm = 
    std::make_unique<Plato::HyperbolicProblem<Plato::Hyperbolic::Mechanics<cSpaceDim>>>
    (tMesh, *tInputParamsAForm, tMachine);

  // create control
  //
  int tNumDofs = cSpaceDim*tMesh->NumNodes();
  Plato::ScalarVector tControl("control", tNumDofs);
  Plato::blas1::fill(1.0, tControl);

  /*****************************************************
   Test Equivalence of U-Form and A-Form Solutions 
   *****************************************************/

  // displacements
  //
  auto tSolutionUForm = tHyperbolicProblemUForm->solution(tControl);
  auto tDisplacementsUForm = tSolutionUForm.get("State");
  auto tDisplacementUForm = Kokkos::subview(tDisplacementsUForm, /*tStepIndex*/1, Kokkos::ALL());

  auto tDisplacementUForm_Host = Kokkos::create_mirror_view( tDisplacementUForm );
  Kokkos::deep_copy( tDisplacementUForm_Host, tDisplacementUForm );

  auto tSolutionAForm = tHyperbolicProblemAForm->solution(tControl);
  auto tDisplacementsAForm = tSolutionAForm.get("State");
  auto tDisplacementAForm = Kokkos::subview(tDisplacementsAForm, /*tStepIndex*/1, Kokkos::ALL());

  auto tDisplacementAForm_Host = Kokkos::create_mirror_view( tDisplacementAForm );
  Kokkos::deep_copy( tDisplacementAForm_Host, tDisplacementAForm );

  // velocities
  //
  auto tVelocitiesUForm = tSolutionUForm.get("StateDot");
  auto tVelocityUForm = Kokkos::subview(tVelocitiesUForm, /*tStepIndex*/1, Kokkos::ALL());

  auto tVelocityUForm_Host = Kokkos::create_mirror_view( tVelocityUForm );
  Kokkos::deep_copy( tVelocityUForm_Host, tVelocityUForm );

  auto tVelocitiesAForm = tSolutionAForm.get("StateDot");
  auto tVelocityAForm = Kokkos::subview(tVelocitiesAForm, /*tStepIndex*/1, Kokkos::ALL());

  auto tVelocityAForm_Host = Kokkos::create_mirror_view( tVelocityAForm );
  Kokkos::deep_copy( tVelocityAForm_Host, tVelocityAForm );

  // accelerations
  //
  auto tAccelerationsUForm = tSolutionUForm.get("StateDotDot");
  auto tAccelerationUForm = Kokkos::subview(tAccelerationsUForm, /*tStepIndex*/1, Kokkos::ALL());

  auto tAccelerationUForm_Host = Kokkos::create_mirror_view( tAccelerationUForm );
  Kokkos::deep_copy( tAccelerationUForm_Host, tAccelerationUForm );

  auto tAccelerationsAForm = tSolutionAForm.get("StateDotDot");
  auto tAccelerationAForm = Kokkos::subview(tAccelerationsAForm, /*tStepIndex*/1, Kokkos::ALL());

  auto tAccelerationAForm_Host = Kokkos::create_mirror_view( tAccelerationAForm );
  Kokkos::deep_copy( tAccelerationAForm_Host, tAccelerationAForm );

  // test displacements
  //
  for(int iNode=0; iNode<int(tDisplacementUForm_Host.size()); iNode++){
    TEST_FLOATING_EQUALITY(
      tDisplacementAForm_Host[iNode],
      tDisplacementUForm_Host[iNode], 1e-12);
  }

  // test velocities
  //
  for(int iNode=0; iNode<int(tVelocityUForm_Host.size()); iNode++){
    TEST_FLOATING_EQUALITY(
      tVelocityAForm_Host[iNode],
      tVelocityUForm_Host[iNode], 1e-12);
  }

  // test accelerations
  //
  for(int iNode=0; iNode<int(tAccelerationUForm_Host.size()); iNode++){
    TEST_FLOATING_EQUALITY(
      tAccelerationAForm_Host[iNode],
      tAccelerationUForm_Host[iNode], 1e-12);
  }

}

TEUCHOS_UNIT_TEST( TransientMechanicsSolverTests, UFormAndAFormEquivalenceWithInitialConditions )
{
  // create comm
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", cMeshWidth);

  // create input for u-form problem
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParamsUForm =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>     \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>            \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>              \n"
    "  <ParameterList name='Hyperbolic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>         \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='Alyoominium'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Alyoominium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
    "        <Parameter name='Mass Density' type='double' value='2.7'/>          \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>      \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>   \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Time Integration'>                                 \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>           \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>           \n"
    "    <Parameter name='Number Time Steps' type='int' value='10'/>            \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>            \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='Computed Fields'>                     \n"
    "    <ParameterList  name='Initial X Displacement'>            \n"
    "      <Parameter name='Function' type='string' value='w=2e-6; p=0.01; pi=3.14159264; p*sin(pi*x/w)'/> \n"
    "    </ParameterList>                                                      \n"
    "    <ParameterList  name='Initial X Velocity'>            \n"
    "      <Parameter name='Function' type='string' value='w=2e-6; p=0.01; pi=3.14159264; p*(pi/w)*cos(pi*x/w)'/> \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='Initial State'>                     \n"
    "    <ParameterList  name='displacement X'>            \n"
    "      <Parameter name='Computed Field' type='string' value='Initial X Displacement'/> \n"
    "    </ParameterList>                                                      \n"
    "    <ParameterList  name='velocity X'>            \n"
    "      <Parameter name='Computed Field' type='string' value='Initial X Velocity'/> \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Linear Solver'>                              \n"
    "    <Parameter name='Solver Stack' type='string' value='Epetra'/>        \n"
    "    <Parameter name='Display Iterations' type='int' value='0'/>     \n"
    "    <Parameter name='Iterations' type='int' value='50'/>            \n"
    "    <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "  </ParameterList>                                                  \n"
    "</ParameterList>                                                          \n"
  );

  // create input for a-form problem
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParamsAForm =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>     \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>            \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>              \n"
    "  <ParameterList name='Hyperbolic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>         \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='Alyoominium'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Alyoominium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
    "        <Parameter name='Mass Density' type='double' value='2.7'/>          \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>      \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>   \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Time Integration'>                                 \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>           \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>           \n"
    "    <Parameter name='A-Form' type='bool' value='true'/>           \n"
    "    <Parameter name='Number Time Steps' type='int' value='10'/>            \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>            \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='Computed Fields'>                     \n"
    "    <ParameterList  name='Initial X Displacement'>            \n"
    "      <Parameter name='Function' type='string' value='w=2e-6; p=0.01; pi=3.14159264; p*sin(pi*x/w)'/> \n"
    "    </ParameterList>                                                      \n"
    "    <ParameterList  name='Initial X Velocity'>            \n"
    "      <Parameter name='Function' type='string' value='w=2e-6; p=0.01; pi=3.14159264; p*(pi/w)*cos(pi*x/w)'/> \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='Initial State'>                     \n"
    "    <ParameterList  name='displacement X'>            \n"
    "      <Parameter name='Computed Field' type='string' value='Initial X Displacement'/> \n"
    "    </ParameterList>                                                      \n"
    "    <ParameterList  name='velocity X'>            \n"
    "      <Parameter name='Computed Field' type='string' value='Initial X Velocity'/> \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Linear Solver'>                              \n"
    "    <Parameter name='Solver Stack' type='string' value='Epetra'/>        \n"
    "    <Parameter name='Display Iterations' type='int' value='0'/>     \n"
    "    <Parameter name='Iterations' type='int' value='50'/>            \n"
    "    <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "  </ParameterList>                                                  \n"
    "</ParameterList>                                                          \n"
  );

  // create u-form problem
  //
  auto tHyperbolicProblemUForm = 
    std::make_unique<Plato::HyperbolicProblem<Plato::Hyperbolic::Mechanics<cSpaceDim>>>
    (tMesh, *tInputParamsUForm, tMachine);

  // create a-form problem
  //
  auto tHyperbolicProblemAForm = 
    std::make_unique<Plato::HyperbolicProblem<Plato::Hyperbolic::Mechanics<cSpaceDim>>>
    (tMesh, *tInputParamsAForm, tMachine);

  // create control
  //
  int tNumDofs = cSpaceDim*tMesh->NumNodes();
  Plato::ScalarVector tControl("control", tNumDofs);
  Plato::blas1::fill(1.0, tControl);

  /*****************************************************
   Test Equivalence of U-Form and A-Form Solutions 
   *****************************************************/

  // displacements
  //
  auto tSolutionUForm = tHyperbolicProblemUForm->solution(tControl);
  auto tDisplacementsUForm = tSolutionUForm.get("State");
  auto tDisplacementUForm = Kokkos::subview(tDisplacementsUForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tDisplacementUForm_Host = Kokkos::create_mirror_view( tDisplacementUForm );
  Kokkos::deep_copy( tDisplacementUForm_Host, tDisplacementUForm );

  auto tSolutionAForm = tHyperbolicProblemAForm->solution(tControl);
  auto tDisplacementsAForm = tSolutionAForm.get("State");
  auto tDisplacementAForm = Kokkos::subview(tDisplacementsAForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tDisplacementAForm_Host = Kokkos::create_mirror_view( tDisplacementAForm );
  Kokkos::deep_copy( tDisplacementAForm_Host, tDisplacementAForm );

  // velocities
  //
  auto tVelocitiesUForm = tSolutionUForm.get("StateDot");
  auto tVelocityUForm = Kokkos::subview(tVelocitiesUForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tVelocityUForm_Host = Kokkos::create_mirror_view( tVelocityUForm );
  Kokkos::deep_copy( tVelocityUForm_Host, tVelocityUForm );

  auto tVelocitiesAForm = tSolutionAForm.get("StateDot");
  auto tVelocityAForm = Kokkos::subview(tVelocitiesAForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tVelocityAForm_Host = Kokkos::create_mirror_view( tVelocityAForm );
  Kokkos::deep_copy( tVelocityAForm_Host, tVelocityAForm );

  // accelerations
  //
  auto tAccelerationsUForm = tSolutionUForm.get("StateDotDot");
  auto tAccelerationUForm = Kokkos::subview(tAccelerationsUForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tAccelerationUForm_Host = Kokkos::create_mirror_view( tAccelerationUForm );
  Kokkos::deep_copy( tAccelerationUForm_Host, tAccelerationUForm );

  auto tAccelerationsAForm = tSolutionAForm.get("StateDotDot");
  auto tAccelerationAForm = Kokkos::subview(tAccelerationsAForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tAccelerationAForm_Host = Kokkos::create_mirror_view( tAccelerationAForm );
  Kokkos::deep_copy( tAccelerationAForm_Host, tAccelerationAForm );

  // test displacements
  //
  for(int iNode=0; iNode<int(tDisplacementUForm_Host.size()); iNode++){
    TEST_FLOATING_EQUALITY(
      tDisplacementAForm_Host[iNode],
      tDisplacementUForm_Host[iNode], 1e-5);
  }

  // test velocities
  //
  for(int iNode=0; iNode<int(tVelocityUForm_Host.size()); iNode++){
    TEST_FLOATING_EQUALITY(
      tVelocityAForm_Host[iNode],
      tVelocityUForm_Host[iNode], 1e-5);
  }

  // test accelerations
  //
  for(int iNode=0; iNode<int(tAccelerationUForm_Host.size()); iNode++){
    TEST_FLOATING_EQUALITY(
      tAccelerationAForm_Host[iNode],
      tAccelerationUForm_Host[iNode], 1e-5);
  }

}

TEUCHOS_UNIT_TEST( TransientMechanicsSolverTests, UFormAndAFormEquivalenceWithTimeDependentNaturalBCs )
{
  // create comm
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", cMeshWidth);

  // create input for u-form problem
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParamsUForm =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>     \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>            \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>              \n"
    "  <ParameterList name='Hyperbolic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>         \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='Alyoominium'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Alyoominium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
    "        <Parameter name='Mass Density' type='double' value='2.7'/>          \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>      \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>   \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Time Integration'>                                 \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>           \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>           \n"
    "    <Parameter name='Number Time Steps' type='int' value='10'/>            \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>            \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                     \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>            \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>     \n"
    "      <Parameter name='Values' type='Array(string)' value='{0.0, w=2e-6; p=1.5e8; pi=3.14159264; p*sin(pi*t/w)^2, 0.0}'/> \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>          \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='State Essential Boundary Conditions'>                     \n"
    "    <ParameterList  name='X fixed'>            \n"
    "      <Parameter name='Type'   type='string' value='Zero Value'/>     \n"
    "      <Parameter name='Index'  type='int'    value='0'/>     \n"
    "      <Parameter name='Sides'  type='string' value='x-'/>     \n"
    "    </ParameterList>                                                      \n"
    "    <ParameterList  name='Y fixed'>            \n"
    "      <Parameter name='Type'   type='string' value='Zero Value'/>     \n"
    "      <Parameter name='Index'  type='int'    value='1'/>     \n"
    "      <Parameter name='Sides'  type='string' value='x+'/>     \n"
    "    </ParameterList>                                                      \n"
    "    <ParameterList  name='Z fixed'>            \n"
    "      <Parameter name='Type'   type='string' value='Zero Value'/>     \n"
    "      <Parameter name='Index'  type='int'    value='2'/>     \n"
    "      <Parameter name='Sides'  type='string' value='x+'/>     \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Linear Solver'>                              \n"
    "    <Parameter name='Solver Stack' type='string' value='Epetra'/>        \n"
    "    <Parameter name='Display Iterations' type='int' value='0'/>     \n"
    "    <Parameter name='Iterations' type='int' value='50'/>            \n"
    "    <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "  </ParameterList>                                                  \n"
    "</ParameterList>                                                          \n"
  );

  // create input for a-form problem
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParamsAForm =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>     \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>            \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>              \n"
    "  <ParameterList name='Hyperbolic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>         \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='Alyoominium'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Alyoominium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
    "        <Parameter name='Mass Density' type='double' value='2.7'/>          \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>      \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>   \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Time Integration'>                                 \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>           \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>           \n"
    "    <Parameter name='A-Form' type='bool' value='true'/>           \n"
    "    <Parameter name='Number Time Steps' type='int' value='10'/>            \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>            \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                     \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>            \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>     \n"
    "      <Parameter name='Values' type='Array(string)' value='{0.0, w=2e-6; p=1.5e8; pi=3.14159264; p*sin(pi*t/w)^2, 0.0}'/> \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>          \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='State Dot Dot Essential Boundary Conditions'>                     \n"
    "    <ParameterList  name='X fixed'>            \n"
    "      <Parameter name='Type'   type='string' value='Zero Value'/>     \n"
    "      <Parameter name='Index'  type='int'    value='0'/>     \n"
    "      <Parameter name='Sides'  type='string' value='x-'/>     \n"
    "    </ParameterList>                                                      \n"
    "    <ParameterList  name='Y fixed'>            \n"
    "      <Parameter name='Type'   type='string' value='Zero Value'/>     \n"
    "      <Parameter name='Index'  type='int'    value='1'/>     \n"
    "      <Parameter name='Sides'  type='string' value='x+'/>     \n"
    "    </ParameterList>                                                      \n"
    "    <ParameterList  name='Z fixed'>            \n"
    "      <Parameter name='Type'   type='string' value='Zero Value'/>     \n"
    "      <Parameter name='Index'  type='int'    value='2'/>     \n"
    "      <Parameter name='Sides'  type='string' value='x+'/>     \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Linear Solver'>                              \n"
    "    <Parameter name='Solver Stack' type='string' value='Epetra'/>        \n"
    "    <Parameter name='Display Iterations' type='int' value='0'/>     \n"
    "    <Parameter name='Iterations' type='int' value='50'/>            \n"
    "    <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "  </ParameterList>                                                  \n"
    "</ParameterList>                                                          \n"
  );

  // create u-form problem
  //
  auto tHyperbolicProblemUForm = 
    std::make_unique<Plato::HyperbolicProblem<Plato::Hyperbolic::Mechanics<cSpaceDim>>>
    (tMesh, *tInputParamsUForm, tMachine);

  // create a-form problem
  //
  auto tHyperbolicProblemAForm = 
    std::make_unique<Plato::HyperbolicProblem<Plato::Hyperbolic::Mechanics<cSpaceDim>>>
    (tMesh, *tInputParamsAForm, tMachine);

  // create control
  //
  int tNumDofs = cSpaceDim*tMesh->NumNodes();
  Plato::ScalarVector tControl("control", tNumDofs);
  Plato::blas1::fill(1.0, tControl);

  /*****************************************************
   Test Equivalence of U-Form and A-Form Solutions 
   *****************************************************/

  // displacements
  //
  auto tSolutionUForm = tHyperbolicProblemUForm->solution(tControl);
  auto tDisplacementsUForm = tSolutionUForm.get("State");
  auto tDisplacementUForm = Kokkos::subview(tDisplacementsUForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tDisplacementUForm_Host = Kokkos::create_mirror_view( tDisplacementUForm );
  Kokkos::deep_copy( tDisplacementUForm_Host, tDisplacementUForm );

  auto tSolutionAForm = tHyperbolicProblemAForm->solution(tControl);
  auto tDisplacementsAForm = tSolutionAForm.get("State");
  auto tDisplacementAForm = Kokkos::subview(tDisplacementsAForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tDisplacementAForm_Host = Kokkos::create_mirror_view( tDisplacementAForm );
  Kokkos::deep_copy( tDisplacementAForm_Host, tDisplacementAForm );

  // velocities
  //
  auto tVelocitiesUForm = tSolutionUForm.get("StateDot");
  auto tVelocityUForm = Kokkos::subview(tVelocitiesUForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tVelocityUForm_Host = Kokkos::create_mirror_view( tVelocityUForm );
  Kokkos::deep_copy( tVelocityUForm_Host, tVelocityUForm );

  auto tVelocitiesAForm = tSolutionAForm.get("StateDot");
  auto tVelocityAForm = Kokkos::subview(tVelocitiesAForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tVelocityAForm_Host = Kokkos::create_mirror_view( tVelocityAForm );
  Kokkos::deep_copy( tVelocityAForm_Host, tVelocityAForm );

  // accelerations
  //
  auto tAccelerationsUForm = tSolutionUForm.get("StateDotDot");
  auto tAccelerationUForm = Kokkos::subview(tAccelerationsUForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tAccelerationUForm_Host = Kokkos::create_mirror_view( tAccelerationUForm );
  Kokkos::deep_copy( tAccelerationUForm_Host, tAccelerationUForm );

  auto tAccelerationsAForm = tSolutionAForm.get("StateDotDot");
  auto tAccelerationAForm = Kokkos::subview(tAccelerationsAForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tAccelerationAForm_Host = Kokkos::create_mirror_view( tAccelerationAForm );
  Kokkos::deep_copy( tAccelerationAForm_Host, tAccelerationAForm );

  // test displacements
  //
  for(int iNode=0; iNode<int(tDisplacementUForm_Host.size()); iNode++){
    TEST_FLOATING_EQUALITY(
      tDisplacementAForm_Host[iNode],
      tDisplacementUForm_Host[iNode], 1e-8);
  }

  // test velocities
  //
  for(int iNode=0; iNode<int(tVelocityUForm_Host.size()); iNode++){
    TEST_FLOATING_EQUALITY(
      tVelocityAForm_Host[iNode],
      tVelocityUForm_Host[iNode], 1e-8);
  }

  // test accelerations
  //
  for(int iNode=0; iNode<int(tAccelerationUForm_Host.size()); iNode++){
    TEST_FLOATING_EQUALITY(
      tAccelerationAForm_Host[iNode],
      tAccelerationUForm_Host[iNode], 1e-8);
  }

}

TEUCHOS_UNIT_TEST( TransientMechanicsSolverTests, UFormAndAFormEquivalenceWithTimeDependentEssentialBCs )
{
  // create comm
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", cMeshWidth);

  // create input for u-form problem
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParamsUForm =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>     \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>            \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>              \n"
    "  <ParameterList name='Hyperbolic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>         \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='Alyoominium'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Alyoominium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
    "        <Parameter name='Mass Density' type='double' value='2.7'/>          \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>      \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>   \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Time Integration'>                                 \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>           \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>           \n"
    "    <Parameter name='Number Time Steps' type='int' value='10'/>            \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>            \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='State Essential Boundary Conditions'>                     \n"
    "    <ParameterList  name='x+ Applied Displacement'>            \n"
    "      <Parameter name='Type'   type='string'        value='Time Dependent'/>     \n"
    "      <Parameter name='Index'  type='int'           value='0'/>     \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>          \n"
    "      <Parameter name='Function' type='string' value='w=2e-6; p=0.01; pi=3.14159264; p*sin(pi*t/w)'/> \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='State Dot Essential Boundary Conditions'>                     \n"
    "    <ParameterList  name='x+ Applied Velocity'>            \n"
    "      <Parameter name='Type'   type='string'        value='Time Dependent'/>     \n"
    "      <Parameter name='Index'  type='int'           value='0'/>     \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>          \n"
    "      <Parameter name='Function' type='string' value='w=2e-6; p=0.01; pi=3.14159264; p*(pi/w)*cos(pi*t/w)'/> \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='State Dot Dot Essential Boundary Conditions'>                     \n"
    "    <ParameterList  name='x+ Applied Acceleration'>            \n"
    "      <Parameter name='Type'   type='string'        value='Time Dependent'/>     \n"
    "      <Parameter name='Index'  type='int'           value='0'/>     \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>          \n"
    "      <Parameter name='Function' type='string' value='w=2e-6; p=0.01; pi=3.14159264; -p*(pi*pi/w/w)*sin(pi*t/w)'/> \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Linear Solver'>                              \n"
    "    <Parameter name='Solver Stack' type='string' value='Epetra'/>        \n"
    "    <Parameter name='Display Iterations' type='int' value='0'/>     \n"
    "    <Parameter name='Iterations' type='int' value='50'/>            \n"
    "    <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "  </ParameterList>                                                  \n"
    "</ParameterList>                                                          \n"
  );

  // create input for a-form problem
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParamsAForm =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>     \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>            \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>              \n"
    "  <ParameterList name='Hyperbolic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>         \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='Alyoominium'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Alyoominium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
    "        <Parameter name='Mass Density' type='double' value='2.7'/>          \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>      \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>   \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Time Integration'>                                 \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>           \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>           \n"
    "    <Parameter name='A-Form' type='bool' value='true'/>           \n"
    "    <Parameter name='Number Time Steps' type='int' value='10'/>            \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>            \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='State Essential Boundary Conditions'>                     \n"
    "    <ParameterList  name='x+ Applied Displacement'>            \n"
    "      <Parameter name='Type'   type='string'        value='Time Dependent'/>     \n"
    "      <Parameter name='Index'  type='int'           value='0'/>     \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>          \n"
    "      <Parameter name='Function' type='string' value='w=2e-6; p=0.01; pi=3.14159264; p*sin(pi*t/w)'/> \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='State Dot Essential Boundary Conditions'>                     \n"
    "    <ParameterList  name='x+ Applied Velocity'>            \n"
    "      <Parameter name='Type'   type='string'        value='Time Dependent'/>     \n"
    "      <Parameter name='Index'  type='int'           value='0'/>     \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>          \n"
    "      <Parameter name='Function' type='string' value='w=2e-6; p=0.01; pi=3.14159264; p*(pi/w)*cos(pi*t/w)'/> \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='State Dot Dot Essential Boundary Conditions'>                     \n"
    "    <ParameterList  name='x+ Applied Acceleration'>            \n"
    "      <Parameter name='Type'   type='string'        value='Time Dependent'/>     \n"
    "      <Parameter name='Index'  type='int'           value='0'/>     \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>          \n"
    "      <Parameter name='Function' type='string' value='w=2e-6; p=0.01; pi=3.14159264; -p*(pi*pi/w/w)*sin(pi*t/w)'/> \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Linear Solver'>                              \n"
    "    <Parameter name='Solver Stack' type='string' value='Epetra'/>        \n"
    "    <Parameter name='Display Iterations' type='int' value='0'/>     \n"
    "    <Parameter name='Iterations' type='int' value='50'/>            \n"
    "    <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "  </ParameterList>                                                  \n"
    "</ParameterList>                                                          \n"
  );

  // create u-form problem
  //
  auto tHyperbolicProblemUForm = 
    std::make_unique<Plato::HyperbolicProblem<Plato::Hyperbolic::Mechanics<cSpaceDim>>>
    (tMesh, *tInputParamsUForm, tMachine);

  // create a-form problem
  //
  auto tHyperbolicProblemAForm = 
    std::make_unique<Plato::HyperbolicProblem<Plato::Hyperbolic::Mechanics<cSpaceDim>>>
    (tMesh, *tInputParamsAForm, tMachine);

  // create control
  //
  int tNumDofs = cSpaceDim*tMesh->NumNodes();
  Plato::ScalarVector tControl("control", tNumDofs);
  Plato::blas1::fill(1.0, tControl);

  /*****************************************************
   Test Equivalence of U-Form and A-Form Solutions 
   *****************************************************/

  // displacements
  //
  auto tSolutionUForm = tHyperbolicProblemUForm->solution(tControl);
  auto tDisplacementsUForm = tSolutionUForm.get("State");
  auto tDisplacementUForm = Kokkos::subview(tDisplacementsUForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tDisplacementUForm_Host = Kokkos::create_mirror_view( tDisplacementUForm );
  Kokkos::deep_copy( tDisplacementUForm_Host, tDisplacementUForm );

  auto tSolutionAForm = tHyperbolicProblemAForm->solution(tControl);
  auto tDisplacementsAForm = tSolutionAForm.get("State");
  auto tDisplacementAForm = Kokkos::subview(tDisplacementsAForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tDisplacementAForm_Host = Kokkos::create_mirror_view( tDisplacementAForm );
  Kokkos::deep_copy( tDisplacementAForm_Host, tDisplacementAForm );

  // velocities
  //
  auto tVelocitiesUForm = tSolutionUForm.get("StateDot");
  auto tVelocityUForm = Kokkos::subview(tVelocitiesUForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tVelocityUForm_Host = Kokkos::create_mirror_view( tVelocityUForm );
  Kokkos::deep_copy( tVelocityUForm_Host, tVelocityUForm );

  auto tVelocitiesAForm = tSolutionAForm.get("StateDot");
  auto tVelocityAForm = Kokkos::subview(tVelocitiesAForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tVelocityAForm_Host = Kokkos::create_mirror_view( tVelocityAForm );
  Kokkos::deep_copy( tVelocityAForm_Host, tVelocityAForm );

  // accelerations
  //
  auto tAccelerationsUForm = tSolutionUForm.get("StateDotDot");
  auto tAccelerationUForm = Kokkos::subview(tAccelerationsUForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tAccelerationUForm_Host = Kokkos::create_mirror_view( tAccelerationUForm );
  Kokkos::deep_copy( tAccelerationUForm_Host, tAccelerationUForm );

  auto tAccelerationsAForm = tSolutionAForm.get("StateDotDot");
  auto tAccelerationAForm = Kokkos::subview(tAccelerationsAForm, /*tStepIndex*/9, Kokkos::ALL());

  auto tAccelerationAForm_Host = Kokkos::create_mirror_view( tAccelerationAForm );
  Kokkos::deep_copy( tAccelerationAForm_Host, tAccelerationAForm );

  // test displacements
  //
  for(int iNode=0; iNode<int(tDisplacementUForm_Host.size()); iNode++){
    TEST_FLOATING_EQUALITY(
      tDisplacementAForm_Host[iNode],
      tDisplacementUForm_Host[iNode], 1.6e-0);
  }

  // test velocities
  //
  for(int iNode=0; iNode<int(tVelocityUForm_Host.size()); iNode++){
    TEST_FLOATING_EQUALITY(
      tVelocityAForm_Host[iNode],
      tVelocityUForm_Host[iNode], 1e-0);
  }

  // test accelerations
  //
  for(int iNode=0; iNode<int(tAccelerationUForm_Host.size()); iNode++){
    TEST_FLOATING_EQUALITY(
      tAccelerationAForm_Host[iNode],
      tAccelerationUForm_Host[iNode], 1e-0);
  }

}
