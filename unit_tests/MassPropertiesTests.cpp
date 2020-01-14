/*
 * PlatoAugLagStressTest.cpp
 *
 *  Created on: Feb 3, 2019
 */

#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoTestHelpers.hpp"

#include "Plato_Diagnostics.hpp"
#include "ScalarFunctionBase.hpp"
#include "WeightedSumFunction.hpp"
#include "PhysicsScalarFunction.hpp"
#include "MassPropertiesFunction.hpp"


namespace MassPropertiesTest
{

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassInsteadOfVolume2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->nelems();

    // Create control workset
    const Plato::Scalar tPseudoDensity = 0.8;
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(tPseudoDensity, tControl);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::fill(0.1, tState);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::WeightedSumFunction<Plato::Mechanics<tSpaceDim>> tWeightedSum(*tMesh, tDataMap);

    const Plato::Scalar tMaterialDensity = 0.5;
    const std::shared_ptr<Plato::MassMoment<Residual>> tCriterion = 
          std::make_shared<Plato::MassMoment<Residual>>(*tMesh, tMeshSets, tDataMap);
    tCriterion->setMaterialDensity(tMaterialDensity);
    tCriterion->setCalculationType("Mass");

    const std::shared_ptr<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>> tPhysicsScalarFunc = 
          std::make_shared<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>>(*tMesh, tDataMap);

    tPhysicsScalarFunc->allocateValue(tCriterion);
    
    const Plato::Scalar tFunctionWeight = 0.75;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFunc);
    tWeightedSum.appendFunctionWeight(tFunctionWeight);

    auto tObjFuncVal = tWeightedSum.value(tState, tControl, 0.0);

    Plato::Scalar tGoldValue = pow(static_cast<Plato::Scalar>(tMeshWidth), tSpaceDim) 
                               * tPseudoDensity * tFunctionWeight * tMaterialDensity;

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(tGoldValue, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassInsteadOfVolume3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->nelems();

    // Create control workset
    const Plato::Scalar tPseudoDensity = 0.8;
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(tPseudoDensity, tControl);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::fill(0.1, tState);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::WeightedSumFunction<Plato::Mechanics<tSpaceDim>> tWeightedSum(*tMesh, tDataMap);

    const Plato::Scalar tMaterialDensity = 0.5;
    const std::shared_ptr<Plato::MassMoment<Residual>> tCriterion = 
          std::make_shared<Plato::MassMoment<Residual>>(*tMesh, tMeshSets, tDataMap);
    tCriterion->setMaterialDensity(tMaterialDensity);
    tCriterion->setCalculationType("Mass");

    const std::shared_ptr<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>> tPhysicsScalarFunc = 
          std::make_shared<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>>(*tMesh, tDataMap);

    tPhysicsScalarFunc->allocateValue(tCriterion);
    
    const Plato::Scalar tFunctionWeight = 0.75;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFunc);
    tWeightedSum.appendFunctionWeight(tFunctionWeight);

    auto tObjFuncVal = tWeightedSum.value(tState, tControl, 0.0);

    Plato::Scalar tGoldValue = pow(static_cast<Plato::Scalar>(tMeshWidth), tSpaceDim) 
                               * tPseudoDensity * tFunctionWeight * tMaterialDensity;

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(tGoldValue, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPropertiesValue3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 15; // Need high mesh density in order to get correct inertias
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    //const Plato::OrdinalType tNumCells = tMesh->nelems();

    // Create control workset
    const Plato::Scalar tPseudoDensity = 0.8;
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(tPseudoDensity, tControl);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::fill(0.1, tState);

    Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                     \n"
    "  <Parameter name='Objective' type='string' value='My Mass Properties'/>                 \n"
    "  <ParameterList name='My Mass Properties'>                                              \n"
    "      <Parameter name='Type' type='string' value='Mass Properties'/>                     \n"
    "      <Parameter name='Properties' type='Array(string)' value='{Mass,CGx,CGy,CGz,Ixx,Iyy,Izz,Ixy,Iyz}'/>     \n"
    "      <Parameter name='Weights' type='Array(double)' value='{2.0,0.1,2.0,3.0,4.0,5.0,6.0,7.0,8.0}'/>         \n"
    "      <Parameter name='Gold Values' type='Array(double)' value='{0.2,0.05,0.55,0.75,0.5,0.5,0.5,0.3,0.3}'/>  \n"
    "  </ParameterList>                                                                       \n"
    "  <ParameterList name='Material Model'>                                                  \n"
    "      <Parameter  name='Density' type='double' value='0.5'/>                             \n"
    "  </ParameterList>                                                                       \n"
    "</ParameterList>                                                                         \n"
  );

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    std::string tFuncName = "My Mass Properties";
    Plato::MassPropertiesFunction<Plato::Mechanics<tSpaceDim>> 
          tMassProperties(*tMesh, tMeshSets, tDataMap, *tParams, tFuncName);

    auto tObjFuncVal = tMassProperties.value(tState, tControl, 0.0);

    Plato::Scalar tGoldValue = 2.0*pow((0.4-0.2)/0.2, 2) + 0.1*pow((0.5-0.05),2)
                             + 2.0*pow((0.5-0.55),2) + 3.0*pow((0.5-0.75),2)
                             + 4.0*pow((0.2666666-0.5)/0.5,2)
                             + 5.0*pow((0.2666666-0.5)/0.5,2)
                             + 6.0*pow((0.2666666-0.5)/0.5,2)
                             + 7.0*pow((-0.1-0.3)/0.3,2)
                             + 8.0*pow((-0.1-0.3)/0.3,2);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(tGoldValue, tObjFuncVal, tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPropertiesValue3DNormalized)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 15; // Need high mesh density in order to get correct inertias
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    //const Plato::OrdinalType tNumCells = tMesh->nelems();

    // Create control workset
    const Plato::Scalar tPseudoDensity = 0.8;
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(tPseudoDensity, tControl);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::fill(0.1, tState);

    Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                     \n"
    "  <Parameter name='Objective' type='string' value='My Mass Properties'/>                 \n"
    "  <ParameterList name='My Mass Properties'>                                              \n"
    "      <Parameter name='Type' type='string' value='Mass Properties'/>                     \n"
    "      <Parameter name='Properties' type='Array(string)' value='{Mass,CGx,CGy,CGz,Ixx,Iyy,Izz,Ixy,Ixz,Iyz}'/>     \n"
    "      <Parameter name='Weights' type='Array(double)' value='{2.0,0.1,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0}'/>         \n"
    "      <Parameter name='Gold Values' type='Array(double)' value='{0.2,0.05,0.55,0.75,5.4,5.5,5.4,-0.1,-0.1,-0.15}'/>  \n"
    "  </ParameterList>                                                                       \n"
    "  <ParameterList name='Material Model'>                                                  \n"
    "      <Parameter  name='Density' type='double' value='0.5'/>                             \n"
    "  </ParameterList>                                                                       \n"
    "</ParameterList>                                                                         \n"
  );

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    std::string tFuncName = "My Mass Properties";
    Plato::MassPropertiesFunction<Plato::Mechanics<tSpaceDim>> 
          tMassProperties(*tMesh, tMeshSets, tDataMap, *tParams, tFuncName);

    auto tObjFuncVal = tMassProperties.value(tState, tControl, 0.0);

    Plato::Scalar tGoldValue = 2.0*pow((0.4-0.2)/0.2, 2) + 0.1*pow((0.5-0.05),2)
                             + 2.0*pow((0.5-0.55),2) + 3.0*pow((0.5-0.75),2)
                             + 4.0*pow((-1.0589e-01-5.1241) /  5.1241,2)
                             + 5.0*pow((2.6130e-02-5.4403)  /  5.4403,2)
                             + 6.0*pow((1.8531e-01-5.3886)  /  5.3886,2)
                             + 7.0*pow((1.9408e-04-0.0000)  /  5.1241,2)
                             + 8.0*pow((9.5366e-02-0.0000)  /  5.1241,2)
                             + 9.0*pow((3.9663e-02-0.0000)  /  5.1241,2);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(tGoldValue, tObjFuncVal, tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPropertiesGradZ_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    //const Plato::OrdinalType tNumCells = tMesh->nelems();

    using GradientZ = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ;

    // Create control workset
    const Plato::Scalar tPseudoDensity = 0.8;
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(tPseudoDensity, tControl);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::fill(0.1, tState);

    Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <Parameter name='Objective' type='string' value='My Mass Properties'/>  \n"
    "  <ParameterList name='My Mass Properties'>                               \n"
    "      <Parameter name='Type' type='string' value='Mass Properties'/>      \n"
    "      <Parameter name='Properties' type='Array(string)' value='{Mass,CGx,CGy,CGz,Ixx,Iyy,Izz,Ixy,Iyz}'/>  \n"
    "      <Parameter name='Weights' type='Array(double)' value='{2.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0}'/>      \n"
    "      <Parameter name='Gold Values' type='Array(double)' value='{0.2,0.45,0.55,0.75,0.5,0.5,0.5,0.3,0.3}'/>  \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Material Model'>                                   \n"
    "      <Parameter  name='Density' type='double' value='0.5'/>              \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                          \n"
  );

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    std::string tFuncName = "My Mass Properties";
    Plato::MassPropertiesFunction<Plato::Mechanics<tSpaceDim>> 
          tMassProperties(*tMesh, tMeshSets, tDataMap, *tParams, tFuncName);

    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tMassProperties);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPropertiesGradU_3D) // All state derivatives should be zero!
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    //const Plato::OrdinalType tNumCells = tMesh->nelems();

    using Jacobian = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian;

    // Create control workset
    const Plato::Scalar tPseudoDensity = 0.8;
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(tPseudoDensity, tControl);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::fill(0.1, tState);

    Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <Parameter name='Objective' type='string' value='My Mass Properties'/>  \n"
    "  <ParameterList name='My Mass Properties'>                               \n"
    "      <Parameter name='Type' type='string' value='Mass Properties'/>      \n"
    "      <Parameter name='Properties' type='Array(string)' value='{Mass,CGx,CGy,CGz}'/>  \n"
    "      <Parameter name='Weights' type='Array(double)' value='{2.0,1.0,2.0,3.0}'/>      \n"
    "      <Parameter name='Gold Values' type='Array(double)' value='{0.2,0.45,0.55,0.75}'/>  \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Material Model'>                                   \n"
    "      <Parameter  name='Density' type='double' value='0.5'/>              \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                          \n"
  );

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    std::string tFuncName = "My Mass Properties";
    Plato::MassPropertiesFunction<Plato::Mechanics<tSpaceDim>> 
          tMassProperties(*tMesh, tMeshSets, tDataMap, *tParams, tFuncName);

    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tMassProperties);
}

} // namespace MassPropertiesTest