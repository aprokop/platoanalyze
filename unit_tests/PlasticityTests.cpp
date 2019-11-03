
// J2 Local Plasticity Equations Unit Tests

#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoTestHelpers.hpp"

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>

#include "plato/Plato_Diagnostics.hpp"

#include "plato/LocalVectorFunctionInc.hpp"
#include "plato/J2PlasticityLocalResidual.hpp"


namespace PlasticityTests
{

using namespace PlatoUtestHelpers;

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_UpdatePlasticStep2D)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumStressTerms = 4;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 10;
    
    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0);
            //printf("PrevLocalStates(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal", tNumCells, tNumStressTerms);
    auto tHostYieldSurfaceNormal = Kokkos::create_mirror(tYieldSurfaceNormal);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumStressTerms; ++j)
        {
            tHostYieldSurfaceNormal(i, j) = (i + 1.0) * (j + 2.0)/ 5.0;
            //printf("YieldSurfaceNormal(%d,%d) = %f\n", i,j,tHostYieldSurfaceNormal(i, j));
        }
    Kokkos::deep_copy(tYieldSurfaceNormal, tHostYieldSurfaceNormal);
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    Kokkos::deep_copy(tLocalState, tPrevLocalState);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Plato::Scalar tPenalizedHardeningModulusKinematic = 3.0;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.updatePlasticStrainAndBackstressPlasticStep(tCellOrdinal, tPrevLocalState, tYieldSurfaceNormal,
                                                                       tPenalizedHardeningModulusKinematic, tLocalState);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = 
      {{tHostPrevLocalState(0,0),tHostPrevLocalState(0,1),3.9798,5.4697,8.91918,8.44949,8.95959,10.9394,12.9192,14.899},
       {tHostPrevLocalState(1,0),tHostPrevLocalState(1,1),9.9192,13.8788,25.6767,21.798,21.8384,27.7576,33.6767,39.5959}};
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    Kokkos::deep_copy(tHostLocalState, tLocalState);

    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; ++tCellIndex)
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumLocalDofsPerCell; ++tDofIndex)
            TEST_FLOATING_EQUALITY(tHostLocalState(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_UpdatePlasticStep3D)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumStressTerms = 6;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    
    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0);
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal", tNumCells, tNumStressTerms);
    auto tHostYieldSurfaceNormal = Kokkos::create_mirror(tYieldSurfaceNormal);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumStressTerms; ++j)
        {
            tHostYieldSurfaceNormal(i, j) = (i + 1.0) * (j + 2.0)/ 5.0;
            //printf("(%d,%d) = %f\n", i,j,tHostYieldSurfaceNormal(i, j));
        }
    Kokkos::deep_copy(tYieldSurfaceNormal, tHostYieldSurfaceNormal);
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    Kokkos::deep_copy(tLocalState, tPrevLocalState);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Plato::Scalar tPenalizedHardeningModulusKinematic = 3.0;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.updatePlasticStrainAndBackstressPlasticStep(tCellOrdinal, tPrevLocalState, tYieldSurfaceNormal,
                                                                       tPenalizedHardeningModulusKinematic, tLocalState);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = 
      {{tHostPrevLocalState(0,0),tHostPrevLocalState(0,1),3.9798,5.4697,6.9596,10.899,12.8788,14.8586,10.9596,12.9394,14.9192,16.899,18.8788,20.8586},
       {tHostPrevLocalState(1,0),tHostPrevLocalState(1,1),9.9192,13.8788,17.8384,31.5959,37.5151,43.4343,25.8384,31.7576,37.6767,43.5959,49.5151,55.4343}};
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    Kokkos::deep_copy(tHostLocalState, tLocalState);

    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; ++tCellIndex)
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumLocalDofsPerCell; ++tDofIndex)
            TEST_FLOATING_EQUALITY(tHostLocalState(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_UpdateElasticStep2D)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 10;
    
    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0);
            //printf("PrevLocalState(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.updatePlasticStrainAndBackstressElasticStep(tCellOrdinal, tPrevLocalState, tLocalState);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-10;
    Kokkos::deep_copy(tHostPrevLocalState, tPrevLocalState);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    Kokkos::deep_copy(tHostLocalState, tLocalState);

    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; ++tCellIndex)
        for(Plato::OrdinalType tDofIndex = 2; tDofIndex < tNumLocalDofsPerCell; ++tDofIndex)
            TEST_FLOATING_EQUALITY(tHostLocalState(tCellIndex, tDofIndex), 
                                   tHostPrevLocalState(tCellIndex, tDofIndex), tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_UpdateElasticStep3D)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    
    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0);
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.updatePlasticStrainAndBackstressElasticStep(tCellOrdinal, tPrevLocalState, tLocalState);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-10;
    Kokkos::deep_copy(tHostPrevLocalState, tPrevLocalState);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    Kokkos::deep_copy(tHostLocalState, tLocalState);

    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; ++tCellIndex)
        for(Plato::OrdinalType tDofIndex = 2; tDofIndex < tNumLocalDofsPerCell; ++tDofIndex)
            TEST_FLOATING_EQUALITY(tHostLocalState(tCellIndex, tDofIndex), 
                                   tHostPrevLocalState(tCellIndex, tDofIndex), tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_computePlasticStrainMisfit2D)
{
    //1. SET DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumStressTerms = 4;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 10;

    Plato::ScalarMultiVector tLocalStateOne("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalStateOne = Kokkos::create_mirror(tLocalStateOne);
    for (unsigned int tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (unsigned int tDofIndex = 0; tDofIndex < tNumLocalDofsPerCell; tDofIndex++)
        {
            tHostLocalStateOne(tCellIndex, tDofIndex) = (tCellIndex + 1.0) * (tDofIndex + 1.0);
            //printf("LocalStateOne(%d,%d) = %f\n", tCellIndex,tDofIndex,tHostLocalStateOne(tCellIndex, tDofIndex));
        }
    }
    Kokkos::deep_copy(tLocalStateOne, tHostLocalStateOne);

    Plato::ScalarMultiVector tLocalStateTwo("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalStateTwo = Kokkos::create_mirror(tLocalStateTwo);
    for (unsigned int tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (unsigned int tDofIndex = 0; tDofIndex < tNumLocalDofsPerCell; tDofIndex++)
        {
            tHostLocalStateTwo(tCellIndex, tDofIndex) = (tCellIndex + 1.0) * (tDofIndex + 3.0);
            //printf("LocalStateTwo(%d,%d) = %f\n", tCellIndex,tDofIndex,tHostLocalStateTwo(tCellIndex, tDofIndex));
        }
    }
    Kokkos::deep_copy(tLocalStateTwo, tHostLocalStateTwo);

    // 2. CALL FUNCTION
    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;
    Plato::ScalarMultiVector tMisfit("misfit", tNumCells, tNumStressTerms);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.computePlasticStrainMisfit(tCellOrdinal, tLocalStateOne, tLocalStateTwo, tMisfit);
    }, "Unit Test");

    // 3. TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{-2,-2,-2,-2}, {-4,-4,-4,-4}};
    auto tHostMisfit = Kokkos::create_mirror(tMisfit);
    Kokkos::deep_copy(tHostMisfit, tMisfit);
    for (unsigned int tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (unsigned int tDofIndex = 0; tDofIndex < tNumStressTerms; tDofIndex++)
        {
            TEST_FLOATING_EQUALITY(tHostMisfit(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
            //printf("HostMisfit(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostMisfit(tCellIndex, tDofIndex+2));
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_computePlasticStrainMisfit3D)
{
    //1. SET DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumStressTerms = 6;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;

    Plato::ScalarMultiVector tLocalStateOne("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalStateOne = Kokkos::create_mirror(tLocalStateOne);
    for (unsigned int tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (unsigned int tDofIndex = 0; tDofIndex < tNumLocalDofsPerCell; tDofIndex++)
        {
            tHostLocalStateOne(tCellIndex, tDofIndex) = (tCellIndex + 1.0) * (tDofIndex + 1.0);
            //printf("LocalStateOne(%d,%d) = %f\n", tCellIndex,tDofIndex,tHostLocalStateOne(tCellIndex, tDofIndex));
        }
    }
    Kokkos::deep_copy(tLocalStateOne, tHostLocalStateOne);

    Plato::ScalarMultiVector tLocalStateTwo("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalStateTwo = Kokkos::create_mirror(tLocalStateTwo);
    for (unsigned int tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (unsigned int tDofIndex = 0; tDofIndex < tNumLocalDofsPerCell; tDofIndex++)
        {
            tHostLocalStateTwo(tCellIndex, tDofIndex) = (tCellIndex + 1.0) * (tDofIndex + 3.0);
            //printf("LocalStateTwo(%d,%d) = %f\n", tCellIndex,tDofIndex,tHostLocalStateTwo(tCellIndex, tDofIndex));
        }
    }
    Kokkos::deep_copy(tLocalStateTwo, tHostLocalStateTwo);

    // 2. CALL FUNCTION
    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;
    Plato::ScalarMultiVector tMisfit("misfit", tNumCells, tNumStressTerms);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.computePlasticStrainMisfit(tCellOrdinal, tLocalStateOne, tLocalStateTwo, tMisfit);
    }, "Unit Test");

    // 3. TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{-2,-2,-2,-2,-2,-2}, {-4,-4,-4,-4,-4,-4}};
    auto tHostMisfit = Kokkos::create_mirror(tMisfit);
    Kokkos::deep_copy(tHostMisfit, tMisfit);
    for (unsigned int tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (unsigned int tDofIndex = 0; tDofIndex < tNumStressTerms; tDofIndex++)
        {
            TEST_FLOATING_EQUALITY(tHostMisfit(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
            //printf("HostMisfit(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostMisfit(tCellIndex, tDofIndex+2));
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_computeCauchyStress2D)
{
    //1. SET DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumStressTerms = 4;

    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumStressTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    for (unsigned int tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (unsigned int tDofIndex = 0; tDofIndex < tNumStressTerms; tDofIndex++)
        {
            tHostElasticStrain(tCellIndex, tDofIndex) = (tCellIndex + 1.0) * (tDofIndex + 1.0);
            //printf("ElasticStrain(%d,%d) = %f\n", tCellIndex,tDofIndex, tHostElasticStrain(tCellIndex, tDofIndex));
        }
    }
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);

    // 2. CALL FUNCTION
    constexpr Plato::Scalar tBulkModulus = 25;
    constexpr Plato::Scalar tShearModulus = 3;
    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;
    Plato::ScalarMultiVector tCauchyStress("cauchy stress", tNumCells, tNumStressTerms);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.computeCauchyStress(tCellOrdinal, tBulkModulus, tShearModulus, tElasticStrain, tCauchyStress);
    }, "Unit Test");

    // 3. TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{153,159,9,171}, {306,318,18,342}};
    auto tHostCauchyStress = Kokkos::create_mirror(tCauchyStress);
    Kokkos::deep_copy(tHostCauchyStress, tCauchyStress);
    for (unsigned int tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (unsigned int tDofIndex = 0; tDofIndex < tNumStressTerms; tDofIndex++)
        {
            TEST_FLOATING_EQUALITY(tHostCauchyStress(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
            //printf("HostCauchyStress(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostCauchyStress(tCellIndex, tDofIndex));
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_YieldSurfaceNormal2D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumStressTerms = 4;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 10;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0);
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tDeviatoricStress("deviatoric stress", tNumCells, tNumStressTerms);
    auto tHostDeviatoricStress = Kokkos::create_mirror(tDeviatoricStress);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumStressTerms; ++j)
        {
            tHostDeviatoricStress(i, j) = (i + 1.0) * (j + 2.0)/ 5.0;
            //printf("(%d,%d) = %f\n", i,j,tHostDeviatoricStress(i, j));
        }
    Kokkos::deep_copy(tDeviatoricStress, tHostDeviatoricStress);
    
    Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal", tNumCells, tNumStressTerms);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Plato::ScalarVector tDevStressMinusBackstressNorm("deviatoric stress minus backstress", tNumCells);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.computeDeviatoricStressMinusBackstressNormalized(tCellOrdinal, tDeviatoricStress, tLocalState,
                                                                       tYieldSurfaceNormal, tDevStressMinusBackstressNorm);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {-0.372578,-0.417739,-0.4629,-0.508061};
    auto tHostDevStressMinusBackstressNorm = Kokkos::create_mirror(tDevStressMinusBackstressNorm);
    Kokkos::deep_copy(tHostDevStressMinusBackstressNorm, tDevStressMinusBackstressNorm);
    auto tHostYieldSurfaceNormal = Kokkos::create_mirror(tYieldSurfaceNormal);
    Kokkos::deep_copy(tHostYieldSurfaceNormal, tYieldSurfaceNormal);

    Plato::OrdinalType tCellIndex = 0;
    TEST_FLOATING_EQUALITY(tHostDevStressMinusBackstressNorm(tCellIndex), 17.7144, tTolerance);
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumStressTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostYieldSurfaceNormal(tCellIndex, tDofIndex), tGold[tDofIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_YieldSurfaceNormal3D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumStressTerms = 6;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0);
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tDeviatoricStress("deviatoric stress", tNumCells, tNumStressTerms);
    auto tHostDeviatoricStress = Kokkos::create_mirror(tDeviatoricStress);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumStressTerms; ++j)
        {
            tHostDeviatoricStress(i, j) = (i + 1.0) * (j + 2.0)/ 5.0;
            //printf("(%d,%d) = %f\n", i,j,tHostDeviatoricStress(i, j));
        }
    Kokkos::deep_copy(tDeviatoricStress, tHostDeviatoricStress);
    
    Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal", tNumCells, tNumStressTerms);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Plato::ScalarVector tDevStressMinusBackstressNorm("deviatoric stress minus backstress", tNumCells);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.computeDeviatoricStressMinusBackstressNormalized(tCellOrdinal, tDeviatoricStress, tLocalState,
                                                                       tYieldSurfaceNormal, tDevStressMinusBackstressNorm);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {-0.25878,-0.28286,-0.30693,-0.33100,-0.35508,-0.37915};
    auto tHostDevStressMinusBackstressNorm = Kokkos::create_mirror(tDevStressMinusBackstressNorm);
    Kokkos::deep_copy(tHostDevStressMinusBackstressNorm, tDevStressMinusBackstressNorm);
    auto tHostYieldSurfaceNormal = Kokkos::create_mirror(tYieldSurfaceNormal);
    Kokkos::deep_copy(tHostYieldSurfaceNormal, tYieldSurfaceNormal);

    Plato::OrdinalType tCellIndex = 0;
    TEST_FLOATING_EQUALITY(tHostDevStressMinusBackstressNorm(tCellIndex), 33.2319, tTolerance);
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumStressTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostYieldSurfaceNormal(tCellIndex, tDofIndex), tGold[tDofIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_DeviatoricStress2D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumStressTerms = 4;
    
    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumStressTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumStressTerms; ++j)
        {
            tHostElasticStrain(i, j) = (i + 1.0) * (j + 1.0);
            //printf("ElasticStrain(%d,%d) = %f\n", i,j,tHostElasticStrain(i, j));
        }
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);

    Plato::ScalarMultiVector tDeviatoricStress("deviatoric stress", tNumCells, tNumStressTerms);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Plato::Scalar tShearModulus = 3.5;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.computeDeviatoricStress(tCellOrdinal, tElasticStrain, tShearModulus,
                                                   tDeviatoricStress);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {-9.33333,-2.33333,10.5,11.6667};
    auto tHostDeviatoricStress = Kokkos::create_mirror(tDeviatoricStress);
    Kokkos::deep_copy(tHostDeviatoricStress, tDeviatoricStress);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumStressTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostDeviatoricStress(tCellIndex, tDofIndex), tGold[tDofIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_DeviatoricStress3D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumStressTerms = 6;
    
    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumStressTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumStressTerms; ++j)
        {
            tHostElasticStrain(i, j) = (i + 1.0) * (j + 1.0);
            //printf("(%d,%d) = %f\n", i,j,tHostElasticStrain(i, j));
        }
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);

    Plato::ScalarMultiVector tDeviatoricStress("deviatoric stress", tNumCells, tNumStressTerms);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Plato::Scalar tShearModulus = 3.5;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.computeDeviatoricStress(tCellOrdinal, tElasticStrain, tShearModulus,
                                                   tDeviatoricStress);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {-7.0,0.0,7.0,14.0,17.5,21.0};
    auto tHostDeviatoricStress = Kokkos::create_mirror(tDeviatoricStress);
    Kokkos::deep_copy(tHostDeviatoricStress, tDeviatoricStress);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumStressTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostDeviatoricStress(tCellIndex, tDofIndex), tGold[tDofIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_PlasticStrainResidualPlasticStep2D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumStressTerms = 4;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 10;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0) / 2.0;
            //printf("LocalState(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0) / 3.0;
            //printf("PrevLocalState(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal", tNumCells, tNumStressTerms);
    auto tHostYieldSurfaceNormal = Kokkos::create_mirror(tYieldSurfaceNormal);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumStressTerms; ++j)
        {
            tHostYieldSurfaceNormal(i, j) = (i + 1.0) * (j + 2.0) / 5.0;
            //printf("YieldSurfaceNormal(%d,%d) = %f\n", i,j,tHostYieldSurfaceNormal(i, j));
        }
    Kokkos::deep_copy(tYieldSurfaceNormal, tHostYieldSurfaceNormal);

    Plato::ScalarMultiVector tResult("plastic strain residual", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.fillPlasticStrainTensorResidualPlasticStep(tCellOrdinal, tLocalState, 
                                                   tPrevLocalState, tYieldSurfaceNormal, tResult);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold = {0.0101021,-0.0681803,-0.146463,-0.224745};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 2; tDofIndex < 2 + tNumStressTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostResult(tCellIndex, tDofIndex), tGold[tDofIndex-2], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_PlasticStrainResidualPlasticStep3D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumStressTerms = 6;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0) / 2.0;
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0) / 3.0;
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal", tNumCells, tNumStressTerms);
    auto tHostYieldSurfaceNormal = Kokkos::create_mirror(tYieldSurfaceNormal);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumStressTerms; ++j)
        {
            tHostYieldSurfaceNormal(i, j) = (i + 1.0) * (j + 2.0) / 5.0;
            //printf("(%d,%d) = %f\n", i,j,tHostYieldSurfaceNormal(i, j));
        }
    Kokkos::deep_copy(tYieldSurfaceNormal, tHostYieldSurfaceNormal);

    Plato::ScalarMultiVector tResult("plastic strain residual", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.fillPlasticStrainTensorResidualPlasticStep(tCellOrdinal, tLocalState, 
                                                   tPrevLocalState, tYieldSurfaceNormal, tResult);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold = {0.0101021,-0.0681803,-0.146463,-0.224745,-0.303027,-0.381309};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 2; tDofIndex < 2 + tNumStressTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostResult(tCellIndex, tDofIndex), tGold[tDofIndex-2], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_BackstressResidualPlasticStep2D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumStressTerms = 4;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 10;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0) / 2.0;
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0) / 3.0;
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal", tNumCells, tNumStressTerms);
    auto tHostYieldSurfaceNormal = Kokkos::create_mirror(tYieldSurfaceNormal);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumStressTerms; ++j)
        {
            tHostYieldSurfaceNormal(i, j) = (i + 1.0) * (j + 2.0) / 5.0;
            //printf("(%d,%d) = %f\n", i,j,tHostYieldSurfaceNormal(i, j));
        }
    Kokkos::deep_copy(tYieldSurfaceNormal, tHostYieldSurfaceNormal);

    Plato::ScalarMultiVector tResult("plastic strain residual", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Plato::Scalar tHardeningModulusKinematic = 3.2;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.fillBackstressTensorResidualPlasticStep(tCellOrdinal, tHardeningModulusKinematic,
                                                   tLocalState, tPrevLocalState, tYieldSurfaceNormal, tResult);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold = {0.121551,-0.23434,-0.590231,-0.946122};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 6; tDofIndex < 6 + tNumStressTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostResult(tCellIndex, tDofIndex), tGold[tDofIndex-6], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_BackstressResidualPlasticStep3D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumStressTerms = 6;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0) / 1.5;
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0) / 3.0;
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal", tNumCells, tNumStressTerms);
    auto tHostYieldSurfaceNormal = Kokkos::create_mirror(tYieldSurfaceNormal);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumStressTerms; ++j)
        {
            tHostYieldSurfaceNormal(i, j) = (i + 1.0) * (j + 2.0) / 7.0;
            //printf("(%d,%d) = %f\n", i,j,tHostYieldSurfaceNormal(i, j));
        }
    Kokkos::deep_copy(tYieldSurfaceNormal, tHostYieldSurfaceNormal);

    Plato::ScalarMultiVector tResult("plastic strain residual", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Plato::Scalar tHardeningModulusKinematic = 3.2;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.fillBackstressTensorResidualPlasticStep(tCellOrdinal, tHardeningModulusKinematic,
                                                   tLocalState, tPrevLocalState, tYieldSurfaceNormal, tResult);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold = {2.00465,1.84031,1.67597,1.51163,1.34729,1.18295};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 8; tDofIndex < 8 + tNumStressTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostResult(tCellIndex, tDofIndex), tGold[tDofIndex-8], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_PlasticStrainResidualElasticStep2D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumStressTerms = 4;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 10;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0) / 2.0;
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0) / 3.0;
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tResult("plastic strain residual", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.fillPlasticStrainTensorResidualElasticStep(tCellOrdinal, tLocalState, 
                                                                      tPrevLocalState, tResult);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold = {0.5,2.0/3.0,0.833333,1};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 2; tDofIndex < 2 + tNumStressTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostResult(tCellIndex, tDofIndex), tGold[tDofIndex-2], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_PlasticStrainResidualElasticStep3D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumStressTerms = 6;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0) / 2.0;
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0) / 3.0;
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tResult("plastic strain residual", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.fillPlasticStrainTensorResidualElasticStep(tCellOrdinal, tLocalState, 
                                                                      tPrevLocalState, tResult);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold = {0.5,0.666666,0.83333,1.0,1.16666,1.33333};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 2; tDofIndex < 2 + tNumStressTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostResult(tCellIndex, tDofIndex), tGold[tDofIndex-2], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_BackstressResidualElasticStep2D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumStressTerms = 4;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 10;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0) / 2.0;
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0) / 3.0;
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tResult("backstress residual", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.fillBackstressTensorResidualElasticStep(tCellOrdinal, tLocalState, 
                                                                   tPrevLocalState, tResult);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold = {1.16667,1.33333,1.5,1.66667};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 6; tDofIndex < 6 + tNumStressTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostResult(tCellIndex, tDofIndex), tGold[tDofIndex-6], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_BackstressResidualElasticStep3D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumStressTerms = 6;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0) / 2.0;
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0) / 3.0;
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tResult("backstress residual", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.fillBackstressTensorResidualElasticStep(tCellOrdinal, tLocalState, 
                                                                   tPrevLocalState, tResult);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold = {1.5, 1.666666, 1.8333333, 2.0, 2.1666666, 2.333333};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 8; tDofIndex < 8 + tNumStressTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostResult(tCellIndex, tDofIndex), tGold[tDofIndex-8], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ThermoPlasticityUtils_ElasticStrainWithThermo2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using PhysicsT = Plato::SimplexThermoPlasticity<tSpaceDim>;

    using Residual = typename Plato::Evaluation<PhysicsT>::Residual;

    using GlobalStateT = typename Residual::StateScalarType;
    using LocalStateT  = typename Residual::LocalStateScalarType;
    using ConfigT      = typename Residual::ConfigScalarType;
    using ResultT      = typename Residual::ResultScalarType;
    using ControlT     = typename Residual::ControlScalarType;

    using ElasticStrainT = typename Plato::fad_type_t<PhysicsT, LocalStateT, 
                                                      GlobalStateT, ConfigT, ControlT>;

    const     Plato::OrdinalType tNumCells            = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerCell         = PhysicsT::mNumDofsPerCell;
    constexpr Plato::OrdinalType tDofsPerNode         = PhysicsT::mNumDofsPerNode;
    constexpr Plato::OrdinalType tNodesPerCell        = PhysicsT::mNumNodesPerCell;
    constexpr Plato::OrdinalType tNumStressTerms      = PhysicsT::mNumStressTerms;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = PhysicsT::mNumLocalDofsPerCell;

    // Create configuration workset
    Plato::WorksetBase<PhysicsT> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(0.9, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create global state workset
    const Plato::OrdinalType tNumNodalDofs = tDofsPerCell * tNumCells;
    Plato::ScalarVector tGlobalState("global state", tNumNodalDofs);
    Plato::fill(0.0, tGlobalState);

    Plato::OrdinalType tDispX = 0;
    Plato::OrdinalType tDispY = 1;
    set_dof_in_scalar_vector_on_boundary_2D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispX, 0.1);
    set_dof_in_scalar_vector_on_boundary_2D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispY, 0.1);
    set_dof_in_scalar_vector_on_boundary_2D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispX, 0.1);
    set_dof_in_scalar_vector_on_boundary_2D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispY, 0.1);

    Plato::OrdinalType tTemperature = 3;
    set_dof_in_scalar_vector(tGlobalState, tDofsPerNode, tTemperature, 310.0);

    Plato::ScalarMultiVectorT<GlobalStateT> tGlobalStateWS("global state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tGlobalState, tGlobalStateWS);

    // Create local state workset
    const Plato::OrdinalType tNumLocalDofs = tNumLocalDofsPerCell * tNumCells;
    Plato::ScalarVector tLocalState("local state", tNumLocalDofs);
    Plato::fill(0.0, tLocalState);
    Plato::OrdinalType tPlasticStrainXX = 2;
    Plato::OrdinalType tPlasticStrainYY = 3;
    Plato::OrdinalType tPlasticStrainXY = 4;
    Plato::OrdinalType tPlasticStrainZZ = 5;
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainXX, 1.0);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainYY, 2.0);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainXY, 3.2);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainZZ, 3.9);
    Plato::ScalarMultiVectorT<LocalStateT> tLocalStateWS("local state workset", tNumCells, tNumLocalDofsPerCell);
    tWorksetBase.worksetLocalState(tLocalState, tLocalStateWS);

    // Create result/output workset
    Plato::ScalarMultiVectorT< ElasticStrainT > 
          tElasticStrain("elastic strain output", tNumCells, tNumStressTerms);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    
    constexpr Plato::Scalar tThermalExpansionCoefficient = 1.0e-2;
    constexpr Plato::Scalar tReferenceTemperature        = 300.0;

    Plato::ThermoPlasticityUtilities<tSpaceDim, PhysicsT> 
          tThermoPlasticityUtils(tThermalExpansionCoefficient, tReferenceTemperature);

    Plato::LinearTetCubRuleDegreeOne<tSpaceDim> tCubatureRule;
    auto tBasisFunctions = tCubatureRule.getBasisFunctions();

    Plato::ComputeGradientWorkset<tSpaceDim> tComputeGradient;
    Plato::ScalarVectorT<ConfigT>  tCellVolume("cell volume unused", tNumCells);
    Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, tNodesPerCell, tSpaceDim);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tComputeGradient(tCellOrdinal, tGradient, tConfigWS, tCellVolume);

        tThermoPlasticityUtils.computeElasticStrain(tCellOrdinal, tGlobalStateWS, tLocalStateWS, 
                                                    tBasisFunctions,   tGradient, tElasticStrain);
    }, "Unit Test");


    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{-1.0,-2.0,-3.0,-4.0},
                                                     {-1.0,-2.0,-3.0,-4.0}};
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    Kokkos::deep_copy(tHostElasticStrain, tElasticStrain);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
        for(Plato::OrdinalType tStressIndex = 0; tStressIndex < tNumStressTerms; tStressIndex++)
            TEST_FLOATING_EQUALITY(tHostElasticStrain(tCellIndex, tStressIndex),
                                                tGold[tCellIndex][tStressIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ThermoPlasticityUtils_ElasticStrainWithoutThermo2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using PhysicsT = Plato::SimplexPlasticity<tSpaceDim>;

    using Residual = typename Plato::Evaluation<PhysicsT>::Residual;

    using GlobalStateT = typename Residual::StateScalarType;
    using LocalStateT  = typename Residual::LocalStateScalarType;
    using ConfigT      = typename Residual::ConfigScalarType;
    using ResultT      = typename Residual::ResultScalarType;
    using ControlT     = typename Residual::ControlScalarType;

    using ElasticStrainT = typename Plato::fad_type_t<PhysicsT, LocalStateT, 
                                                      GlobalStateT, ConfigT, ControlT>;

    const     Plato::OrdinalType tNumCells            = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerCell         = PhysicsT::mNumDofsPerCell;
    constexpr Plato::OrdinalType tDofsPerNode         = PhysicsT::mNumDofsPerNode;
    constexpr Plato::OrdinalType tNodesPerCell        = PhysicsT::mNumNodesPerCell;
    constexpr Plato::OrdinalType tNumStressTerms      = PhysicsT::mNumStressTerms;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = PhysicsT::mNumLocalDofsPerCell;

    // Create configuration workset
    Plato::WorksetBase<PhysicsT> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(0.9, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create global state workset
    const Plato::OrdinalType tNumNodalDofs = tDofsPerCell * tNumCells;
    Plato::ScalarVector tGlobalState("global state", tNumNodalDofs);
    Plato::fill(0.0, tGlobalState);

    Plato::OrdinalType tDispX = 0;
    Plato::OrdinalType tDispY = 1;
    set_dof_in_scalar_vector_on_boundary_2D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispX, 0.1);
    set_dof_in_scalar_vector_on_boundary_2D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispY, 0.1);
    set_dof_in_scalar_vector_on_boundary_2D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispX, 0.1);
    set_dof_in_scalar_vector_on_boundary_2D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispY, 0.1);
    Plato::ScalarMultiVectorT<GlobalStateT> tGlobalStateWS("global state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tGlobalState, tGlobalStateWS);

    // Create local state workset
    const Plato::OrdinalType tNumLocalDofs = tNumLocalDofsPerCell * tNumCells;
    Plato::ScalarVector tLocalState("local state", tNumLocalDofs);
    Plato::fill(0.0, tLocalState);
    Plato::OrdinalType tPlasticStrainXX = 2;
    Plato::OrdinalType tPlasticStrainYY = 3;
    Plato::OrdinalType tPlasticStrainXY = 4;
    Plato::OrdinalType tPlasticStrainZZ = 5;
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainXX, 1.1);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainYY, 2.1);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainXY, 3.2);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainZZ, 4.0);
    Plato::ScalarMultiVectorT<LocalStateT> tLocalStateWS("local state workset", tNumCells, tNumLocalDofsPerCell);
    tWorksetBase.worksetLocalState(tLocalState, tLocalStateWS);

    // Create result/output workset
    Plato::ScalarMultiVectorT< ElasticStrainT > 
          tElasticStrain("elastic strain output", tNumCells, tNumStressTerms);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    
    constexpr Plato::Scalar tThermalExpansionCoefficient = 1.0;
    constexpr Plato::Scalar tReferenceTemperature        = 0.0;

    Plato::ThermoPlasticityUtilities<tSpaceDim, PhysicsT> 
          tThermoPlasticityUtils(tThermalExpansionCoefficient, tReferenceTemperature);

    Plato::LinearTetCubRuleDegreeOne<tSpaceDim> tCubatureRule;
    auto tBasisFunctions = tCubatureRule.getBasisFunctions();

    Plato::ComputeGradientWorkset<tSpaceDim> tComputeGradient;
    Plato::ScalarVectorT<ConfigT>  tCellVolume("cell volume unused", tNumCells);
    Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, tNodesPerCell, tSpaceDim);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tComputeGradient(tCellOrdinal, tGradient, tConfigWS, tCellVolume);

        tThermoPlasticityUtils.computeElasticStrain(tCellOrdinal, tGlobalStateWS, tLocalStateWS, 
                                                    tBasisFunctions,   tGradient, tElasticStrain);
    }, "Unit Test");


    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{-1.0,-2.0,-3.0,-4.0},
                                                     {-1.0,-2.0,-3.0,-4.0}};
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    Kokkos::deep_copy(tHostElasticStrain, tElasticStrain);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
        for(Plato::OrdinalType tStressIndex = 0; tStressIndex < tNumStressTerms; tStressIndex++)
            TEST_FLOATING_EQUALITY(tHostElasticStrain(tCellIndex, tStressIndex),
                                                tGold[tCellIndex][tStressIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ThermoPlasticityUtils_ElasticStrainWithThermo3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using PhysicsT = Plato::SimplexThermoPlasticity<tSpaceDim>;

    using Residual = typename Plato::Evaluation<PhysicsT>::Residual;

    using GlobalStateT = typename Residual::StateScalarType;
    using LocalStateT  = typename Residual::LocalStateScalarType;
    using ConfigT      = typename Residual::ConfigScalarType;
    using ResultT      = typename Residual::ResultScalarType;
    using ControlT     = typename Residual::ControlScalarType;

    using ElasticStrainT = typename Plato::fad_type_t<PhysicsT, LocalStateT, 
                                                      GlobalStateT, ConfigT, ControlT>;

    const     Plato::OrdinalType tNumCells            = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerCell         = PhysicsT::mNumDofsPerCell;
    constexpr Plato::OrdinalType tDofsPerNode         = PhysicsT::mNumDofsPerNode;
    constexpr Plato::OrdinalType tNodesPerCell        = PhysicsT::mNumNodesPerCell;
    constexpr Plato::OrdinalType tNumStressTerms       = PhysicsT::mNumVoigtTerms;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = PhysicsT::mNumLocalDofsPerCell;

    // Create configuration workset
    Plato::WorksetBase<PhysicsT> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(0.9, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create global state workset
    const Plato::OrdinalType tNumNodalDofs = tDofsPerCell * tNumCells;
    Plato::ScalarVector tGlobalState("global state", tNumNodalDofs);
    Plato::fill(0.0, tGlobalState);

    Plato::OrdinalType tDispX = 0;
    Plato::OrdinalType tDispY = 1;
    Plato::OrdinalType tDispZ = 2;
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispX, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispY, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispZ, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispX, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispY, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispZ, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "z1", tGlobalState, tDofsPerNode, tDispX, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "z1", tGlobalState, tDofsPerNode, tDispY, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "z1", tGlobalState, tDofsPerNode, tDispZ, 0.1);

    Plato::OrdinalType tTemperature = 4;
    set_dof_in_scalar_vector(tGlobalState, tDofsPerNode, tTemperature, 310.0);

    Plato::ScalarMultiVectorT<GlobalStateT> tGlobalStateWS("global state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tGlobalState, tGlobalStateWS);

    // Create local state workset
    const Plato::OrdinalType tNumLocalDofs = tNumLocalDofsPerCell * tNumCells;
    Plato::ScalarVector tLocalState("local state", tNumLocalDofs);
    Plato::fill(0.0, tLocalState);
    Plato::OrdinalType tPlasticStrainXX = 2;
    Plato::OrdinalType tPlasticStrainYY = 3;
    Plato::OrdinalType tPlasticStrainZZ = 4;
    Plato::OrdinalType tPlasticStrainYZ = 5;
    Plato::OrdinalType tPlasticStrainXZ = 6;
    Plato::OrdinalType tPlasticStrainXY = 7;
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainXX, 1.0);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainYY, 2.0);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainZZ, 3.0);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainXY, 3.2);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainYZ, 3.2);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainXZ, 3.2);
    Plato::ScalarMultiVectorT<LocalStateT> tLocalStateWS("local state workset", tNumCells, tNumLocalDofsPerCell);
    tWorksetBase.worksetLocalState(tLocalState, tLocalStateWS);

    // Create result/output workset
    Plato::ScalarMultiVectorT< ElasticStrainT > 
          tElasticStrain("elastic strain output", tNumCells, tNumStressTerms);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    
    constexpr Plato::Scalar tThermalExpansionCoefficient = 1.0e-2;
    constexpr Plato::Scalar tReferenceTemperature        = 300.0;

    Plato::ThermoPlasticityUtilities<tSpaceDim, PhysicsT> 
          tThermoPlasticityUtils(tThermalExpansionCoefficient, tReferenceTemperature);

    Plato::LinearTetCubRuleDegreeOne<tSpaceDim> tCubatureRule;
    auto tBasisFunctions = tCubatureRule.getBasisFunctions();

    Plato::ComputeGradientWorkset<tSpaceDim> tComputeGradient;
    Plato::ScalarVectorT<ConfigT>  tCellVolume("cell volume unused", tNumCells);
    Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, tNodesPerCell, tSpaceDim);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tComputeGradient(tCellOrdinal, tGradient, tConfigWS, tCellVolume);

        tThermoPlasticityUtils.computeElasticStrain(tCellOrdinal, tGlobalStateWS, tLocalStateWS, 
                                                    tBasisFunctions,   tGradient, tElasticStrain);
    }, "Unit Test");


    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{-1.0,-2.0,-3.0,-3.0,-3.0,-3.0},
                                                     {-1.0,-2.0,-3.0,-3.0,-3.0,-3.0},
                                                     {-1.0,-2.0,-3.0,-3.0,-3.0,-3.0},
                                                     {-1.0,-2.0,-3.0,-3.0,-3.0,-3.0},
                                                     {-1.0,-2.0,-3.0,-3.0,-3.0,-3.0},
                                                     {-1.0,-2.0,-3.0,-3.0,-3.0,-3.0}};
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    Kokkos::deep_copy(tHostElasticStrain, tElasticStrain);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
        for(Plato::OrdinalType tStressIndex = 0; tStressIndex < tNumStressTerms; tStressIndex++)
            TEST_FLOATING_EQUALITY(tHostElasticStrain(tCellIndex, tStressIndex),
                                                tGold[tCellIndex][tStressIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ThermoPlasticityUtils_ElasticStrainWithoutThermo3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using PhysicsT = Plato::SimplexPlasticity<tSpaceDim>;

    using Residual = typename Plato::Evaluation<PhysicsT>::Residual;

    using GlobalStateT = typename Residual::StateScalarType;
    using LocalStateT  = typename Residual::LocalStateScalarType;
    using ConfigT      = typename Residual::ConfigScalarType;
    using ResultT      = typename Residual::ResultScalarType;
    using ControlT     = typename Residual::ControlScalarType;

    using ElasticStrainT = typename Plato::fad_type_t<PhysicsT, LocalStateT, 
                                                      GlobalStateT, ConfigT, ControlT>;

    const     Plato::OrdinalType tNumCells            = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerCell         = PhysicsT::mNumDofsPerCell;
    constexpr Plato::OrdinalType tDofsPerNode         = PhysicsT::mNumDofsPerNode;
    constexpr Plato::OrdinalType tNodesPerCell        = PhysicsT::mNumNodesPerCell;
    constexpr Plato::OrdinalType tNumStressTerms       = PhysicsT::mNumVoigtTerms;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = PhysicsT::mNumLocalDofsPerCell;

    // Create configuration workset
    Plato::WorksetBase<PhysicsT> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(0.9, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create global state workset
    const Plato::OrdinalType tNumNodalDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tGlobalState("global state", tNumNodalDofs);
    Plato::fill(0.0, tGlobalState);

    Plato::OrdinalType tDispX = 0;
    Plato::OrdinalType tDispY = 1;
    Plato::OrdinalType tDispZ = 2;
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispX, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispY, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispZ, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispX, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispY, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispZ, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "z1", tGlobalState, tDofsPerNode, tDispX, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "z1", tGlobalState, tDofsPerNode, tDispY, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "z1", tGlobalState, tDofsPerNode, tDispZ, 0.1);
    Plato::ScalarMultiVectorT<GlobalStateT> tGlobalStateWS("global state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tGlobalState, tGlobalStateWS);

    // Create local state workset
    const Plato::OrdinalType tNumLocalDofs = tNumLocalDofsPerCell * tNumCells;
    Plato::ScalarVector tLocalState("local state", tNumLocalDofs);
    Plato::fill(0.0, tLocalState);
    Plato::OrdinalType tPlasticStrainXX = 2;
    Plato::OrdinalType tPlasticStrainYY = 3;
    Plato::OrdinalType tPlasticStrainZZ = 4;
    Plato::OrdinalType tPlasticStrainYZ = 5;
    Plato::OrdinalType tPlasticStrainXZ = 6;
    Plato::OrdinalType tPlasticStrainXY = 7;
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainXX, 1.1);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainYY, 2.1);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainZZ, 3.1);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainXY, 3.2);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainYZ, 3.2);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainXZ, 3.2);
    Plato::ScalarMultiVectorT<LocalStateT> tLocalStateWS("local state workset", tNumCells, tNumLocalDofsPerCell);
    tWorksetBase.worksetLocalState(tLocalState, tLocalStateWS);

    // Create result/output workset
    Plato::ScalarMultiVectorT< ElasticStrainT > 
          tElasticStrain("elastic strain output", tNumCells, tNumStressTerms);

    // // ALLOCATE PLATO CRITERION
    // Plato::DataMap tDataMap;
    // Omega_h::MeshSets tMeshSets;
    
    constexpr Plato::Scalar tThermalExpansionCoefficient = 1.0;
    constexpr Plato::Scalar tReferenceTemperature        = 0.0;

    Plato::ThermoPlasticityUtilities<tSpaceDim, PhysicsT> 
          tThermoPlasticityUtils(tThermalExpansionCoefficient, tReferenceTemperature);

    Plato::LinearTetCubRuleDegreeOne<tSpaceDim> tCubatureRule;
    auto tBasisFunctions = tCubatureRule.getBasisFunctions();

    Plato::ComputeGradientWorkset<tSpaceDim> tComputeGradient;
    Plato::ScalarVectorT<ConfigT>  tCellVolume("cell volume unused", tNumCells);
    Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, tNodesPerCell, tSpaceDim);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tComputeGradient(tCellOrdinal, tGradient, tConfigWS, tCellVolume);

        tThermoPlasticityUtils.computeElasticStrain(tCellOrdinal, tGlobalStateWS, tLocalStateWS, 
                                                    tBasisFunctions,   tGradient, tElasticStrain);
    }, "Unit Test");


    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{-1.0,-2.0,-3.0,-3.0,-3.0,-3.0},
                                                     {-1.0,-2.0,-3.0,-3.0,-3.0,-3.0},
                                                     {-1.0,-2.0,-3.0,-3.0,-3.0,-3.0},
                                                     {-1.0,-2.0,-3.0,-3.0,-3.0,-3.0},
                                                     {-1.0,-2.0,-3.0,-3.0,-3.0,-3.0},
                                                     {-1.0,-2.0,-3.0,-3.0,-3.0,-3.0}};
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    Kokkos::deep_copy(tHostElasticStrain, tElasticStrain);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
        for(Plato::OrdinalType tStressIndex = 0; tStressIndex < tNumStressTerms; tStressIndex++)
            TEST_FLOATING_EQUALITY(tHostElasticStrain(tCellIndex, tStressIndex),
                                                tGold[tCellIndex][tStressIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2Plasticity_GradGlobalState3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::MeshSets tMeshSets;

    // ### NOTICE THAT THIS IS ONLY PLASTICITY (NO TEMPERATURE) ###
    using PhysicsT = Plato::Plasticity<tSpaceDim>;

    using EvalType = typename Plato::Evaluation<PhysicsT>::Jacobian;

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                   \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e2'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='910.0'/> \n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "    <ParameterList name='J2 Plasticity'>                                                \n"
    "      <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>             \n"
    "      <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
    "      <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/> \n"
    "      <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
    "      <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

    std::string tProblemType = "J2Plasticity";
    Plato::LocalVectorFunctionInc<PhysicsT> tLocalVectorFuncInc(*tMesh, tMeshSets, tDataMap,
                                                                *tParamList, tProblemType);

    Plato::test_partial_global_state<EvalType, PhysicsT>(*tMesh, tLocalVectorFuncInc);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2Plasticity_GradGlobalState2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::MeshSets tMeshSets;

    // ### NOTICE THAT THIS IS ONLY PLASTICITY (NO TEMPERATURE) ###
    using PhysicsT = Plato::Plasticity<tSpaceDim>;

    using EvalType = typename Plato::Evaluation<PhysicsT>::Jacobian;

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                   \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e2'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='910.0'/> \n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "    <ParameterList name='J2 Plasticity'>                                                \n"
    "      <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>             \n"
    "      <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
    "      <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/> \n"
    "      <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
    "      <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

    std::string tProblemType = "J2Plasticity";
    Plato::LocalVectorFunctionInc<PhysicsT> tLocalVectorFuncInc(*tMesh, tMeshSets, tDataMap,
                                                                *tParamList, tProblemType);

    Plato::test_partial_global_state<EvalType, PhysicsT>(*tMesh, tLocalVectorFuncInc);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2Plasticity_GradPrevGlobalState3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::MeshSets tMeshSets;

    // ### NOTICE THAT THIS IS ONLY PLASTICITY (NO TEMPERATURE) ###
    using PhysicsT = Plato::Plasticity<tSpaceDim>;

    using EvalType = typename Plato::Evaluation<PhysicsT>::JacobianP;

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                   \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e2'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='910.0'/> \n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "    <ParameterList name='J2 Plasticity'>                                                \n"
    "      <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>             \n"
    "      <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
    "      <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/> \n"
    "      <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
    "      <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

    std::string tProblemType = "J2Plasticity";
    Plato::LocalVectorFunctionInc<PhysicsT> tLocalVectorFuncInc(*tMesh, tMeshSets, tDataMap,
                                                                *tParamList, tProblemType);

    Plato::test_partial_prev_global_state<EvalType, PhysicsT>(*tMesh, tLocalVectorFuncInc);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2Plasticity_GradPrevGlobalState2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::MeshSets tMeshSets;

    // ### NOTICE THAT THIS IS ONLY PLASTICITY (NO TEMPERATURE) ###
    using PhysicsT = Plato::Plasticity<tSpaceDim>;

    using EvalType = typename Plato::Evaluation<PhysicsT>::JacobianP;

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                   \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e2'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='910.0'/> \n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "    <ParameterList name='J2 Plasticity'>                                                \n"
    "      <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>             \n"
    "      <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
    "      <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/> \n"
    "      <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
    "      <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

    std::string tProblemType = "J2Plasticity";
    Plato::LocalVectorFunctionInc<PhysicsT> tLocalVectorFuncInc(*tMesh, tMeshSets, tDataMap,
                                                                *tParamList, tProblemType);

    Plato::test_partial_prev_global_state<EvalType, PhysicsT>(*tMesh, tLocalVectorFuncInc);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2Plasticity_GradLocalState3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::MeshSets tMeshSets;

    // ### NOTICE THAT THIS IS ONLY PLASTICITY (NO TEMPERATURE) ###
    using PhysicsT = Plato::Plasticity<tSpaceDim>;

    using EvalType = typename Plato::Evaluation<PhysicsT>::LocalJacobian;

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                   \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e2'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='910.0'/> \n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "    <ParameterList name='J2 Plasticity'>                                                \n"
    "      <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>             \n"
    "      <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
    "      <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/> \n"
    "      <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
    "      <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

    std::string tProblemType = "J2Plasticity";
    Plato::LocalVectorFunctionInc<PhysicsT> tLocalVectorFuncInc(*tMesh, tMeshSets, tDataMap,
                                                                *tParamList, tProblemType);

    Plato::test_partial_local_state<EvalType, PhysicsT>(*tMesh, tLocalVectorFuncInc);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2Plasticity_GradLocalState2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::MeshSets tMeshSets;

    // ### NOTICE THAT THIS IS ONLY PLASTICITY (NO TEMPERATURE) ###
    using PhysicsT = Plato::Plasticity<tSpaceDim>;

    using EvalType = typename Plato::Evaluation<PhysicsT>::LocalJacobian;

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                   \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e2'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='910.0'/> \n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "    <ParameterList name='J2 Plasticity'>                                                \n"
    "      <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>             \n"
    "      <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
    "      <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/> \n"
    "      <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
    "      <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

    std::string tProblemType = "J2Plasticity";
    Plato::LocalVectorFunctionInc<PhysicsT> tLocalVectorFuncInc(*tMesh, tMeshSets, tDataMap,
                                                                *tParamList, tProblemType);

    Plato::test_partial_local_state<EvalType, PhysicsT>(*tMesh, tLocalVectorFuncInc);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2Plasticity_GradPrevLocalState3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::MeshSets tMeshSets;

    // ### NOTICE THAT THIS IS ONLY PLASTICITY (NO TEMPERATURE) ###
    using PhysicsT = Plato::Plasticity<tSpaceDim>;

    using EvalType = typename Plato::Evaluation<PhysicsT>::LocalJacobianP;

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                   \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e2'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='910.0'/> \n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "    <ParameterList name='J2 Plasticity'>                                                \n"
    "      <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>             \n"
    "      <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
    "      <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/> \n"
    "      <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
    "      <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

    std::string tProblemType = "J2Plasticity";
    Plato::LocalVectorFunctionInc<PhysicsT> tLocalVectorFuncInc(*tMesh, tMeshSets, tDataMap,
                                                                *tParamList, tProblemType);

    Plato::test_partial_prev_local_state<EvalType, PhysicsT>(*tMesh, tLocalVectorFuncInc);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2Plasticity_GradPrevLocalState2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::MeshSets tMeshSets;

    // ### NOTICE THAT THIS IS ONLY PLASTICITY (NO TEMPERATURE) ###
    using PhysicsT = Plato::Plasticity<tSpaceDim>;

    using EvalType = typename Plato::Evaluation<PhysicsT>::LocalJacobianP;

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                   \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e2'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='910.0'/> \n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "    <ParameterList name='J2 Plasticity'>                                                \n"
    "      <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>             \n"
    "      <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
    "      <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/> \n"
    "      <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
    "      <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

    std::string tProblemType = "J2Plasticity";
    Plato::LocalVectorFunctionInc<PhysicsT> tLocalVectorFuncInc(*tMesh, tMeshSets, tDataMap,
                                                                *tParamList, tProblemType);

    Plato::test_partial_prev_local_state<EvalType, PhysicsT>(*tMesh, tLocalVectorFuncInc);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2Plasticity_GradControl3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::MeshSets tMeshSets;

    // ### NOTICE THAT THIS IS ONLY PLASTICITY (NO TEMPERATURE) ###
    using PhysicsT = Plato::Plasticity<tSpaceDim>;

    using EvalType = typename Plato::Evaluation<PhysicsT>::GradientZ;

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                   \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e2'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='910.0'/> \n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "    <ParameterList name='J2 Plasticity'>                                                \n"
    "      <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>             \n"
    "      <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='2'/>  \n"
    "      <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/> \n"
    "      <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='1.5'/>\n"
    "      <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

    std::string tProblemType = "J2Plasticity";
    Plato::LocalVectorFunctionInc<PhysicsT> tLocalVectorFuncInc(*tMesh, tMeshSets, tDataMap,
                                                                *tParamList, tProblemType);

    Plato::test_partial_local_vect_func_inc_wrt_control<EvalType, PhysicsT>(*tMesh, tLocalVectorFuncInc);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2Plasticity_GradControl2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::MeshSets tMeshSets;

    // ### NOTICE THAT THIS IS ONLY PLASTICITY (NO TEMPERATURE) ###
    using PhysicsT = Plato::Plasticity<tSpaceDim>;

    using EvalType = typename Plato::Evaluation<PhysicsT>::GradientZ;

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                   \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e2'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='910.0'/> \n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "    <ParameterList name='J2 Plasticity'>                                                \n"
    "      <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>      \n"
    "      <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>             \n"
    "      <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='2'/>  \n"
    "      <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/> \n"
    "      <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='1.5'/>\n"
    "      <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

    std::string tProblemType = "J2Plasticity";
    Plato::LocalVectorFunctionInc<PhysicsT> tLocalVectorFuncInc(*tMesh, tMeshSets, tDataMap,
                                                                *tParamList, tProblemType);

    Plato::test_partial_local_vect_func_inc_wrt_control<EvalType, PhysicsT>(*tMesh, tLocalVectorFuncInc);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2Plasticity_Evaluate3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::MeshSets tMeshSets;

    // ### NOTICE THAT THIS IS ONLY PLASTICITY (NO TEMPERATURE) ###
    using PhysicsT = Plato::Plasticity<tSpaceDim>;

    using EvalType = typename Plato::Evaluation<PhysicsT>::Residual;

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='520.0'/>                   \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e2'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='910.0'/> \n"
    "      <Parameter  name='Reference Temperature' type='double' value='100.0'/>            \n"
    "    </ParameterList>                                                                    \n"
    "    <ParameterList name='J2 Plasticity'>                                                \n"
    "      <Parameter  name='Hardening Modulus Isotropic' type='double' value='20.0'/>       \n"
    "      <Parameter  name='Hardening Modulus Kinematic' type='double' value='15.0'/>       \n"
    "      <Parameter  name='Initial Yield Stress' type='double' value='3.0'/>               \n"
    "      <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
    "      <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "      <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
    "      <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

    const     Plato::OrdinalType tNumCells            = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerNode         = PhysicsT::mNumDofsPerNode;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = PhysicsT::mNumLocalDofsPerCell;

    // Create control
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(0.9, tControl);

    // Create global state
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tGlobalState("Global State", tTotalNumDofs);
    Plato::fill(0.0, tGlobalState);
    Plato::OrdinalType tDispX = 0;
    Plato::OrdinalType tDispY = 1;
    Plato::OrdinalType tDispZ = 2;
    // set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispX, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispY, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispZ, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispX, 0.1);
    // set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispY, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispZ, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "z1", tGlobalState, tDofsPerNode, tDispX, 0.1);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "z1", tGlobalState, tDofsPerNode, tDispY, 0.1);
    // set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "z1", tGlobalState, tDofsPerNode, tDispZ, 0.1);

    // Create previous global state
    Plato::ScalarVector tPrevGlobalState("Previous Global State", tTotalNumDofs);
    Plato::fill(0.0, tPrevGlobalState);

    // Create local state
    const Plato::OrdinalType tNumLocalDofs = tNumLocalDofsPerCell * tNumCells;
    Plato::ScalarVector tLocalState("local state", tNumLocalDofs);
    Plato::fill(0.0, tLocalState);
    Plato::OrdinalType tAccumulatedPlasticStrain   = 0;
    Plato::OrdinalType tPlasticMultiplierIncrement = 1;
    Plato::OrdinalType tPlasticStrainXX = 2;
    Plato::OrdinalType tPlasticStrainYY = 3;
    Plato::OrdinalType tPlasticStrainZZ = 4;
    Plato::OrdinalType tPlasticStrainYZ = 5;
    Plato::OrdinalType tPlasticStrainXZ = 6;
    Plato::OrdinalType tPlasticStrainXY = 7;
    Plato::OrdinalType tBackstressXX =  8;
    Plato::OrdinalType tBackstressYY =  9;
    Plato::OrdinalType tBackstressZZ = 10;
    Plato::OrdinalType tBackstressYZ = 11;
    Plato::OrdinalType tBackstressXZ = 12;
    Plato::OrdinalType tBackstressXY = 13;
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tAccumulatedPlasticStrain, 0.3);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticMultiplierIncrement, 0.5);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainXX, -0.1);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainYY, 0.0);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainZZ, 0.1);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainXY, 0.2);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainYZ, 0.2);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainXZ, 0.2);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tBackstressXX, 12.0);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tBackstressYY, -4.0);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tBackstressZZ, -8.0);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tBackstressXY, 1.0);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tBackstressYZ, 1.0);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tBackstressXZ, 1.0);

    // Create previous local state
    
    Plato::ScalarVector tPrevLocalState("Previous Local State", tNumLocalDofs);
    Plato::fill(0.0, tPrevLocalState);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tAccumulatedPlasticStrain, 0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticMultiplierIncrement, 0.75);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainXX, 0.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainYY, -0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainZZ, 0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainXY, 0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainYZ, 0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainXZ, 0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressXX, 2.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressYY, 40.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressZZ, -42.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressXY, 2.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressYZ, 2.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressXZ, 2.0);

    std::string tProblemType = "J2Plasticity";
    Plato::LocalVectorFunctionInc<PhysicsT> tLocalVectorFuncInc(*tMesh, tMeshSets, tDataMap,
                                                                *tParamList, tProblemType);

    Plato::ScalarVector tLocalResidual = tLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                   tLocalState, tPrevLocalState,
                                                                   tControl, 0.0); 

    constexpr Plato::Scalar tTolerance = 1.0e-5;
    auto tHostLocalResidual = Kokkos::create_mirror(tLocalResidual);
    Kokkos::deep_copy(tHostLocalResidual, tLocalResidual);

    std::vector<Plato::Scalar> tGold = {-0.400000,26.941399,-0.480125,0.111393,
        0.368732,0.022152,0.022152,0.022152,7.078994,-44.680887,37.601894,
        -0.829778,-0.829778,-0.829778};
    for (Plato::OrdinalType tIndex = 0; tIndex < tNumLocalDofs; ++tIndex)
        TEST_FLOATING_EQUALITY(tHostLocalResidual(tIndex), 
                                tGold[tIndex % tNumLocalDofsPerCell], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2Plasticity_Evaluate2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::MeshSets tMeshSets;

    // ### NOTICE THAT THIS IS ONLY PLASTICITY (NO TEMPERATURE) ###
    using PhysicsT = Plato::Plasticity<tSpaceDim>;

    using EvalType = typename Plato::Evaluation<PhysicsT>::Residual;

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='520.0'/>                   \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e2'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='910.0'/> \n"
    "      <Parameter  name='Reference Temperature' type='double' value='100.0'/>            \n"
    "    </ParameterList>                                                                    \n"
    "    <ParameterList name='J2 Plasticity'>                                                \n"
    "      <Parameter  name='Hardening Modulus Isotropic' type='double' value='20.0'/>       \n"
    "      <Parameter  name='Hardening Modulus Kinematic' type='double' value='15.0'/>       \n"
    "      <Parameter  name='Initial Yield Stress' type='double' value='3.0'/>               \n"
    "      <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
    "      <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "      <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
    "      <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

    const     Plato::OrdinalType tNumCells            = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerNode         = PhysicsT::mNumDofsPerNode;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = PhysicsT::mNumLocalDofsPerCell;

    // Create control
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(0.9, tControl);

    // Create global state
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tGlobalState("Global State", tTotalNumDofs);
    Plato::fill(0.0, tGlobalState);
    Plato::OrdinalType tDispX = 0;
    Plato::OrdinalType tDispY = 1;
    set_dof_in_scalar_vector_on_boundary_2D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispY, 0.1);
    set_dof_in_scalar_vector_on_boundary_2D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispX, 0.1);

    // Create previous global state
    Plato::ScalarVector tPrevGlobalState("Previous Global State", tTotalNumDofs);
    Plato::fill(0.0, tPrevGlobalState);

    // Create local state
    const Plato::OrdinalType tNumLocalDofs = tNumLocalDofsPerCell * tNumCells;
    Plato::ScalarVector tLocalState("local state", tNumLocalDofs);
    Plato::fill(0.0, tLocalState);
    Plato::OrdinalType tAccumulatedPlasticStrain   = 0;
    Plato::OrdinalType tPlasticMultiplierIncrement = 1;
    Plato::OrdinalType tPlasticStrainXX = 2;
    Plato::OrdinalType tPlasticStrainYY = 3;
    Plato::OrdinalType tPlasticStrainXY = 4;
    Plato::OrdinalType tPlasticStrainZZ = 5;
    Plato::OrdinalType tBackstressXX = 6;
    Plato::OrdinalType tBackstressYY = 7;
    Plato::OrdinalType tBackstressXY = 8;
    Plato::OrdinalType tBackstressZZ = 9;
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tAccumulatedPlasticStrain, 0.3);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticMultiplierIncrement, 0.5);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainXX, -0.1);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainYY, 0.0);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainXY, 0.2);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tPlasticStrainZZ, 0.);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tBackstressXX, 12.0);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tBackstressYY, -4.0);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tBackstressXY, 1.0);
    set_dof_in_scalar_vector(tLocalState, tNumLocalDofsPerCell, tBackstressZZ, 0.);

    // Create previous local state
    
    Plato::ScalarVector tPrevLocalState("Previous Local State", tNumLocalDofs);
    Plato::fill(0.0, tPrevLocalState);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tAccumulatedPlasticStrain, 0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticMultiplierIncrement, 0.75);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainXX, 0.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainYY, -0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainXY, 0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainZZ, 0.);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressXX, 2.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressYY, 40.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressXY, 2.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressZZ, 0.);

    std::string tProblemType = "J2Plasticity";
    Plato::LocalVectorFunctionInc<PhysicsT> tLocalVectorFuncInc(*tMesh, tMeshSets, tDataMap,
                                                                *tParamList, tProblemType);

    Plato::ScalarVector tLocalResidual = tLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                   tLocalState, tPrevLocalState,
                                                                   tControl, 0.0); 

    constexpr Plato::Scalar tTolerance = 1.0e-4;
    auto tHostLocalResidual = Kokkos::create_mirror(tLocalResidual);
    Kokkos::deep_copy(tHostLocalResidual, tLocalResidual);

    std::vector<Plato::Scalar> tGold = {-0.400000,9.72218,
                                        -0.435375,0.457842,0.0450773,0.438152,
                                        7.42286,-42.0187,-0.653611,3.3669};
    for (Plato::OrdinalType tIndex = 0; tIndex < tNumLocalDofs; ++tIndex)
        TEST_FLOATING_EQUALITY(tHostLocalResidual(tIndex), 
                                tGold[tIndex % tNumLocalDofsPerCell], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2Plasticity_UpdateLocalState3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::MeshSets tMeshSets;

    // ### NOTICE THAT THIS IS ONLY PLASTICITY (NO TEMPERATURE) ###
    using PhysicsT = Plato::Plasticity<tSpaceDim>;

    using EvalType = typename Plato::Evaluation<PhysicsT>::Residual;

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='520.0'/>                   \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e2'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='910.0'/> \n"
    "      <Parameter  name='Reference Temperature' type='double' value='100.0'/>            \n"
    "    </ParameterList>                                                                    \n"
    "    <ParameterList name='J2 Plasticity'>                                                \n"
    "      <Parameter  name='Hardening Modulus Isotropic' type='double' value='20.0'/>       \n"
    "      <Parameter  name='Hardening Modulus Kinematic' type='double' value='15.0'/>       \n"
    "      <Parameter  name='Initial Yield Stress' type='double' value='3.0'/>               \n"
    "      <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
    "      <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "      <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
    "      <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

    const     Plato::OrdinalType tNumCells            = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerNode         = PhysicsT::mNumDofsPerNode;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = PhysicsT::mNumLocalDofsPerCell;

    // Create control
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(0.9, tControl);

    // Create global state
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tGlobalState("Global State", tTotalNumDofs);
    Plato::fill(0.0, tGlobalState);
    Plato::OrdinalType tDispX = 0;
    Plato::OrdinalType tDispY = 1;
    Plato::OrdinalType tDispZ = 2;
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispY, 0.2);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispZ, 0.2);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispX, 0.2);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispZ, 0.2);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "z1", tGlobalState, tDofsPerNode, tDispX, 0.2);
    set_dof_in_scalar_vector_on_boundary_3D(*tMesh, "z1", tGlobalState, tDofsPerNode, tDispY, 0.2);

    // Create previous global state
    Plato::ScalarVector tPrevGlobalState("Previous Global State", tTotalNumDofs);
    Plato::fill(0.0, tPrevGlobalState);

    // Create local state
    const Plato::OrdinalType tNumLocalDofs = tNumLocalDofsPerCell * tNumCells;
    Plato::ScalarVector tLocalState("local state", tNumLocalDofs);
    Plato::fill(0.0, tLocalState);
    Plato::OrdinalType tAccumulatedPlasticStrain   = 0;
    Plato::OrdinalType tPlasticMultiplierIncrement = 1;
    Plato::OrdinalType tPlasticStrainXX = 2;
    Plato::OrdinalType tPlasticStrainYY = 3;
    Plato::OrdinalType tPlasticStrainZZ = 4;
    Plato::OrdinalType tPlasticStrainYZ = 5;
    Plato::OrdinalType tPlasticStrainXZ = 6;
    Plato::OrdinalType tPlasticStrainXY = 7;
    Plato::OrdinalType tBackstressXX =  8;
    Plato::OrdinalType tBackstressYY =  9;
    Plato::OrdinalType tBackstressZZ = 10;
    Plato::OrdinalType tBackstressYZ = 11;
    Plato::OrdinalType tBackstressXZ = 12;
    Plato::OrdinalType tBackstressXY = 13;

    // Create previous local state
    
    Plato::ScalarVector tPrevLocalState("Previous Local State", tNumLocalDofs);
    Plato::fill(0.0, tPrevLocalState);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tAccumulatedPlasticStrain, 0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticMultiplierIncrement, 0.75);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainXX, 0.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainYY, -0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainZZ, 0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainXY, 0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainYZ, 0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainXZ, 0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressXX, 2.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressYY, 40.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressZZ, -42.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressXY, 2.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressYZ, 2.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressXZ, 2.0);

    std::string tProblemType = "J2Plasticity";
    Plato::LocalVectorFunctionInc<PhysicsT> tLocalVectorFuncInc(*tMesh, tMeshSets, tDataMap,
                                                                *tParamList, tProblemType);

    tLocalVectorFuncInc.updateLocalState(tGlobalState, tPrevGlobalState,
                                         tLocalState, tPrevLocalState,
                                         tControl, 0.0); 

    constexpr Plato::Scalar tTolerance = 1.0e-5;
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    Kokkos::deep_copy(tHostLocalState, tLocalState);

    std::vector<Plato::Scalar> tGold = {0.3755345,0.1755345,
        -0.00606135,-0.144478,0.15053935,2.0*0.18231319,2.0*0.18231319,2.0*0.18231319,
        1.95342253,40.42664966,-42.38007219,2.63252209,2.63252209,2.63252209};
    for (Plato::OrdinalType tIndex = 0; tIndex < tNumLocalDofs; ++tIndex)
        TEST_FLOATING_EQUALITY(tHostLocalState(tIndex), 
                                tGold[tIndex % tNumLocalDofsPerCell], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2Plasticity_UpdateLocalState2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::MeshSets tMeshSets;

    // ### NOTICE THAT THIS IS ONLY PLASTICITY (NO TEMPERATURE) ###
    using PhysicsT = Plato::Plasticity<tSpaceDim>;

    using EvalType = typename Plato::Evaluation<PhysicsT>::Residual;

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='520.0'/>                   \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e2'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='910.0'/> \n"
    "      <Parameter  name='Reference Temperature' type='double' value='100.0'/>            \n"
    "    </ParameterList>                                                                    \n"
    "    <ParameterList name='J2 Plasticity'>                                                \n"
    "      <Parameter  name='Hardening Modulus Isotropic' type='double' value='20.0'/>       \n"
    "      <Parameter  name='Hardening Modulus Kinematic' type='double' value='15.0'/>       \n"
    "      <Parameter  name='Initial Yield Stress' type='double' value='3.0'/>               \n"
    "      <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
    "      <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "      <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
    "      <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

    const     Plato::OrdinalType tNumCells            = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerNode         = PhysicsT::mNumDofsPerNode;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = PhysicsT::mNumLocalDofsPerCell;

    // Create control
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(0.9, tControl);

    // Create global state
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tGlobalState("Global State", tTotalNumDofs);
    Plato::fill(0.0, tGlobalState);
    Plato::OrdinalType tDispX = 0;
    Plato::OrdinalType tDispY = 1;
    set_dof_in_scalar_vector_on_boundary_2D(*tMesh, "x1", tGlobalState, tDofsPerNode, tDispY, 0.2);
    set_dof_in_scalar_vector_on_boundary_2D(*tMesh, "y1", tGlobalState, tDofsPerNode, tDispX, 0.2);

    // Create previous global state
    Plato::ScalarVector tPrevGlobalState("Previous Global State", tTotalNumDofs);
    Plato::fill(0.0, tPrevGlobalState);

    // Create local state
    const Plato::OrdinalType tNumLocalDofs = tNumLocalDofsPerCell * tNumCells;
    Plato::ScalarVector tLocalState("local state", tNumLocalDofs);
    Plato::fill(0.0, tLocalState);
    Plato::OrdinalType tAccumulatedPlasticStrain   = 0;
    Plato::OrdinalType tPlasticMultiplierIncrement = 1;
    Plato::OrdinalType tPlasticStrainXX = 2;
    Plato::OrdinalType tPlasticStrainYY = 3;
    Plato::OrdinalType tPlasticStrainXY = 4;
    Plato::OrdinalType tPlasticStrainZZ = 5;
    Plato::OrdinalType tBackstressXX =  6;
    Plato::OrdinalType tBackstressYY =  7;
    Plato::OrdinalType tBackstressXY =  8;
    Plato::OrdinalType tBackstressZZ =  9;

    // Create previous local state
    
    Plato::ScalarVector tPrevLocalState("Previous Local State", tNumLocalDofs);
    Plato::fill(0.0, tPrevLocalState);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tAccumulatedPlasticStrain, 0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticMultiplierIncrement, 0.75);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainXX, 0.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainYY, -0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainXY, 0.2);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tPlasticStrainZZ, -0.01);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressXX, 2.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressYY, 40.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressXY, 2.0);
    set_dof_in_scalar_vector(tPrevLocalState, tNumLocalDofsPerCell, tBackstressZZ, -0.02);

    std::string tProblemType = "J2Plasticity";
    Plato::LocalVectorFunctionInc<PhysicsT> tLocalVectorFuncInc(*tMesh, tMeshSets, tDataMap,
                                                                *tParamList, tProblemType);

    tLocalVectorFuncInc.updateLocalState(tGlobalState, tPrevGlobalState,
                                         tLocalState, tPrevLocalState,
                                         tControl, 0.0); 

    constexpr Plato::Scalar tTolerance = 1.0e-4;
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    Kokkos::deep_copy(tHostLocalState, tLocalState);

    std::vector<Plato::Scalar> tGold = {0.314575,0.114575,
                                        -0.0657574,-0.206138,0.359376,-0.0612751,
                                        1.4947,39.9528,2.61235,-0.414015};
    for (Plato::OrdinalType tIndex = 0; tIndex < tNumLocalDofsPerCell; ++tIndex)
        TEST_FLOATING_EQUALITY(tHostLocalState(tIndex), 
                                tGold[tIndex % tNumLocalDofsPerCell], tTolerance);
}


} // namespace AugLagStressTest
