/*
 * CriterionVolume.hpp
 *
 *  Created on: Jul 22, 2021
 */

#pragma once

#include "SpatialModel.hpp"
#include "UtilsTeuchos.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/SimplexFluids.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Fluids
{

template<typename PhysicsT, typename EvaluationT>
class CriterionVolume : public Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell = PhysicsT::SimplexT::mNumNodesPerCell; /*!< number of nodes per cell */

    // set local FAD types
    using ResultT  = typename EvaluationT::ResultScalarType; /*!< result FAD type */
    using ConfigT  = typename EvaluationT::ConfigScalarType; /*!< configuration FAD type */
    using ControlT = typename EvaluationT::ControlScalarType; /*!< control FAD evaluation type */

    // set local typenames
    using CubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>; /*!< local short name for cubature rule class */

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< spatial domain metadata */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature integration rule */
    
    std::string mFuncName; /*!< scalar funciton name */
    std::vector<std::string> mBlocks; /*!< element blocks considered for the volume integral caclulation */
    Plato::Scalar mPenaltyExponent = 3.0; /*!< penalty exponent used for volume penalization */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aName    scalar function name
     * \param [in] aDomain  holds mesh and entity sets (e.g. node and side sets) metadata
     * \param [in] aDataMap holds output metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    CriterionVolume
    (const std::string          & aName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
        mDataMap(aDataMap),
        mCubatureRule(CubatureRule()),
        mSpatialDomain(aDomain),
        mFuncName(aName)
    {
        auto tMyCriteria = aInputs.sublist("Criteria").sublist(aName);
        mBlocks = Plato::teuchos::parse_array<std::string>("Blocks", tMyCriteria);
        if( mBlocks.empty() )
        {
            // default: all the element blocks will be used in the volume evaluation
            auto tMyBlockName = mSpatialDomain.getElementBlockName();
            mBlocks.push_back(tMyBlockName);
        }

        if (tMyCriteria.isSublist("Penalty Function"))
        {
            auto tPenaltyFuncList = tMyCriteria.sublist("Penalty Function");
            if(tPenaltyFuncList.isParameter("Penalty Exponent"))
            {
                mPenaltyExponent = tPenaltyFuncList.get<Plato::Scalar>("Penalty Exponent");
            }
        }
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~CriterionVolume(){}

    /***************************************************************************//**
     * \fn std::string name
     * \brief Returns scalar function name
     * \return scalar function name
     ******************************************************************************/
    std::string name() const override { return mFuncName; }

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate scalar function inside the computational domain \f$ \Omega \f$.
     * \param [in] aWorkSets holds state work sets initialized with correct FAD types
     * \param [in] aResult   1D output work set of size number of cells
     ******************************************************************************/
    void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    {
        auto tMyBlockName = mSpatialDomain.getElementBlockName();
        auto tEvaluateDomain = std::find(mBlocks.begin(), mBlocks.end(), tMyBlockName) != mBlocks.end();
        if(tEvaluateDomain)
        {
            // local functors
            Plato::ComputeCellVolume<mNumSpatialDims> tComputeCellVolume;

            // set local input worksets
            auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
            auto tControlWS  = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));

            // transfer member data to device
            auto tPenaltyExponent = mPenaltyExponent;
            auto tCubatureWeight = mCubatureRule.getCubWeight();
            auto tBasisFunctions = mCubatureRule.getBasisFunctions();

            auto tNumCells = mSpatialDomain.numCells();
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                ConfigT tCellVolume;
                tComputeCellVolume(aCellOrdinal, tConfigWS, tCellVolume);
                tCellVolume *= tCubatureWeight;

                ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, tControlWS);
                ControlT tPenalizedDensity = pow(tDensity, tPenaltyExponent);
                aResult(aCellOrdinal) = tPenalizedDensity * tCellVolume;
            },"volume");
        }
    }

    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel, 
     const Plato::WorkSets & aWorkSets, 
     Plato::ScalarVectorT<ResultT> & aResult) 
     const override
    { return; }
};
// class CriterionVolume

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::CriterionVolume, Plato::IncompressibleFluids, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::CriterionVolume, Plato::IncompressibleFluids, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::CriterionVolume, Plato::IncompressibleFluids, Plato::SimplexFluids, 3, 1)
#endif