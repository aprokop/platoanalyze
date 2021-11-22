/*
 * CriterionMeanTemperature.hpp
 *
 *  Created on: Nov 3, 2021
 */

#pragma once

#include "SpatialModel.hpp"
#include "ExpInstMacros.hpp"

#include "hyperbolic/SimplexFluids.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Fluids
{

template<typename PhysicsT, typename EvaluationT>
class CriterionMeanTemperature : public Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims     = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell    = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell; /*!< number of degrees of freedom per cell */

    // set local AD typenames
    using ResultT  = typename EvaluationT::ResultScalarType; /*!< result FAD type */
    using CurTempT = typename EvaluationT::CurrentEnergyScalarType; /*!< current temperature FAD evaluation type */

    // member parameters
    std::string mFuncName; /*!< scalar funciton name */

    // member metadata
    Plato::DataMap& mDataMap; /*!< holds output metadata */
    const Plato::SpatialDomain& mSpatialDomain; /*!< holds mesh and entity sets metadata for a volume domain (i.e. element block) */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aName    scalar function name
     * \param [in] aDomain  spatial domain metadata
     * \param [in] aDataMap output database
     * \param [in] aInputs  input database
     ******************************************************************************/
    CriterionMeanTemperature
    (const std::string          & aName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mFuncName(aName)
    { return; }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~CriterionMeanTemperature(){ return; }

    /***************************************************************************//**
     * \fn std::string name
     * \brief Returns scalar function name
     * \return scalar function name
     ******************************************************************************/
    std::string name() const override { return mFuncName; }

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate scalar function within the computational domain \f$ \Omega \f$.
     * \param [in] aWorkSets holds state work sets initialize with correct FAD types
     * \param [in] aResult   1D output work set of size number of cells
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets &aWorkSets, 
     Plato::ScalarVectorT<ResultT> &aResultWS) const
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "cell number does not match. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' cells/elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' cells/elements.")
        }

        // set current state worksets
        auto tCurTempWS = Plato::metadata<Plato::ScalarMultiVectorT<CurTempT>>(aWorkSets.get("current temperature"));
        
        // calculate mean temperature criteria    
        auto tNumNodes = mSpatialDomain.numNodes();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {               
            for(Plato::OrdinalType tDof = 0; tDof < mNumTempDofsPerCell; tDof++)
            {
                aResultWS(aCellOrdinal) += tCurTempWS(aCellOrdinal, tDof);
            }
            aResultWS(aCellOrdinal) *= static_cast<Plato::Scalar>(1.0 / tNumNodes);
        }, "calculate mean temperature");
    }


    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate scalar function along the computational boudary \f$ \Gamma \f$.
     * \param [in] aSpatialModel holds mesh and entity sets (e.g. node and side sets) metadata
     * \param [in] aWorkSets holds state work sets initialize with correct FAD types
     * \param [in] aResult   1D output work set of size number of cells
     ******************************************************************************/
    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel, 
     const Plato::WorkSets & aWorkSets, 
     Plato::ScalarVectorT<ResultT> & aResult) 
    const override
    { return; }
};
// class CriterionMeanTemperature

}
// namespace Fluids

}
// namespace Plato