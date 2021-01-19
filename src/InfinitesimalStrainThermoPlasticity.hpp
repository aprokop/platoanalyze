/*
 * InfinitesimalStrainThermoPlasticity.hpp
 *
 *  Created on: Jan 20, 2021
 */

#pragma once

#include "Projection.hpp"
#include "ElasticWorkCriterion.hpp"
#include "PlasticWorkCriterion.hpp"
#include "InfinitesimalStrainThermoPlasticityResidual.hpp"

namespace Plato
{

namespace InfinitesimalStrainThermoPlasticityFactory
{

/*********************************************************************************//**
 * \brief Factory for stabilized infinitesimal strain thermoplasticity vector function.
*************************************************************************************/
struct FunctionFactory
{
    /***************************************************************************//**
     * \brief Create a stabilized vector function with local path-dependent states
     *  (e.g. thermoplasticity)
     *
     * \tparam automatic differentiation evaluation type, e.g. JacobianU, JacobianZ, etc.
     *
     * \param [in] aMesh         mesh database
     * \param [in] aMeshSets     side sets database
     * \param [in] aDataMap      output data database
     * \param [in] aInputParams  input parameters
     * \param [in] aFunctionName vector function name
     *
     * \return shared pointer to stabilized vector function with local path-dependent states
    *******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<EvaluationType>>
    createGlobalVectorFunctionInc(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
              std::string              aFunctionName
    )
    {
        if(aFunctionName == "Elliptic")
        {
            constexpr auto tSpaceDim = EvaluationType::SpatialDim;
            return ( std::make_shared<Plato::InfinitesimalStrainThermoPlasticityResidual<EvaluationType, Plato::SimplexThermoPlasticity<tSpaceDim>>>
                    (aSpatialDomain, aDataMap, aInputParams) );
        }
        else
        {
            const auto tError = std::string("Unknown Vector Function with path-dependent states. '")
                    + "User specified '" + aFunctionName + "'.  This Vector Function is not supported in PLATO.";
            THROWERR(tError)
        }
    }

    /***************************************************************************//**
     * \brief Create a scalar function with local path-dependent states (e.g. thermoplasticity)
     *
     * \tparam automatic differentiation evaluation type, e.g. JacobianU, JacobianZ, etc.
     *
     * \param [in] aMesh        mesh database
     * \param [in] aMeshSets    side sets database
     * \param [in] aDataMap     output data database
     * \param [in] aInputParams input parameters
     * \param [in] aFuncType    function type, used to identify requested function
     * \param [in] aFuncName    user defined name for requested function
     *
     * \return shared pointer to scalar function with local path-dependent states
    *******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<EvaluationType>>
    createLocalScalarFunctionInc(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
              std::string              aFuncType,
              std::string              aFuncName
    )
    {
        if(aFuncType == "Plastic Work")
        {
            constexpr auto tSpaceDim = EvaluationType::SpatialDim;
            return ( std::make_shared<Plato::PlasticWorkCriterion<EvaluationType, Plato::SimplexThermoPlasticity<tSpaceDim>>>
                    (aSpatialDomain, aDataMap, aInputParams, aFuncName) );
        } else
        if(aFuncType == "Elastic Work")
        {
            constexpr auto tSpaceDim = EvaluationType::SpatialDim;
            return ( std::make_shared<Plato::ElasticWorkCriterion<EvaluationType, Plato::SimplexThermoPlasticity<tSpaceDim>>>
                    (aSpatialDomain, aDataMap, aInputParams, aFuncName) );
        }
        else
        {
            const auto tError = std::string("Unknown Scalar Function with local path-dependent states. '")
                    + "User specified '" + aFuncType + "'.  This Scalar Function is not supported in PLATO.";
            THROWERR(tError)
        }
    }
};
// struct FunctionFactory

}
// namespace InfinitesimalStrainThermoPlasticityFactory

/*************************************************************************//**
 * \brief Defines the concrete physics Type templates for an infinitesimal
 * strain plasticity application.  An infinitesimal strain plasticity application
 * is defined by an implicitly integrated in time stabilized Partial Differential
 * Equation (PDE).  The stabilization technique is based on a Variational Multiscale
 * (VMS) approach.
*****************************************************************************/
template<Plato::OrdinalType NumSpaceDim>
class InfinitesimalStrainThermoPlasticity: public Plato::SimplexThermoPlasticity<NumSpaceDim>
{
public:
    static constexpr auto mSpaceDim = NumSpaceDim; /*!< number of spatial dimensions */

    /*!< short name for plasticity factory */
    typedef Plato::InfinitesimalStrainThermoPlasticityFactory::FunctionFactory FunctionFactory;

    /*!< short name for simplex thermoplasticity physics */
    using SimplexT = Plato::SimplexThermoPlasticity<NumSpaceDim>;

    /*!< short name for projected pressure gradient physics */
    using ProjectorT = typename Plato::Projection<NumSpaceDim, SimplexT::mNumDofsPerNode, SimplexT::mPressureDofOffset>;
};
// class InfinitesimalStrainThermoPlasticity

}
// namespace Plato
