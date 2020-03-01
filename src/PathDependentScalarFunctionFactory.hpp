/*
 * PathDependentScalarFunctionFactory.hpp
 *
 *  Created on: Mar 1, 2020
 */

#pragma once

#include "BasicLocalScalarFunctionInc.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Factory for scalar functions interface with local path-dependent states
 **********************************************************************************/
template<typename PhysicsT>
class PathDependentScalarFunctionFactory
{
public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    PathDependentScalarFunctionFactory () {}

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    ~PathDependentScalarFunctionFactory() {}

    /******************************************************************************//**
     * \brief Create interface for the evaluation of path-dependent scalar function
     *  operators, e.g. value and sensitivities.
     * \param [in] aMesh         mesh database
     * \param [in] aMeshSets     side sets database
     * \param [in] aDataMap      output data map for PLATO Analyze
     * \param [in] aInputParams  problem inputs in XML file
     * \param [in] aFunctionName scalar function name, i.e. type
     * \return shared pointer to the interface of path-dependent scalar functions
     **********************************************************************************/
    std::shared_ptr<Plato::LocalScalarFunctionInc>
    create(Omega_h::Mesh& aMesh,
           Omega_h::MeshSets& aMeshSets,
           Plato::DataMap & aDataMap,
           Teuchos::ParameterList& aInputParams,
           std::string& aFunctionName)
    {
        auto tProblemFunction = aInputParams.sublist(aFunctionName);
        auto tFunctionType = tProblemFunction.get < std::string > ("Type", "UNDEFINED");
        if(tFunctionType == "Scalar Function")
        {
            return ( std::make_shared <Plato::BasicLocalScalarFunctionInc<PhysicsT>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, aFunctionName) );
        }
        else
        {
            const auto tError = std::string("UNKNOWN SCALAR FUNCTION '") + tFunctionType
                    + "'. OBJECTIVE OR CONSTRAINT KEYWORD WITH NAME '" + aFunctionName
                    + "' IS NOT DEFINED.  MOST LIKELY, SUBLIST '" + aFunctionName
                    + "' IS NOT DEFINED IN THE INPUT FILE.";
            THROWERR(tError);
        }
    }
};
// class ScalarFunctionBaseFactory

}
// namespace Plato