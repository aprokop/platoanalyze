/*
 * PlatoProblemFactory.hpp
 *
 *  Created on: Apr 19, 2018
 */

#ifndef PLATOPROBLEMFACTORY_HPP_
#define PLATOPROBLEMFACTORY_HPP_

#include <memory>
#include <sstream>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include <Teuchos_ParameterList.hpp>

#include "plato/EllipticProblem.hpp"
#include "plato/EllipticVMSProblem.hpp"
#include "plato/ParabolicProblem.hpp"
#include "plato/AnalyzeMacros.hpp"

#include "plato/Mechanics.hpp"
#include "plato/StabilizedMechanics.hpp"
#include "plato/Electromechanics.hpp"
#include "plato/Thermomechanics.hpp"
#include "plato/StabilizedThermomechanics.hpp"
//#include "plato/StructuralDynamics.hpp"
//#include "plato/StructuralDynamicsProblem.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Manages the construction of a PLATO problem, i.e. physics simulation, in
 * PLATO Analyze.  There are multiple options: 1) mechanics, thermal, thermo-mechanics,
 * and electro-statics.  Also, PLATO Analyze supports transient simulations for thermal
 * and thermo-mechanics problems.
 **********************************************************************************/
template<int SpatialDim>
class ProblemFactory
{
public:
    /******************************************************************************//**
     * @brief Returns a shared pointer to a PLATO problem
     * @param [in] aMesh mesh metadata
     * @param [in] aMeshSets sidesets mesh metadata
     * @param [in] aInputParams xml metadata
     * @returns shared pointer to a PLATO problem
     **********************************************************************************/
    std::shared_ptr<Plato::AbstractProblem> create(Omega_h::Mesh& aMesh,
                                                   Omega_h::MeshSets& aMeshSets,
                                                   Teuchos::ParameterList& aInputParams)
    {

        auto tProblemSpecs = aInputParams.sublist("Plato Problem");
        auto tProblemPhysics = tProblemSpecs.get < std::string > ("Physics");
        auto tProblemPDE = tProblemSpecs.get < std::string > ("PDE Constraint");

        if(tProblemPhysics == "Mechanical")
        {
            if(tProblemPDE == "Elliptic")
            {
                return std::make_shared < EllipticProblem<::Plato::Mechanics<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
            }
            else
            {
                std::stringstream ss;
                ss << "Unknown PDE type (" << tProblemPDE << ") requested.";
                THROWERR(ss.str());
            }
        }
        else if(tProblemPhysics == "Stabilized Mechanical")
        {
            if(tProblemPDE == "Elliptic")
            {
                return std::make_shared < EllipticVMSProblem<::Plato::StabilizedMechanics<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
            }
            else
            {
                std::stringstream tStringStream;
                tStringStream << "Unknown PDE type (" << tProblemPDE << ") requested.";
                THROWERR(tStringStream.str());
            }
        }
        else if(tProblemPhysics == "Thermal")
        {
            if(tProblemPDE == "Heat Equation")
            {
                return std::make_shared < ParabolicProblem<::Plato::Thermal<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
            }
            else if(tProblemPDE == "Thermostatics")
            {
                return std::make_shared < EllipticProblem<::Plato::Thermal<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
            }
            else
            {
                std::stringstream tStringStream;
                tStringStream << "Unknown PDE type (" << tProblemPDE << ") requested.";
                THROWERR(tStringStream.str());
            }
        }
        else if(tProblemPhysics == "StructuralDynamics")
        {
//            return std::make_shared<Plato::StructuralDynamicsProblem<Plato::StructuralDynamics<SpatialDim>>>(aMesh, aMeshSets, tProblemSpecs);
        }
        else if(tProblemPhysics == "Electromechanical")
        {
            return std::make_shared < EllipticProblem<::Plato::Electromechanics<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
        }
        else if(tProblemPhysics == "Stabilized Thermomechanical")
        {
            if(tProblemPDE == "Elliptic")
            {
                return std::make_shared < EllipticVMSProblem<::Plato::StabilizedThermomechanics<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
            }
            else
            {
                std::stringstream ss;
                ss << "Unknown PDE type (" << tProblemPDE << ") requested.";
                THROWERR(ss.str());
            }
        }
        else if(tProblemPhysics == "Thermomechanical")
        {
            if(tProblemPDE == "Parabolic")
            {
                return std::make_shared < ParabolicProblem<::Plato::Thermomechanics<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
            }
            else if(tProblemPDE == "Elliptic")
            {
                return std::make_shared < EllipticProblem<::Plato::Thermomechanics<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
            }
            else
            {
                std::stringstream ss;
                ss << "Unknown PDE type (" << tProblemPDE << ") requested.";
                THROWERR(ss.str());
            }
        }
        else
        {
            std::stringstream tStringStream;
            tStringStream << "Unknown Physics type (" << tProblemPhysics << ") requested.";
            THROWERR(tStringStream.str());
        }

        return (nullptr);
    }
};
// class ProblemFactory

}// namespace Plato

#endif /* PLATOPROBLEMFACTORY_HPP_ */
