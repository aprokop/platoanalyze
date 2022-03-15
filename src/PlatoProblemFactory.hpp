/*
 * PlatoProblemFactory.hpp
 *
 *  Created on: Apr 19, 2018
 */

#ifndef PLATOPROBLEMFACTORY_HPP_
#define PLATOPROBLEMFACTORY_HPP_

#include <memory>
#include <sstream>
#include <Teuchos_ParameterList.hpp>

#include "PlatoMesh.hpp"
#include "AnalyzeMacros.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"
#include "alg/ParallelComm.hpp"

#ifdef PLATO_PLASTICITY
#include "PlasticityProblem.hpp"
#endif

#ifdef PLATO_ELLIPTIC
#include "elliptic/Problem.hpp"
#include "elliptic/updated_lagrangian/Problem.hpp"
#endif

#ifdef PLATO_PARABOLIC
#include "parabolic/Problem.hpp"
#endif

#ifdef PLATO_HYPERBOLIC
#include "hyperbolic/HyperbolicProblem.hpp"
#include "hyperbolic/FluidsQuasiImplicit.hpp"
#endif

#ifdef PLATO_STABILIZED
#include "EllipticVMSProblem.hpp"
#include "StabilizedMechanics.hpp"
#include "StabilizedThermomechanics.hpp"
#endif

#ifdef PLATO_HELMHOLTZ
#include "helmholtz/Helmholtz.hpp"
#include "helmholtz/Problem.hpp"
#endif

//#include "StructuralDynamicsProblem.hpp"

namespace Plato
{

/******************************************************************************//**
* \brief Check if input PDE type is supported by Analyze.
* \param [in] aPlatoProb input xml metadata
* \returns return lowercase pde type
**********************************************************************************/
inline std::string
is_pde_constraint_supported
(Teuchos::ParameterList & aPlatoProb)
{
    if(aPlatoProb.isParameter("PDE Constraint") == false)
    {
        ANALYZE_THROWERR("Parameter 'PDE Constraint' is not defined in 'Plato Problem' parameter list.")
    }
    auto tPDE = aPlatoProb.get < std::string > ("PDE Constraint");
    auto tLowerPDE = Plato::tolower(tPDE);
    return tLowerPDE;
}
// function is_pde_constraint_supported

/******************************************************************************//**
* \brief Create mechanical problem.
* \param [in] aMesh        plato abstract mesh
* \param [in] aPlatoProb   input xml metadata
* \param [in] aMachine     mpi communicator interface
* \returns shared pointer to abstract problem of type mechanical
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
inline
std::shared_ptr<Plato::AbstractProblem>
create_mechanical_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
{
    auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_ELLIPTIC
    if (tLowerPDE == "elliptic")
    {
        auto tOutput = std::make_shared<Plato::Elliptic::Problem<::Plato::Mechanics<SpatialDim>>>(aMesh, aPlatoProb, aMachine);
        tOutput->readEssentialBoundaryConditions(aPlatoProb);
        return tOutput;
    }
    else
    if(tLowerPDE == "updated lagrangian elliptic")
    {
        using PhysicsType = Plato::Elliptic::UpdatedLagrangian::Mechanics<SpatialDim>;
        auto tOutput = std::make_shared<Plato::Elliptic::UpdatedLagrangian::Problem<PhysicsType>> (aMesh, aPlatoProb, aMachine);
        return tOutput;
    }
#endif
#ifdef PLATO_HYPERBOLIC
    else
    if (tLowerPDE == "hyperbolic")
    {
        return std::make_shared<HyperbolicProblem<::Plato::Hyperbolic::Mechanics<SpatialDim>>>(aMesh, aPlatoProb, aMachine);
    }
    else
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
}
// function create_mechanical_problem

/******************************************************************************//**
* \brief Create plasticity problem.
* \param [in] aMesh        Plato mesh database
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine     mpi communicator interface
* \returns shared pointer to abstract problem of type plasticity
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
inline
std::shared_ptr<Plato::AbstractProblem>
create_plasticity_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
     auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_ELLIPTIC
    if(tLowerPDE == "elliptic")
    {
        auto tOutput = std::make_shared < PlasticityProblem<::Plato::InfinitesimalStrainPlasticity<SpatialDim>> > (aMesh, aPlatoProb, aMachine);
        tOutput->readEssentialBoundaryConditions(aPlatoProb);
        return tOutput;
    }
    else
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_plasticity_problem

/******************************************************************************//**
* \brief Create a thermoplasticity problem.
* \param [in] aMesh      mesh metadata
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine   mpi communicator interface
* \returns shared pointer to abstract problem of type thermoplasticity
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
inline
std::shared_ptr<Plato::AbstractProblem>
create_thermoplasticity_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
     auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_ELLIPTIC
    if(tLowerPDE == "elliptic")
    {
        auto tOutput = std::make_shared < PlasticityProblem<::Plato::InfinitesimalStrainThermoPlasticity<SpatialDim>> > (aMesh, aPlatoProb, aMachine);
        tOutput->readEssentialBoundaryConditions(aPlatoProb);
        return tOutput;
    }
    else
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_thermoplasticity_problem

/******************************************************************************//**
* \brief Create a abstract problem of type stabilized mechanical.
* \param [in] aMesh      mesh metadata
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine   mpi communicator interface
* \returns shared pointer to abstract problem of type stabilized mechanical
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
inline
std::shared_ptr<Plato::AbstractProblem>
create_stabilized_mechanical_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
     auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);
  #ifdef PLATO_ELLIPTIC
    if(tLowerPDE == "elliptic")
    {
        auto tOutput = std::make_shared < EllipticVMSProblem<::Plato::StabilizedMechanics<SpatialDim>> > (aMesh, aPlatoProb, aMachine);
        tOutput->readEssentialBoundaryConditions(aPlatoProb);
        return tOutput;
    }
    else
  #endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
}
 // function create_stabilized_mechanical_problem

/******************************************************************************//**
* \brief Create a abstract problem of type thermal.
* \param [in] aMesh      mesh metadata
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine   mpi communicator interface
* \returns shared pointer to abstract problem of type thermal
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
inline
std::shared_ptr<Plato::AbstractProblem>
create_thermal_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
     auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_PARABOLIC
    if(tLowerPDE == "parabolic")
    {
        return std::make_shared < Plato::Parabolic::Problem<::Plato::Thermal<SpatialDim>> > (aMesh, aPlatoProb, aMachine);
    }
    else
#endif
#ifdef PLATO_ELLIPTIC
    if(tLowerPDE == "elliptic")
    {
        auto tOutput = std::make_shared < Plato::Elliptic::Problem<::Plato::Thermal<SpatialDim>> > (aMesh, aPlatoProb, aMachine);
        tOutput->readEssentialBoundaryConditions(aPlatoProb);
        return tOutput;
    }
    else
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_thermal_problem

/******************************************************************************//**
* \brief Create a abstract problem of type electromechanical.
* \param [in] aMesh      mesh metadata
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine   mpi communicator interface
* \returns shared pointer to abstract problem of type electromechanical
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
inline
std::shared_ptr<Plato::AbstractProblem>
create_electromechanical_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
     auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_ELLIPTIC
    if(tLowerPDE == "elliptic")
    {
        auto tOutput = std::make_shared < Plato::Elliptic::Problem<::Plato::Electromechanics<SpatialDim>> > (aMesh, aPlatoProb, aMachine);
        tOutput->readEssentialBoundaryConditions(aPlatoProb);
        return tOutput;
    }
    else
#endif
#ifdef PLATO_HYPERBOLIC
    if (tLowerPDE == "hyperbolic")
    {
        return std::make_shared<HyperbolicProblem<::Plato::Hyperbolic::Electromechanics<SpatialDim>>>(aMesh, aPlatoProb, aMachine);
    }
    else
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_electromechanical_problem

/******************************************************************************//**
* \brief Create a abstract problem of type stabilized thermomechanical.
* \param [in] aMesh      mesh metadata
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine   mpi communicator interface
* \returns shared pointer to abstract problem of type stabilized thermomechanical
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
inline
std::shared_ptr<Plato::AbstractProblem>
create_stabilized_thermomechanical_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
    auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_ELLIPTIC
    if(tLowerPDE == "elliptic")
    {
        auto tOutput = std::make_shared < EllipticVMSProblem<::Plato::StabilizedThermomechanics<SpatialDim>> > (aMesh, aPlatoProb, aMachine);
        tOutput->readEssentialBoundaryConditions(aPlatoProb);
        return tOutput;
    }
    else
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_stabilized_thermomechanical_problem

/******************************************************************************//**
* \brief Create a abstract problem of type thermomechanical.
* \param [in] aMesh        mesh metadata
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine     mpi communicator interface
* \returns shared pointer to abstract problem of type thermomechanical
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
inline
std::shared_ptr<Plato::AbstractProblem>
create_thermomechanical_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
    auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_PARABOLIC
    if(tLowerPDE == "parabolic")
    {
        return std::make_shared < Plato::Parabolic::Problem<::Plato::Thermomechanics<SpatialDim>> > (aMesh, aPlatoProb, aMachine);
    }
    else
#endif
#ifdef PLATO_ELLIPTIC
    if(tLowerPDE == "elliptic")
    {
        auto tOutput = std::make_shared < Plato::Elliptic::Problem<::Plato::Thermomechanics<SpatialDim>> > (aMesh, aPlatoProb, aMachine);
        tOutput->readEssentialBoundaryConditions(aPlatoProb);
        return tOutput;
    }
    else
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_thermomechanical_problem

/******************************************************************************//**
* \brief Create a abstract problem of type incompressible fluid.
* \param [in] aMesh        mesh metadata
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine     mpi communicator interface
* \returns shared pointer to abstract problem of type incompressible fluid
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
inline
std::shared_ptr<Plato::AbstractProblem>
create_incompressible_fluid_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
    auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_HYPERBOLIC
    if (tLowerPDE == "hyperbolic")
    {
        return std::make_shared<Plato::Fluids::QuasiImplicit<::Plato::IncompressibleFluids<SpatialDim>>>(aMesh, aPlatoProb, aMachine);
    }
    else
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_incompressible_fluid_problem

/******************************************************************************//**
* \brief Create a abstract problem of micromorphic mechanics.
* \param [in] aMesh        mesh metadata
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine     mpi communicator interface
* \returns shared pointer to abstract problem of type micromorphic mechanics
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
inline
std::shared_ptr<Plato::AbstractProblem>
create_micromorphic_mechanics_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
    auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_HYPERBOLIC
    if (tLowerPDE == "hyperbolic")
    {
        return std::make_shared<HyperbolicProblem<::Plato::Hyperbolic::MicromorphicMechanics<SpatialDim>>>(aMesh, aPlatoProb, aMachine);
    }
    else
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_incompressible_fluid_problem

/******************************************************************************//**
 * \brief This class is responsible for the creation of a Plato problem, which enables
 * finite element simulations of multiphysics problem.
 **********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class ProblemFactory
{
public:
    /******************************************************************************//**
     * \brief Returns a shared pointer to a PLATO problem
     * \param [in] aMesh        abstract mesh
     * \param [in] aInputParams xml metadata
     * \param [in] aMachine     mpi communicator interface
     * \returns shared pointer to a PLATO problem
     **********************************************************************************/
    std::shared_ptr<Plato::AbstractProblem>
    create(
        Plato::Mesh              aMesh,
        Teuchos::ParameterList & aInputParams,
        Comm::Machine            aMachine
    )
    {
        auto tInputData = aInputParams.sublist("Plato Problem");
        auto tPhysics = tInputData.get < std::string > ("Physics");
        auto tLowerPhysics = Plato::tolower(tPhysics);

        if(tLowerPhysics == "mechanical")
        {
            return ( Plato::create_mechanical_problem<SpatialDim>(aMesh, tInputData, aMachine) );
        }
#ifdef PLATO_PLASTICITY
        else
        if(tLowerPhysics == "plasticity")
        {
            return ( Plato::create_plasticity_problem<SpatialDim>(aMesh, tInputData, aMachine) );
        }
        else
        if(tLowerPhysics == "thermoplasticity")
        {
            return ( Plato::create_thermoplasticity_problem<SpatialDim>(aMesh, tInputData, aMachine) );
        }
#endif
#ifdef PLATO_STABILIZED
        else
        if(tLowerPhysics == "stabilized mechanical")
        {
            return ( Plato::create_stabilized_mechanical_problem<SpatialDim>(aMesh, tInputData, aMachine) );
        }
#endif
        else if(tLowerPhysics == "thermal")
        {
            return ( Plato::create_thermal_problem<SpatialDim>(aMesh, tInputData, aMachine) );
        }
        else
        if(tLowerPhysics == "electromechanical")
        {
            return ( Plato::create_electromechanical_problem<SpatialDim>(aMesh, tInputData, aMachine) );
        }
#ifdef PLATO_STABILIZED
        else
        if(tLowerPhysics == "stabilized thermomechanical")
        {
            return ( Plato::create_stabilized_thermomechanical_problem<SpatialDim>(aMesh, tInputData, aMachine) );
        }
#endif
        else
        if(tLowerPhysics == "thermomechanical")
        {
            return ( Plato::create_thermomechanical_problem<SpatialDim>(aMesh, tInputData, aMachine) );
        }
        else
        if(tLowerPhysics == "incompressible fluids")
        {
            return ( Plato::create_incompressible_fluid_problem<SpatialDim>(aMesh, tInputData, aMachine) );
        }
        if(tLowerPhysics == "micromorphic mechanical")
        {
            return ( Plato::create_micromorphic_mechanics_problem<SpatialDim>(aMesh, tInputData, aMachine) );
        }
#ifdef PLATO_HELMHOLTZ
        else
        if(tLowerPhysics == "helmholtz filter")
        {
            auto tOutput = std::make_shared < Plato::Helmholtz::Problem<::Plato::HelmholtzFilter<SpatialDim>> > (aMesh, tInputData, aMachine);
            return tOutput;
        }
#endif
        else
        {
            ANALYZE_THROWERR(std::string("'Physics' of type ") + tLowerPhysics + "' is not supported.");
        }
        return nullptr;
    }
};
// class ProblemFactory

}
// namespace Plato

#endif /* PLATOPROBLEMFACTORY_HPP_ */
