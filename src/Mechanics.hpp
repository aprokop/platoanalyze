#ifndef PLATO_MECHANICS_HPP
#define PLATO_MECHANICS_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/Plato_AugLagStressCriterionQuadratic.hpp"
#include "plato/Plato_AugLagStressCriterionGeneral.hpp"
#include "plato/Plato_AugLagStressCriterion.hpp"
#include "plato/SimplexMechanics.hpp"
#include "plato/AbstractScalarFunctionInc.hpp"
#include "plato/ElastostaticResidual.hpp"
#include "plato/InternalElasticEnergy.hpp"
#include "plato/EffectiveEnergy.hpp"
#include "plato/Volume.hpp"
#include "plato/StressPNorm.hpp"
#include "plato/IntermediateDensityPenalty.hpp"
#include "plato/AnalyzeMacros.hpp"

#include "plato/AbstractLocalMeasure.hpp"
#include "plato/VonMisesLocalMeasure.hpp"
#include "plato/TensileEnergyDensityLocalMeasure.hpp"

#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"

namespace Plato
{

namespace MechanicsFactory
{

  /******************************************************************************//**
   * @brief Create a local measure for use in augmented lagrangian quadratic
   * @param [in] aInputParams input parameters
   * @param [in] aFuncName scalar function name
  **********************************************************************************/
  template <typename EvaluationType>
  inline std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType,Plato::SimplexMechanics<EvaluationType::SpatialDim>>> 
  create_local_measure(Teuchos::ParameterList& aInputParams, const std::string & aFuncName)
  {
      auto tFunctionSpecs = aInputParams.sublist(aFuncName);
      auto tLocalMeasure = tFunctionSpecs.get<std::string>("Local Measure", "VonMises");

      if(tLocalMeasure == "VonMises")
      {
          return std::make_shared<VonMisesLocalMeasure<EvaluationType, Plato::SimplexMechanics<EvaluationType::SpatialDim>>>(aInputParams, "VonMises");
      }
      else if(tLocalMeasure == "TensileEnergyDensity")
      {
          return std::make_shared<TensileEnergyDensityLocalMeasure<EvaluationType, Plato::SimplexMechanics<EvaluationType::SpatialDim>>>
                                                           (aInputParams, "TensileEnergyDensity");
      }
      else
      {
          THROWERR("Unknown 'Local Measure' specified in 'Plato Problem' ParameterList")
      }
  }

/******************************************************************************//**
 * @brief Create elastostatics residual equation
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractVectorFunction<EvaluationType>>
elastostatics_residual(Omega_h::Mesh& aMesh,
                       Omega_h::MeshSets& aMeshSets,
                       Plato::DataMap& aDataMap,
                       Teuchos::ParameterList& aInputParams,
                       std::string aFuncName)
{
    std::shared_ptr<AbstractVectorFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aInputParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::ElastostaticResidual<EvaluationType, Plato::MSIMP>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::ElastostaticResidual<EvaluationType, Plato::RAMP>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::ElastostaticResidual<EvaluationType, Plato::Heaviside>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    return (tOutput);
}
// function elastostatics_residual

/******************************************************************************//**
 * @brief Create augmented Lagrangian stress constraint criterion tailored for linear problems
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
stress_constraint_linear(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap,
                         Teuchos::ParameterList & aInputParams,
                         std::string & aFuncName)
{
    std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>> tOutput;
    tOutput = std::make_shared< Plato::AugLagStressCriterion<EvaluationType> >
                (aMesh, aMeshSets, aDataMap, aInputParams, aFuncName);
    return (tOutput);
}

/******************************************************************************//**
 * @brief Create augmented Lagrangian stress constraint criterion tailored for general problems
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
stress_constraint_general(Omega_h::Mesh& aMesh,
                          Omega_h::MeshSets& aMeshSets,
                          Plato::DataMap& aDataMap,
                          Teuchos::ParameterList & aInputParams,
                          std::string & aFuncName)
{
    std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>> tOutput;
    tOutput = std::make_shared <Plato::AugLagStressCriterionGeneral<EvaluationType> >
                (aMesh, aMeshSets, aDataMap, aInputParams, aFuncName);
    return (tOutput);
}


/******************************************************************************//**
 * @brief Create augmented Lagrangian local constraint criterion with quadratic constraint formulation
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
stress_constraint_quadratic(Omega_h::Mesh& aMesh,
                            Omega_h::MeshSets& aMeshSets,
                            Plato::DataMap& aDataMap,
                            Teuchos::ParameterList & aInputParams,
                            std::string & aFuncName)
{
    auto EvalMeasure = Plato::MechanicsFactory::create_local_measure<EvaluationType>(aInputParams, aFuncName);
    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<EvaluationType::SpatialDim>>;
    auto PODMeasure = Plato::MechanicsFactory::create_local_measure<Residual>(aInputParams, aFuncName);

    using SimplexT = Plato::SimplexMechanics<EvaluationType::SpatialDim>;
    std::shared_ptr<Plato::AugLagStressCriterionQuadratic<EvaluationType,SimplexT>> tOutput;
    tOutput = std::make_shared< Plato::AugLagStressCriterionQuadratic<EvaluationType,SimplexT> >
                (aMesh, aMeshSets, aDataMap, aInputParams, aFuncName);

    tOutput->setLocalMeasure(EvalMeasure, PODMeasure);
    return (tOutput);
}


/******************************************************************************//**
 * @brief Create internal elastic energy criterion
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
 * @param [in] aFuncType vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
internal_elastic_energy(Omega_h::Mesh& aMesh,
                        Omega_h::MeshSets& aMeshSets,
                        Plato::DataMap& aDataMap,
                        Teuchos::ParameterList & aInputParams,
                        std::string & aFuncName)
{
    std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aInputParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::InternalElasticEnergy<EvaluationType, Plato::MSIMP>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::InternalElasticEnergy<EvaluationType, Plato::RAMP>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::InternalElasticEnergy<EvaluationType, Plato::Heaviside>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    return (tOutput);
}
// function internal_elastic_energy

/******************************************************************************//**
 * @brief Create stress p-norm criterion
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
 * @param [in] aFuncType vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
stress_p_norm(Omega_h::Mesh& aMesh,
              Omega_h::MeshSets& aMeshSets,
              Plato::DataMap& aDataMap,
              Teuchos::ParameterList & aInputParams,
              std::string & aFuncName)
{
    std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aInputParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::StressPNorm<EvaluationType, Plato::MSIMP>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::StressPNorm<EvaluationType, Plato::RAMP>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::StressPNorm<EvaluationType, Plato::Heaviside>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    return (tOutput);
}
// function stress_p_norm

/******************************************************************************//**
 * @brief Create effective energy criterion
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
 * @param [in] aFuncType vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
effective_energy(Omega_h::Mesh& aMesh,
                 Omega_h::MeshSets& aMeshSets,
                 Plato::DataMap& aDataMap,
                 Teuchos::ParameterList & aInputParams,
                 std::string & aFuncName)
{
    std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aInputParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::EffectiveEnergy<EvaluationType, Plato::MSIMP>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::EffectiveEnergy<EvaluationType, Plato::RAMP>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::EffectiveEnergy<EvaluationType, Plato::Heaviside>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    return (tOutput);
}
// function effective_energy

/******************************************************************************//**
 * @brief Create volume criterion
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
 * @param [in] aFuncType vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
volume(Omega_h::Mesh& aMesh,
       Omega_h::MeshSets& aMeshSets,
       Plato::DataMap& aDataMap,
       Teuchos::ParameterList & aInputParams,
       std::string & aFuncName)
{
    std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aInputParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::Volume<EvaluationType, Plato::MSIMP>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::Volume<EvaluationType, Plato::RAMP>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::Volume<EvaluationType, Plato::Heaviside>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    return (tOutput);
}
// function volume

/******************************************************************************//**
 * @brief Factory for linear mechanics problem
**********************************************************************************/
struct FunctionFactory
{
    /******************************************************************************//**
     * @brief Create a PLATO vector function (i.e. residual equation)
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Analyze physics-based database
     * @param [in] aInputParams input parameters
     * @param [in] aFuncName vector function name
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(Omega_h::Mesh& aMesh, 
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap, 
                         Teuchos::ParameterList& aInputParams,
                         std::string aFuncName)
    {

        if(aFuncName == "Elliptic")
        {
            return (Plato::MechanicsFactory::elastostatics_residual<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
        else
        {
            THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList")
        }
    }


    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<AbstractScalarFunctionInc<EvaluationType>>
    createScalarFunctionInc(Omega_h::Mesh& aMesh,
                            Omega_h::MeshSets& aMeshSets,
                            Plato::DataMap& aDataMap,
                            Teuchos::ParameterList& aParamList,
                            std::string strScalarFunctionType,
                            std::string aStrScalarFunctionName )
    /******************************************************************************/
    {
        THROWERR("Not yet implemented")
    }

    /******************************************************************************//**
     * @brief Create a PLATO scalar function (i.e. optimization criterion)
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Analyze physics-based database
     * @param [in] aInputParams input parameters
     * @param [in] aFuncName scalar function name
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
    createScalarFunction(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap, 
                         Teuchos::ParameterList & aInputParams,
                         std::string aFuncType,
                         std::string aFuncName)
    {
        if(aFuncType == "Internal Elastic Energy")
        {
            return (Plato::MechanicsFactory::internal_elastic_energy<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
        else if(aFuncType == "Stress P-Norm")
        {
            return (Plato::MechanicsFactory::stress_p_norm<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
        else if(aFuncType == "Effective Energy")
        {
            return (Plato::MechanicsFactory::effective_energy<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
        else if(aFuncType == "Stress Constraint")
        {
            return (Plato::MechanicsFactory::stress_constraint_linear<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
        else if(aFuncType == "Stress Constraint General")
        {
            return (Plato::MechanicsFactory::stress_constraint_general<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
        else if(aFuncType == "Stress Constraint Quadratic")
        {
            return (Plato::MechanicsFactory::stress_constraint_quadratic<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
        else if(aFuncType == "Volume")
        {
            return (Plato::MechanicsFactory::volume<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
        else if(aFuncType == "Density Penalty")
        {
            return std::make_shared<Plato::IntermediateDensityPenalty<EvaluationType>>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName);
        }
        else
        {
            THROWERR("Unknown 'Objective' specified in 'Plato Problem' ParameterList")
        }
    }

    /******************************************************************************//**
     * @brief Create a local measure for use in augmented lagrangian quadratic
     * @param [in] aInputParams input parameters
     * @param [in] aFuncName scalar function name
    **********************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType,Plato::SimplexMechanics<EvaluationType::SpatialDim>>> 
    createLocalMeasure(Teuchos::ParameterList& aInputParams, const std::string & aFuncName)
    {
        auto tFunctionSpecs = aInputParams.sublist(aFuncName);
        auto tLocalMeasure = tFunctionSpecs.get<std::string>("Local Measure", "VonMises");

        if(tLocalMeasure == "VonMises")
        {
            return std::make_shared<VonMisesLocalMeasure<EvaluationType, Plato::SimplexMechanics<EvaluationType::SpatialDim>>>(aInputParams, "VonMises");
        }
        else if(tLocalMeasure == "TensileEnergyDensity")
        {
            return std::make_shared<TensileEnergyDensityLocalMeasure<EvaluationType, Plato::SimplexMechanics<EvaluationType::SpatialDim>>>
                                                             (aInputParams, "TensileEnergyDensity");
        }
        else
        {
            THROWERR("Unknown 'Local Measure' specified in 'Plato Problem' ParameterList")
        }
    }
};
// struct FunctionFactory

} // namespace MechanicsFactory

/******************************************************************************//**
 * @brief Concrete class for use as the SimplexPhysics template argument in
 *        EllipticProblem
**********************************************************************************/
template<Plato::OrdinalType SpaceDimParam>
class Mechanics: public Plato::SimplexMechanics<SpaceDimParam>
{
public:
    typedef Plato::MechanicsFactory::FunctionFactory FunctionFactory;
    using SimplexT = SimplexMechanics<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};
} // namespace Plato

#endif
