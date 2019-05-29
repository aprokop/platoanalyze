#ifndef VOLUME_HPP
#define VOLUME_HPP

#include "plato/ApplyWeighting.hpp"
#include "plato/Simplex.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/AbstractScalarFunction.hpp"

#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"
#include "plato/ExpInstMacros.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, typename PenaltyFunctionType>
class Volume : public Plato::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;
    
    using Plato::AbstractScalarFunction<EvaluationType>::mMesh;
    using Plato::AbstractScalarFunction<EvaluationType>::m_dataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::Scalar mQuadratureWeight;
    Plato::Scalar mCellMaterialDensity;

    PenaltyFunctionType mPenaltyFunction;
    Plato::ApplyWeighting<SpaceDim,1,PenaltyFunctionType> mApplyWeighting;

  public:
    /**************************************************************************/
    Volume(Omega_h::Mesh& aMesh, 
           Omega_h::MeshSets& aMeshSets,
           Plato::DataMap& aDataMap, 
           Teuchos::ParameterList& aInputParams, 
           Teuchos::ParameterList& aPenaltyParams) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Volume"),
            mPenaltyFunction(aPenaltyParams),
            mApplyWeighting(mPenaltyFunction)
    /**************************************************************************/
    {
      mQuadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (Plato::OrdinalType tDimIndex=2; tDimIndex<=SpaceDim; tDimIndex++)
      { 
        mQuadratureWeight /= Plato::Scalar(tDimIndex);
      }

      auto tMaterialModelInputs = aInputParams.get<Teuchos::ParameterList>("Material Model");
      mCellMaterialDensity = tMaterialModelInputs.get<Plato::Scalar>("Density", 1.0);
    }

    /**************************************************************************
     * Unit testing constructor
    /**************************************************************************/
    Volume(Omega_h::Mesh& aMesh, 
           Omega_h::MeshSets& aMeshSets,
           Plato::DataMap& aDataMap) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Volume"),
            mPenaltyFunction(3.0, 0.0),
            mApplyWeighting(mPenaltyFunction)
    /**************************************************************************/
    {
      mQuadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (Plato::OrdinalType tDimIndex=2; tDimIndex<=SpaceDim; tDimIndex++)
      { 
        mQuadratureWeight /= Plato::Scalar(tDimIndex);
      }
    }

    /**************************************************************************/
    void setMaterialDensity(const Plato::Scalar aMaterialDensity)
    /**************************************************************************/
    {
      mCellMaterialDensity = aMaterialDensity;
    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> &,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto tNumCells = mMesh.nelems();

      Plato::ComputeCellVolume<SpaceDim> tComputeCellVolume;

      auto tMaterialDensity  = mCellMaterialDensity;
      auto tQuadratureWeight = mQuadratureWeight;
      auto tApplyWeighting  = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        ConfigScalarType tCellVolume;
        tComputeCellVolume(aCellOrdinal, aConfig, tCellVolume);
        tCellVolume *= tQuadratureWeight;

        aResult(aCellOrdinal) = tMaterialDensity * tCellVolume;

        // apply weighting
        //
        tApplyWeighting(aCellOrdinal, aResult, aControl);
    
      },"volume");
    }
};
// class Volume

} // namespace Plato

#endif
