#pragma once

#include "AnalyzeMacros.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "BLAS1.hpp"
#include "BLAS2.hpp"
#include "ToMap.hpp"
#include "Heaviside.hpp"
#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "Plato_Solve.hpp"
#include "ProjectToNode.hpp"
#include "RayleighStress.hpp"
#include "ApplyWeighting.hpp"
#include "ApplyConstraints.hpp"
#include "PlatoMathHelpers.hpp"
#include "InterpolateFromNodal.hpp"
#include "PlatoAbstractProblem.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/Newmark.hpp"
#include "hyperbolic/InertialContent.hpp"
#include "hyperbolic/HyperbolicVectorFunction.hpp"
#include "hyperbolic/SimplexMicromorphicMechanics.hpp"
#include "hyperbolic/MicromorphicElasticModelFactory.hpp"
#include "hyperbolic/MicromorphicInertiaModelFactory.hpp"
#include "hyperbolic/MicromorphicKinematics.hpp"
#include "hyperbolic/MicromorphicKinetics.hpp"
#include "hyperbolic/FullStressDivergence.hpp"
#include "hyperbolic/ProjectStressToNode.hpp"

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class RelaxedMicromorphicResidual :
  public Plato::SimplexMicromorphicMechanics<EvaluationType::SpatialDim>,
  public Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    using PhysicsType = typename Plato::SimplexMicromorphicMechanics<SpaceDim>;

    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using PhysicsType::mNumFullTerms;
    using PhysicsType::mNumVoigtTerms;
    using PhysicsType::mNumSkwTerms;
    using PhysicsType::mNumDofsPerCell;
    using PhysicsType::mNumDofsPerNode;

    using Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>::mSpatialDomain;
    using Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>::mDataMap;

    using StateScalarType       = typename EvaluationType::StateScalarType;
    using StateDotScalarType    = typename EvaluationType::StateDotScalarType;
    using StateDotDotScalarType = typename EvaluationType::StateDotDotScalarType;
    using ControlScalarType     = typename EvaluationType::ControlScalarType;
    using ConfigScalarType      = typename EvaluationType::ConfigScalarType;
    using ResultScalarType      = typename EvaluationType::ResultScalarType;

    using FunctionBaseType = Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>;
    using CubatureType = Plato::LinearTetCubRuleDegreeOne<SpaceDim>;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<SpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;
    Plato::ApplyWeighting<SpaceDim, SpaceDim,       IndicatorFunctionType> mApplyMassWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType, PhysicsType>> mBodyLoads;
    std::shared_ptr<CubatureType> mCubatureRule;
    std::shared_ptr<Plato::NaturalBCs<SpaceDim,mNumDofsPerNode>> mBoundaryLoads;

    bool mRayleighDamping;

    using MaterialType = Plato::MicromorphicLinearElasticMaterial<SpaceDim>;
    Teuchos::RCP<MaterialType> mMaterialModel;

    using InertiaMaterialType = Plato::MicromorphicInertiaMaterial<SpaceDim>;
    Teuchos::RCP<InertiaMaterialType> mInertiaModel;

    std::vector<std::string> mPlottable;

  public:
    /**************************************************************************/
    RelaxedMicromorphicResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap,
                               {"Displacement X", "Displacement Y", "Displacement Z",
                               "Micro Distortion XX", "Micro Distortion YX", "Micro Distortion ZX",
                               "Micro Distortion XY", "Micro Distortion YY", "Micro Distortion ZY",
                               "Micro Distortion XZ", "Micro Distortion YZ", "Micro Distortion ZZ"},
                               {"Velocity X", "Velocity Y", "Velocity Z",
                               "Micro Velocity XX", "Micro Velocity YX", "Micro Velocity ZX",
                               "Micro Velocity XY", "Micro Velocity YY", "Micro Velocity ZY",
                               "Micro Velocity XZ", "Micro Velocity YZ", "Micro Velocity ZZ"},
                               {"Acceleration X", "Acceleration Y", "Acceleration Z",
                               "Micro Acceleration XX", "Micro Acceleration YX", "Micro Acceleration ZX",
                               "Micro Acceleration XY", "Micro Acceleration YY", "Micro Acceleration ZY",
                               "Micro Acceleration XZ", "Micro Acceleration YZ", "Micro Acceleration ZZ"}),
        mIndicatorFunction    (aPenaltyParams),
        mApplyStressWeighting (mIndicatorFunction),
        mApplyMassWeighting   (mIndicatorFunction),
        mBodyLoads            (nullptr),
        mCubatureRule         (std::make_shared<CubatureType>()),
        mBoundaryLoads        (nullptr)
    /**************************************************************************/
    {
        // check that explicit a-form solver is used
        this->checkTimeIntegrator(aProblemParams.sublist("Time Integration"));

        // create material model and get stiffness
        //
        Plato::MicromorphicElasticModelFactory<SpaceDim> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());

        Plato::MicromorphicInertiaModelFactory<SpaceDim> tInertiaModelFactory(aProblemParams);
        mInertiaModel = tInertiaModelFactory.create(aSpatialDomain.getMaterialName());

        mRayleighDamping = (mInertiaModel->getRayleighA() != 0.0)
                        || (mMaterialModel->getRayleighB() != 0.0);

        // parse body loads
        //
        if(aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType, PhysicsType>>
                         (aProblemParams.sublist("Body Loads"));
        }

        // parse boundary Conditions
        //
        if(aProblemParams.isSublist("Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim,mNumDofsPerNode>>
                             (aProblemParams.sublist("Natural Boundary Conditions"));
        }

        auto tResidualParams = aProblemParams.sublist("Hyperbolic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
            mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }
    }

    /**************************************************************************//**
    * \brief return the maximum eigenvalue of the gradient wrt state
    ******************************************************************************/
    void
    checkTimeIntegrator(Teuchos::ParameterList & aIntegratorParams)
    {
        if (aIntegratorParams.isType<bool>("A-Form"))
        {
            auto tAForm = aIntegratorParams.get<bool>("A-Form");
            auto tBeta = aIntegratorParams.get<double>("Newmark Beta");
            if (tAForm == false)
            {
                ANALYZE_THROWERR("In RelaxedMicromorphicResidual constructor: Newmark A-Form must be specified for micromorphic mechanics")
            }
            else if (tBeta != 0.0)
            {
                ANALYZE_THROWERR("In RelaxedMicromorphicResidual constructor: Newmark explicit (beta=0, gamma=0.5) must be specified for micromorphic mechanics")
            }
        }
        else
        {
            ANALYZE_THROWERR("In RelaxedMicromorphicResidual constructor: Newmark A-Form must be specified for micromorphic mechanics")
        }
    }

    /**************************************************************************//**
    * \brief return the maximum eigenvalue of the gradient wrt state
    ******************************************************************************/
    Plato::Scalar
    getMaxEigenvalue(
        const Plato::ScalarArray3D & aConfig
    ) const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        Plato::ScalarVector tCellVolume("cell weight", tNumCells);

        Plato::ComputeCellVolume<SpaceDim> tComputeVolume;

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            Plato::Scalar tThisVolume;
            tComputeVolume(aCellOrdinal, aConfig, tThisVolume);
            tCellVolume(aCellOrdinal) = tThisVolume;
        }, "compute volume");

        Plato::Scalar tMinVolume;
        Plato::blas1::min(tCellVolume, tMinVolume);
        Plato::Scalar tLength = pow(tMinVolume, 1.0/SpaceDim);

        auto tStiffnessMatrixCe = mMaterialModel->getStiffnessMatrixCe();
        auto tMassDensity     = mInertiaModel->getMassDensity();
        auto tSoundSpeed = sqrt(tStiffnessMatrixCe(0,0)/tMassDensity);

        return 2.0*tSoundSpeed/tLength;
    }

    /**************************************************************************//**
    *
    * \brief Call the output state function in the residual
    * 
    * \param [in] aSolutions State solutions database
    * \return output solutions database
    * 
    ******************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const override
    {
      return aSolutions;
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep = 0.0,
              Plato::Scalar aCurrentTime = 0.0
    ) const override
    /**************************************************************************/
    {
        if ( mRayleighDamping )
        {
             evaluateWithDamping(aState, aStateDot, aStateDotDot, aControl, aConfig, aResult, aTimeStep, aCurrentTime);
        }
        else
        {
             evaluateWithoutDamping(aState, aStateDot, aStateDotDot, aControl, aConfig, aResult, aTimeStep, aCurrentTime);
        }
    }

    /**************************************************************************/
    void
    evaluateWithoutDamping(
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep = 0.0,
              Plato::Scalar aCurrentTime = 0.0
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using StrainScalarType =
          typename Plato::fad_type_t<Plato::SimplexMicromorphicMechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      using InertiaStrainScalarType =
          typename Plato::fad_type_t<Plato::SimplexMicromorphicMechanics<EvaluationType::SpatialDim>, StateDotDotScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<SpaceDim> tComputeGradient;
      Plato::MicromorphicKinematics<SpaceDim> tKinematics;
      Plato::MicromorphicKinetics<SpaceDim>   tKinetics(mMaterialModel);
      Plato::MicromorphicKinetics<SpaceDim>   tInertiaKinetics(mInertiaModel);
      Plato::FullStressDivergence<SpaceDim>   tComputeStressDivergence;
      Plato::InertialContent<SpaceDim, InertiaMaterialType> tInertialContent(mInertiaModel);
      Plato::ProjectToNode<SpaceDim, mNumDofsPerNode> tProjectInertialContent;
      Plato::ProjectStressToNode<SpaceDim,SpaceDim> tComputeStressForMicromorphicResidual;
      Plato::InterpolateFromNodal<SpaceDim, mNumDofsPerNode, /*offset=*/0, SpaceDim> tInterpolateFromNodal;

      Plato::ScalarArray3DT<ConfigScalarType>
        tGradient("gradient",tNumCells,mNumNodesPerCell,SpaceDim);

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType> 
        tSymDisplacementGradient("strain",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<StrainScalarType> 
        tSkwDisplacementGradient("strain",tNumCells,mNumSkwTerms);

      Plato::ScalarMultiVectorT<StateScalarType> 
        tSymMicroDistortionTensor("strain",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<StateScalarType> 
        tSkwMicroDistortionTensor("strain",tNumCells,mNumSkwTerms);

      Plato::ScalarMultiVectorT<InertiaStrainScalarType> 
        tSymGradientMicroInertia("inertia strain",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<InertiaStrainScalarType> 
        tSkwGradientMicroInertia("inertia strain",tNumCells,mNumSkwTerms);

      Plato::ScalarMultiVectorT<StateDotDotScalarType> 
        tSymFreeMicroInertia("inertia strain",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<StateDotDotScalarType> 
        tSkwFreeMicroInertia("inertia strain",tNumCells,mNumSkwTerms);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tSymCauchyStress("stress",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tSkwCauchyStress("stress",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tSymMicroStress("stress",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tSymGradientInertiaStress("inertia stress",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tSkwGradientInertiaStress("inertia stress",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tSymFreeInertiaStress("inertia stress",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tSkwFreeInertiaStress("inertia stress",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<StateDotDotScalarType>
        tAccelerationGP("acceleration at Gauss point", tNumCells, SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tInertialContentGP("Inertial content at Gauss point", tNumCells, SpaceDim);

      auto tBasisFunctions = mCubatureRule->getBasisFunctions();

      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto& tApplyStressWeighting = mApplyStressWeighting;
      auto& tApplyMassWeighting = mApplyMassWeighting;

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;

        // kinematics
        tKinematics(aCellOrdinal, tSymDisplacementGradient, tSkwDisplacementGradient, tSymMicroDistortionTensor, tSkwMicroDistortionTensor, aState, tBasisFunctions, tGradient);
        tKinematics(aCellOrdinal, tSymGradientMicroInertia, tSkwGradientMicroInertia, tSymFreeMicroInertia, tSkwFreeMicroInertia, aStateDotDot, tBasisFunctions, tGradient);

        // kinetics
        tKinetics(aCellOrdinal, tSymCauchyStress, tSkwCauchyStress, tSymMicroStress, tSymDisplacementGradient, tSkwDisplacementGradient, tSymMicroDistortionTensor, tSkwMicroDistortionTensor);
        tInertiaKinetics(aCellOrdinal, tSymGradientInertiaStress, tSkwGradientInertiaStress, tSymFreeInertiaStress, tSkwFreeInertiaStress, tSymGradientMicroInertia, tSkwGradientMicroInertia, tSymFreeMicroInertia, tSkwFreeMicroInertia);

        // apply stress weighting
        tApplyStressWeighting(aCellOrdinal, tSymCauchyStress, aControl);
        tApplyStressWeighting(aCellOrdinal, tSkwCauchyStress, aControl);
        tApplyStressWeighting(aCellOrdinal, tSymMicroStress, aControl);
        tApplyStressWeighting(aCellOrdinal, tSymGradientInertiaStress, aControl);
        tApplyStressWeighting(aCellOrdinal, tSkwGradientInertiaStress, aControl);
        tApplyStressWeighting(aCellOrdinal, tSymFreeInertiaStress, aControl);
        tApplyStressWeighting(aCellOrdinal, tSkwFreeInertiaStress, aControl);

        // compute stress components of residual
        tComputeStressDivergence(aCellOrdinal, aResult, tSymCauchyStress, tSkwCauchyStress, tGradient, tCellVolume);
        tComputeStressForMicromorphicResidual(aCellOrdinal, aResult, tSymCauchyStress, tSkwCauchyStress, tSymMicroStress, tBasisFunctions, tCellVolume);

        tComputeStressDivergence(aCellOrdinal, aResult, tSymGradientInertiaStress, tSkwGradientInertiaStress, tGradient, tCellVolume);
        tComputeStressForMicromorphicResidual(aCellOrdinal, aResult, tSymFreeInertiaStress, tSkwFreeInertiaStress, tBasisFunctions, tCellVolume);

        // compute displacement accelerations at gausspoints
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, aStateDotDot, tAccelerationGP);

        // compute macro inertia at gausspoints
        tInertialContent(aCellOrdinal, tInertialContentGP, tAccelerationGP);

        // apply inertia weighting
        tApplyMassWeighting(aCellOrdinal, tInertialContentGP, aControl);

        // project to nodes
        tProjectInertialContent(aCellOrdinal, tCellVolume, tBasisFunctions, tInertialContentGP, aResult);

      }, "Compute Residual");

     if( std::count(mPlottable.begin(),mPlottable.end(),"stress") ) toMap(mDataMap, tSymCauchyStress, "stress", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"strain") ) toMap(mDataMap, tSymDisplacementGradient, "strain", mSpatialDomain);

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aState, aControl, aResult, -1.0 );
      }
    }
    /**************************************************************************/
    void
    evaluateWithDamping(
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep = 0.0,
              Plato::Scalar aCurrentTime = 0.0
    ) const
    /**************************************************************************/
    {
        ANALYZE_THROWERR("Relaxed Micromorphic residual does not support damping currently.")
    }
    /**************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                                & aSpatialModel,
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep = 0.0,
              Plato::Scalar aCurrentTime = 0.0
    ) const override
    /**************************************************************************/
    {
        if( mBoundaryLoads != nullptr )
        {
            mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0, aCurrentTime );
        }
    }
};

