#ifndef HYPERBOLIC_ELECTROELASTOMECHANICS_RESIDUAL_HPP
#define HYPERBOLIC_ELECTROELASTOMECHANICS_RESIDUAL_HPP

#include "Simp.hpp"
#include "Ramp.hpp"
#include "BLAS1.hpp"
#include "BLAS2.hpp"
#include "ToMap.hpp"
#include "Strain.hpp"
#include "Heaviside.hpp"
#include "BodyLoads.hpp"
#include "EMKinetics.hpp"
#include "NaturalBCs.hpp"
#include "Plato_Solve.hpp"
#include "EMKinematics.hpp"
#include "ProjectToNode.hpp"
#include "RayleighStress.hpp"
#include "ApplyWeighting.hpp"
#include "FluxDivergence.hpp"
#include "StressDivergence.hpp"
#include "ApplyConstraints.hpp"
#include "PlatoMathHelpers.hpp"
#include "InterpolateFromNodal.hpp"
#include "PlatoAbstractProblem.hpp"
#include "SimplexElectromechanics.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "LinearElectroelasticMaterial.hpp"

#include "hyperbolic/Newmark.hpp"
#include "hyperbolic/InertialContent.hpp"
#include "hyperbolic/HyperbolicVectorFunction.hpp"
#include "hyperbolic/HyperbolicLinearStress.hpp"

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class TransientElectromechanicsResidual :
  public Plato::SimplexElectromechanics<EvaluationType::SpatialDim>,
  public Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;

    static constexpr Plato::OrdinalType NElecDims = 1;
    static constexpr Plato::OrdinalType NMechDims = mSpaceDim;

    static constexpr Plato::OrdinalType EDofOffset = mSpaceDim;
    static constexpr Plato::OrdinalType MDofOffset = 0;

    using PhysicsType = typename Plato::SimplexElectromechanics<mSpaceDim>;

    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;
    using PhysicsType::mNumVoigtTerms;
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
    using CubatureType = Plato::LinearTetCubRuleDegreeOne<mSpaceDim>;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mSpaceDim, mSpaceDim,      IndicatorFunctionType> mApplyEDispWeighting;
    Plato::ApplyWeighting<mSpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;
    Plato::ApplyWeighting<mSpaceDim, mSpaceDim,      IndicatorFunctionType> mApplyMassWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType, PhysicsType>> mBodyLoads;
    std::shared_ptr<CubatureType> mCubatureRule;

    std::shared_ptr<Plato::NaturalBCs<mSpaceDim, NMechDims, mNumDofsPerNode, MDofOffset>> mBoundaryLoads;
    std::shared_ptr<Plato::NaturalBCs<mSpaceDim, NElecDims, mNumDofsPerNode, EDofOffset>> mBoundaryCharges;

    bool mRayleighDamping;

    using MaterialType = Plato::LinearElectroelasticMaterial<mSpaceDim>;
    Teuchos::RCP<MaterialType> mMaterialModel;

    std::vector<std::string> mPlottable;

  public:
    /**************************************************************************/
    TransientElectromechanicsResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap,
                               {"displacement X", "displacement Y", "displacement Z", "electric potential"},
                               {"velocity X",     "velocity Y",     "velocity Z",     "electric potential dot"},
                               {"acceleration X", "acceleration Y", "acceleration Z", "electric potential dot dot"}),
        mIndicatorFunction    (aPenaltyParams),
        mApplyEDispWeighting  (mIndicatorFunction),
        mApplyStressWeighting (mIndicatorFunction),
        mApplyMassWeighting   (mIndicatorFunction),
        mBodyLoads            (nullptr),
        mCubatureRule         (std::make_shared<CubatureType>()),
        mBoundaryLoads        (nullptr)
    /**************************************************************************/
    {
        // create material model and get stiffness
        //
        Plato::ElectroelasticModelFactory<mSpaceDim> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());

        mRayleighDamping = false;
//        mRayleighDamping = (mMaterialModel->getRayleighA() != 0.0)
//                        || (mMaterialModel->getRayleighB() != 0.0);

        // parse body loads
        //
        if(aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType, PhysicsType>>
                         (aProblemParams.sublist("Body Loads"));
        }

        // parse mechanical boundary Conditions
        //
        if(aProblemParams.isSublist("Mechanical Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<mSpaceDim, NMechDims, mNumDofsPerNode, MDofOffset>>
                             (aProblemParams.sublist("Mechanical Natural Boundary Conditions"));
        }

        // parse electrical boundary Conditions
        // 
        if(aProblemParams.isSublist("Electrical Natural Boundary Conditions"))
        {
            mBoundaryCharges = std::make_shared<Plato::NaturalBCs<mSpaceDim, NElecDims, mNumDofsPerNode, EDofOffset>>
                (aProblemParams.sublist("Electrical Natural Boundary Conditions"));
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
    Plato::Scalar
    getMaxEigenvalue(
        const Plato::ScalarArray3D & aConfig
    ) const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        Plato::ScalarVector tCellVolume("cell weight", tNumCells);

        Plato::ComputeCellVolume<mSpaceDim> tComputeVolume;

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            Plato::Scalar tThisVolume;
            tComputeVolume(aCellOrdinal, aConfig, tThisVolume);
            tCellVolume(aCellOrdinal) = tThisVolume;
        }, "compute volume");

        Plato::Scalar tMinVolume;
        Plato::blas1::min(tCellVolume, tMinVolume);
        Plato::Scalar tLength = pow(tMinVolume, 1.0/mSpaceDim);

        auto tStiffnessMatrix = mMaterialModel->getStiffnessMatrix();
        auto tMassDensity     = mMaterialModel->getMassDensity();
        auto tSoundSpeed = sqrt(tStiffnessMatrix(0,0)/tMassDensity);

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
      using SimplexPhysics = typename Plato::SimplexElectromechanics<mSpaceDim>;
      using GradScalarType =
          typename Plato::fad_type_t<SimplexPhysics, StateScalarType, ConfigScalarType>;

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;

      Plato::EMKinematics<mSpaceDim> tKinematics;
      Plato::EMKinetics<mSpaceDim>   tKinetics(mMaterialModel);

      Plato::StressDivergence<mSpaceDim, mNumDofsPerNode, MDofOffset> tComputeStressDivergence;
      Plato::FluxDivergence  <mSpaceDim, mNumDofsPerNode, EDofOffset> tComputeEdispDivergence;

      Plato::InertialContent<mSpaceDim, MaterialType> tInertialContent(mMaterialModel);

      Plato::ProjectToNode<mSpaceDim, mNumDofsPerNode> tProjectInertialContent;

      Plato::InterpolateFromNodal<mSpaceDim, mNumDofsPerNode, MDofOffset, NMechDims> tInterpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType> tCellVolume("cell weight", tNumCells);

      Plato::ScalarArray3DT<ConfigScalarType> tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

      Plato::ScalarMultiVectorT<GradScalarType> tStrain("strain", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType> tEfield("efield", tNumCells, mSpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType> tStress("stress", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> tEDisp ("edisp",  tNumCells, mSpaceDim);

      Plato::ScalarMultiVectorT<StateDotDotScalarType>
        tAccelerationGP("acceleration at Gauss point", tNumCells, mSpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tInertialContentGP("Inertial content at Gauss point", tNumCells, mSpaceDim);

      auto tBasisFunctions = mCubatureRule->getBasisFunctions();

      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto& tApplyStressWeighting = mApplyStressWeighting;
      auto& tApplyEDispWeighting = mApplyEDispWeighting;
      auto& tApplyMassWeighting = mApplyMassWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;

        // compute strain and electric field
        tKinematics(aCellOrdinal, tStrain, tEfield, aState, tGradient);

        // compute stress and electric displacement
        tKinetics(aCellOrdinal, tStress, tEDisp, tStrain, tEfield);

        // apply weighting
        tApplyStressWeighting(aCellOrdinal, tStress, aControl);
        tApplyEDispWeighting (aCellOrdinal, tEDisp,  aControl);

        // compute stress divergence
        tComputeStressDivergence(aCellOrdinal, aResult, tStress, tGradient, tCellVolume);
        tComputeEdispDivergence (aCellOrdinal, aResult, tEDisp,  tGradient, tCellVolume);

        // compute accelerations at gausspoints
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, aStateDotDot, tAccelerationGP);

        // compute inertia at gausspoints
        tInertialContent(aCellOrdinal, tInertialContentGP, tAccelerationGP);

        // apply weighting
        tApplyMassWeighting(aCellOrdinal, tInertialContentGP, aControl);

        // project to nodes
        tProjectInertialContent(aCellOrdinal, tCellVolume, tBasisFunctions, tInertialContentGP, aResult);

      }, "Compute Residual");

     if( std::count(mPlottable.begin(),mPlottable.end(),"stress") ) toMap(mDataMap, tStress, "stress", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"strain") ) toMap(mDataMap, tStress, "strain", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"efield") ) toMap(mDataMap, tEfield, "efield", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"edisp" ) ) toMap(mDataMap, tEDisp,  "edisp" , mSpatialDomain);

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
#ifdef NOPE
      using SimplexPhysics = typename Plato::SimplexMechanics<mSpaceDim>;
      using StrainScalarType =
          typename Plato::fad_type_t<SimplexPhysics, StateScalarType, ConfigScalarType>;
      using VelGradScalarType =
          typename Plato::fad_type_t<SimplexPhysics, StateDotScalarType, ConfigScalarType>;
      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
      Plato::Strain<mSpaceDim>                 tComputeVoigtStrain;
      Plato::RayleighStress<mSpaceDim>         tComputeVoigtStress(mMaterialModel);
      Plato::StressDivergence<mSpaceDim>       tComputeStressDivergence;
      Plato::InertialContent<mSpaceDim>        tInertialContent(mMaterialModel);

      Plato::ProjectToNode<mSpaceDim, mNumDofsPerNode>        tProjectInertialContent;

      Plato::InterpolateFromNodal<mSpaceDim, mNumDofsPerNode, /*offset=*/0, mSpaceDim> tInterpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType>
        tStrain("strain",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<VelGradScalarType>
        tVelGrad("velocity gradient", tNumCells, mNumVoigtTerms);

      Plato::ScalarArray3DT<ConfigScalarType>
        tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tStress("stress", tNumCells, mNumVoigtTerms);

      Plato::ScalarMultiVectorT<StateDotDotScalarType>
        tAccelerationGP("acceleration at Gauss point", tNumCells, mSpaceDim);

      Plato::ScalarMultiVectorT<StateDotScalarType>
        tVelocityGP("velocity at Gauss point", tNumCells, mSpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tInertialContentGP("Inertial content at Gauss point", tNumCells, mSpaceDim);

      auto tBasisFunctions = mCubatureRule->getBasisFunctions();

      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto& tApplyStressWeighting = mApplyStressWeighting;
      auto& tApplyMassWeighting = mApplyMassWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;

        // compute strain
        tComputeVoigtStrain(aCellOrdinal, tStrain, aState, tGradient);

        // compute velocity gradient
        tComputeVoigtStrain(aCellOrdinal, tVelGrad, aStateDot, tGradient);

        // compute stress
        tComputeVoigtStress(aCellOrdinal, tStress, tStrain, tVelGrad);

        // apply weighting
        tApplyStressWeighting(aCellOrdinal, tStress, aControl);

        // compute stress divergence
        tComputeStressDivergence(aCellOrdinal, aResult, tStress, tGradient, tCellVolume);

        // compute accelerations at gausspoints
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, aStateDotDot, tAccelerationGP);

        // compute velocities at gausspoints
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, aStateDot, tVelocityGP);

        // compute inertia at gausspoints
        tInertialContent(aCellOrdinal, tInertialContentGP, tVelocityGP, tAccelerationGP);

        // apply weighting
        tApplyMassWeighting(aCellOrdinal, tInertialContentGP, aControl);

        // project to nodes
        tProjectInertialContent(aCellOrdinal, tCellVolume, tBasisFunctions, tInertialContentGP, aResult);

      }, "Compute Residual");

     if( std::count(mPlottable.begin(),mPlottable.end(),"stress") ) toMap(mDataMap, tStress, "stress", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"strain") ) toMap(mDataMap, tStress, "strain", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"velgrad") ) toMap(mDataMap, tStress, "velgrad", mSpatialDomain);

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aState, aControl, aResult, -1.0 );
      }
#endif
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

        if( mBoundaryCharges != nullptr )
        {
            mBoundaryCharges->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0, aCurrentTime );
        }
    }
};

#endif
