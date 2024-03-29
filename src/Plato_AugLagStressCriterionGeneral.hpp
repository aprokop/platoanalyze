/*
 * Plato_AugLagStressCriterionGeneral.hpp
 *
 *  Created on: Feb 12, 2019
 */

#pragma once

#include <algorithm>

#include "ImplicitFunctors.hpp"
#include "ElasticModelFactory.hpp"

#include "Simp.hpp"
#include "BLAS1.hpp"
#include "ToMap.hpp"
#include "Strain.hpp"
#include "WorksetBase.hpp"
#include "LinearStress.hpp"
#include "SimplexFadTypes.hpp"
#include "PlatoMathHelpers.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "VonMisesYieldFunction.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Augmented Lagrangian stress constraint criterion tailored for general problems
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class AugLagStressCriterionGeneral :
        public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
        public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */
    static constexpr Plato::OrdinalType mNumVoigtTerms = Plato::SimplexMechanics<mSpaceDim>::mNumVoigtTerms; /*!< number of Voigt terms */
    static constexpr Plato::OrdinalType mNumNodesPerCell = Plato::SimplexMechanics<mSpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell/element */

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain; /*!< mesh database */
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap; /*!< PLATO Engine output database */

    using StateT = typename EvaluationType::StateScalarType; /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */
    using ControlT = typename EvaluationType::ControlScalarType; /*!< control variables automatic differentiation type */

    Plato::Scalar mPenalty; /*!< penalty parameter in SIMP model */
    Plato::Scalar mStressLimit; /*!< stress limit/upper bound */
    Plato::Scalar mAugLagPenalty; /*!< augmented Lagrangian penalty */
    Plato::Scalar mMinErsatzValue; /*!< minimum ersatz material value in SIMP model */
    Plato::Scalar mCellMaterialDensity; /*!< material density */
    Plato::Scalar mMassCriterionWeight; /*!< weight for mass term, i.e. /f$ \alpha_{\mbox{mass}} * f_{\mbox{mass}} /f$ */
    Plato::Scalar mStressCriterionWeight; /*!< weight for constraint term, i.e. /f$ \alpha_{\mbox{constraint}} * f_{\mbox{constraint}} /f$ */
    Plato::Scalar mAugLagPenaltyUpperBound; /*!< upper bound on augmented Lagrangian penalty */
    Plato::Scalar mMassNormalizationMultiplier; /*!< normalization multipliers for mass criterion */
    Plato::Scalar mInitialLagrangeMultipliersValue; /*!< initial value for Lagrange multipliers */
    Plato::Scalar mAugLagPenaltyExpansionMultiplier; /*!< expansion parameter for augmented Lagrangian penalty */

    Plato::ScalarVector mLagrangeMultipliers; /*!< Lagrange multipliers */
    Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffMatrix; /*!< cell/element Lame constants matrix */

private:
    /******************************************************************************//**
     * \brief Allocate member data
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aInputParams)
    {
        Plato::ElasticModelFactory<mSpaceDim> tMaterialModelFactory(aInputParams);
        auto tMaterialModel = tMaterialModelFactory.create(mSpatialDomain.getMaterialName());
        mCellStiffMatrix = tMaterialModel->getStiffnessMatrix();

        Teuchos::ParameterList tMaterialModelsInputs = aInputParams.sublist("Material Models");
        Teuchos::ParameterList tMaterialModelInputs  = tMaterialModelsInputs.sublist(mSpatialDomain.getMaterialName());
        mCellMaterialDensity = tMaterialModelInputs.get<Plato::Scalar>("Density", 1.0);

        this->readInputs(aInputParams);

        Plato::blas1::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Read user inputs
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void readInputs(Teuchos::ParameterList & aInputParams)
    {
        Teuchos::ParameterList & tParams = aInputParams.sublist("Criteria").get<Teuchos::ParameterList>(this->getName());
        mPenalty = tParams.get<Plato::Scalar>("SIMP penalty", 3.0);
        mStressLimit = tParams.get<Plato::Scalar>("Stress Limit", 1.0);
        mAugLagPenalty = tParams.get<Plato::Scalar>("Initial Penalty", 0.25);
        mMinErsatzValue = tParams.get<Plato::Scalar>("Min. Ersatz Material", 1e-9);
        mMassCriterionWeight = tParams.get<Plato::Scalar>("Mass Criterion Weight", 1.0);
        mAugLagPenaltyUpperBound = tParams.get<Plato::Scalar>("Penalty Upper Bound", 500.0);
        mStressCriterionWeight = tParams.get<Plato::Scalar>("Stress Criterion Weight", 1.0);
        mMassNormalizationMultiplier = tParams.get<Plato::Scalar>("Mass Normalization Multiplier", 1.0);
        mInitialLagrangeMultipliersValue = tParams.get<Plato::Scalar>("Initial Lagrange Multiplier", 0.01);
        mAugLagPenaltyExpansionMultiplier = tParams.get<Plato::Scalar>("Penalty Expansion Multiplier", 1.5);
    }

    /******************************************************************************//**
     * \brief Update augmented Lagrangian penalty and upper bound on mass multipliers.
    **********************************************************************************/
    void updateAugLagPenaltyMultipliers()
    {
        mAugLagPenalty = mAugLagPenaltyExpansionMultiplier * mAugLagPenalty;
        mAugLagPenalty = std::min(mAugLagPenalty, mAugLagPenaltyUpperBound);
    }

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     **********************************************************************************/
    AugLagStressCriterionGeneral(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aFuncName
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aFuncName),
        mPenalty(3),
        mStressLimit(1),
        mAugLagPenalty(0.1),
        mMinErsatzValue(0.0),
        mCellMaterialDensity(1.0),
        mMassCriterionWeight(1.0),
        mStressCriterionWeight(1.0),
        mAugLagPenaltyUpperBound(100),
        mMassNormalizationMultiplier(1.0),
        mInitialLagrangeMultipliersValue(0.01),
        mAugLagPenaltyExpansionMultiplier(1.05),
        mLagrangeMultipliers("Lagrange Multipliers", aSpatialDomain.Mesh->NumElements())
    {
        this->initialize(aInputParams);
        this->computeStructuralMass();
    }

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     **********************************************************************************/
    AugLagStressCriterionGeneral(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, "Stress Constraint"),
        mPenalty(3),
        mStressLimit(1),
        mAugLagPenalty(0.1),
        mMinErsatzValue(0.0),
        mCellMaterialDensity(1.0),
        mMassCriterionWeight(1.0),
        mStressCriterionWeight(1.0),
        mAugLagPenaltyUpperBound(100),
        mMassNormalizationMultiplier(1.0),
        mInitialLagrangeMultipliersValue(0.01),
        mAugLagPenaltyExpansionMultiplier(1.05),
        mLagrangeMultipliers("Lagrange Multipliers", aSpatialDomain.Mesh->NumElements())
    {
        Plato::blas1::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~AugLagStressCriterionGeneral()
    {
    }

    /******************************************************************************//**
     * \brief Return augmented Lagrangian penalty multiplier
     * \return augmented Lagrangian penalty multiplier
    **********************************************************************************/
    Plato::Scalar getAugLagPenalty() const
    {
        return (mAugLagPenalty);
    }

    /******************************************************************************//**
     * \brief Return multiplier used to normalized mass contribution to the objective function
     * \return upper mass normalization multiplier
    **********************************************************************************/
    Plato::Scalar getMassNormalizationMultiplier() const
    {
        return (mMassNormalizationMultiplier);
    }

    /******************************************************************************//**
     * \brief Return Lagrange multipliers
     * \return 1D view of Lagrange multipliers
    **********************************************************************************/
    Plato::ScalarVector getLagrangeMultipliers() const
    {
        return (mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Set stress constraint limit/upper bound
     * \param [in] aInput stress constraint limit
    **********************************************************************************/
    void setStressLimit(const Plato::Scalar & aInput)
    {
        mStressLimit = aInput;
    }

    /******************************************************************************//**
     * \brief Set augmented Lagrangian function penalty multiplier
     * \param [in] aInput penalty multiplier
     **********************************************************************************/
    void setAugLagPenalty(const Plato::Scalar & aInput)
    {
        mAugLagPenalty = aInput;
    }

    /******************************************************************************//**
     * \brief Set cell material density
     * \param [in] aInput material density
     **********************************************************************************/
    void setCellMaterialDensity(const Plato::Scalar & aInput)
    {
        mCellMaterialDensity = aInput;
    }

    /******************************************************************************//**
     * \brief Set Lagrange multipliers
     * \param [in] aInput Lagrange multipliers
     **********************************************************************************/
    void setLagrangeMultipliers(const Plato::ScalarVector & aInput)
    {
        assert(aInput.size() == mLagrangeMultipliers.size());
        Plato::blas1::copy(aInput, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Set cell material stiffness matrix
     * \param [in] aInput cell material stiffness matrix
    **********************************************************************************/
    void setCellStiffMatrix(const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> & aInput)
    {
        mCellStiffMatrix = aInput;
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    void updateProblem(const Plato::ScalarMultiVector & aStateWS,
                       const Plato::ScalarMultiVector & aControlWS,
                       const Plato::ScalarArray3D & aConfigWS) override
    {
        this->updateLagrangeMultipliers(aStateWS, aControlWS, aConfigWS);
        this->updateAugLagPenaltyMultipliers();
    }

    /******************************************************************************//**
     * \brief Evaluate augmented Lagrangian stress constraint criterion
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateT>   & aStateWS,
        const Plato::ScalarMultiVectorT <ControlT> & aControlWS,
        const Plato::ScalarArray3DT     <ConfigT>  & aConfigWS,
              Plato::ScalarVectorT      <ResultT>  & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const
    {
        using SimplexPhysics = typename Plato::SimplexMechanics<mSpaceDim>;
        using StrainT = typename Plato::fad_type_t<SimplexPhysics, StateT, ConfigT>;

        const Plato::OrdinalType tNumCells = mSpatialDomain.numCells();

        Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);
        Plato::Strain<mSpaceDim> tComputeCauchyStrain;
        Plato::VonMisesYieldFunction<mSpaceDim> tComputeVonMises;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::LinearStress<EvaluationType,
                            SimplexPhysics>      tComputeCauchyStress(mCellStiffMatrix);

        // ****** ALLOCATE TEMPORARY ARRAYS ON DEVICE ******
        Plato::ScalarVectorT<ResultT> tVonMises("von mises", tNumCells);
        Plato::ScalarVectorT<ConfigT> tVolume("cell volume", tNumCells);
        Plato::ScalarVectorT<ResultT> tObjective("objective", tNumCells);
        Plato::ScalarVectorT<ResultT> tConstraint("constraint", tNumCells);
        Plato::ScalarVectorT<ResultT> tConstraintValue("constraint", tNumCells);
        Plato::ScalarVectorT<ResultT> tTrueConstraintValue("true constraint", tNumCells);
        Plato::ScalarVectorT<ResultT> tOutputVonMises("output von mises", tNumCells);
        Plato::ScalarVectorT<ResultT> tTrialConstraintValue("trial constraint", tNumCells);
        Plato::ScalarVectorT<ResultT> tVonMisesOverStressLimit("stress over limit", tNumCells);
        Plato::ScalarVectorT<ResultT> tVonMisesOverLimitMinusOne("stress over limit minus one", tNumCells);

        // ****** ALLOCATE TEMPORARY MULTI-DIM ARRAYS ON DEVICE ******
        Plato::ScalarMultiVectorT<StrainT> tCauchyStrain("strain", tNumCells, mNumVoigtTerms);
        Plato::ScalarMultiVectorT<ResultT> tCauchyStress("stress", tNumCells, mNumVoigtTerms);
        Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        // ****** TRANSFER MEMBER ARRAYS TO DEVICE ******
        auto tStressLimit = mStressLimit;
        auto tAugLagPenalty = mAugLagPenalty;
        auto tMaterialDensity = mCellMaterialDensity;
        auto tLagrangeMultipliers = mLagrangeMultipliers;
        auto tMassCriterionWeight = mMassCriterionWeight;
        auto tStressCriterionWeight = mStressCriterionWeight;
        auto tMassNormalizationMultiplier = mMassNormalizationMultiplier;

        // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
        Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
        auto tCubWeight = tCubatureRule.getCubWeight();
        auto tBasisFunc = tCubatureRule.getBasisFunctions();
        Plato::Scalar tLagrangianMultiplier = static_cast<Plato::Scalar>(1.0 / tNumCells);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, aConfigWS, tVolume);
            tVolume(aCellOrdinal) *= tCubWeight;
            tComputeCauchyStrain(aCellOrdinal, tCauchyStrain, aStateWS, tGradient);
            tComputeCauchyStress(aCellOrdinal, tCauchyStress, tCauchyStrain);
            tComputeVonMises(aCellOrdinal, tCauchyStress, tVonMises);

            // Compute Von Mises stress constraint residual
            tVonMisesOverStressLimit(aCellOrdinal) = tVonMises(aCellOrdinal) / tStressLimit;
            tVonMisesOverLimitMinusOne(aCellOrdinal) = tVonMisesOverStressLimit(aCellOrdinal) - static_cast<Plato::Scalar>(1.0);
            tConstraintValue(aCellOrdinal) = ( tVonMisesOverLimitMinusOne(aCellOrdinal) * tVonMisesOverLimitMinusOne(aCellOrdinal)
                    * tVonMisesOverLimitMinusOne(aCellOrdinal) * tVonMisesOverLimitMinusOne(aCellOrdinal) )
                    + ( tVonMisesOverLimitMinusOne(aCellOrdinal) * tVonMisesOverLimitMinusOne(aCellOrdinal) );

            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControlWS);
            ControlT tMaterialPenalty = tSIMP(tDensity);
            tOutputVonMises(aCellOrdinal) = tMaterialPenalty * tVonMises(aCellOrdinal);
            tTrialConstraintValue(aCellOrdinal) = tMaterialPenalty * tConstraintValue(aCellOrdinal);
            tTrueConstraintValue(aCellOrdinal) = tVonMisesOverStressLimit(aCellOrdinal) > static_cast<ResultT>(1.0) ?
                    tTrialConstraintValue(aCellOrdinal) : static_cast<ResultT>(0.0);

            // Compute constraint contribution to augmented Lagrangian function
            tConstraint(aCellOrdinal) = tLagrangianMultiplier * ( ( tLagrangeMultipliers(aCellOrdinal) *
                    tTrueConstraintValue(aCellOrdinal) ) + ( static_cast<Plato::Scalar>(0.5) * tAugLagPenalty *
                            tTrueConstraintValue(aCellOrdinal) * tTrueConstraintValue(aCellOrdinal) ) );

            // Compute objective contribution to augmented Lagrangian function
            tObjective(aCellOrdinal) = ( Plato::cell_mass<mNumNodesPerCell>(aCellOrdinal, tBasisFunc, aControlWS) *
                    tMaterialDensity * tVolume(aCellOrdinal) ) / tMassNormalizationMultiplier;

            // Compute augmented Lagrangian function
            aResultWS(aCellOrdinal) = (tMassCriterionWeight * tObjective(aCellOrdinal))
                    + (tStressCriterionWeight * tConstraint(aCellOrdinal));
        },"Compute Augmented Lagrangian Function");

       Plato::toMap(mDataMap, tOutputVonMises, "Vonmises", mSpatialDomain);
    }

    /******************************************************************************//**
     * \brief Update Lagrange multipliers
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    void updateLagrangeMultipliers(
        const Plato::ScalarMultiVector & aStateWS,
        const Plato::ScalarMultiVector & aControlWS,
        const Plato::ScalarArray3D & aConfigWS
    )
    {
        using SimplexPhysics = typename Plato::SimplexMechanics<mSpaceDim>;

        const Plato::OrdinalType tNumCells = mSpatialDomain.numCells();

        // Create Cauchy stress functors
        Plato::Strain<mSpaceDim> tComputeCauchyStrain;
        Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);
        Plato::VonMisesYieldFunction<mSpaceDim> tComputeVonMises;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::LinearStress<Plato::ResidualTypes<SimplexPhysics>,
                            SimplexPhysics>      tComputeCauchyStress(mCellStiffMatrix);

        // ****** ALLOCATE TEMPORARY ARRAYS ON DEVICE ******
        Plato::ScalarVector tVonMises("von mises", tNumCells);
        Plato::ScalarVector tVolume("cell volume", tNumCells);
        Plato::ScalarVector tTrueConstraint("true constraint", tNumCells);
        Plato::ScalarVector tTrialConstraint("trial constraint", tNumCells);
        Plato::ScalarVector tTrialMultiplier("trial multiplier", tNumCells);
        Plato::ScalarVector tConstraintValue("constraint residual", tNumCells);
        Plato::ScalarVector tVonMisesOverStressLimit("stress over limit", tNumCells);
        Plato::ScalarVector tVonMisesOverLimitMinusOne("stress over limit - 1", tNumCells);

        // ****** ALLOCATE TEMPORARY MULTI-DIM ARRAYS ON DEVICE ******
        Plato::ScalarMultiVector tCauchyStress("stress", tNumCells, mNumVoigtTerms);
        Plato::ScalarMultiVector tCauchyStrain("strain", tNumCells, mNumVoigtTerms);
        Plato::ScalarArray3D tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        // ****** TRANSFER MEMBER ARRAYS TO DEVICE ******
        auto tStressLimit = mStressLimit;
        auto tAugLagPenalty = mAugLagPenalty;
        auto tLagrangeMultipliers = mLagrangeMultipliers;

        // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
        Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
        auto tCubWeight = tCubatureRule.getCubWeight();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            // Compute 3D Cauchy Stress
            tComputeGradient(aCellOrdinal, tGradient, aConfigWS, tVolume);
            tVolume(aCellOrdinal) *= tCubWeight;
            tComputeCauchyStrain(aCellOrdinal, tCauchyStrain, aStateWS, tGradient);
            tComputeCauchyStress(aCellOrdinal, tCauchyStress, tCauchyStrain);
            tComputeVonMises(aCellOrdinal, tCauchyStress, tVonMises);

            // Compute Von Mises stress constraint residual
            tVonMisesOverStressLimit(aCellOrdinal) = tVonMises(aCellOrdinal) / tStressLimit;
            tVonMisesOverLimitMinusOne(aCellOrdinal) = tVonMisesOverStressLimit(aCellOrdinal) - static_cast<Plato::Scalar>(1.0);
            tConstraintValue(aCellOrdinal) = ( tVonMisesOverLimitMinusOne(aCellOrdinal) * tVonMisesOverLimitMinusOne(aCellOrdinal)
                    * tVonMisesOverLimitMinusOne(aCellOrdinal) * tVonMisesOverLimitMinusOne(aCellOrdinal) )
                    + ( tVonMisesOverLimitMinusOne(aCellOrdinal) * tVonMisesOverLimitMinusOne(aCellOrdinal) );

            // Compute penalized Von Mises stress constraint
            auto tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControlWS);
            auto tPenalty = tSIMP(tDensity);
            tTrialConstraint(aCellOrdinal) = tPenalty * tConstraintValue(aCellOrdinal);
            tTrueConstraint(aCellOrdinal) = tVonMisesOverStressLimit(aCellOrdinal) > static_cast<Plato::Scalar>(1.0) ?
                    tTrialConstraint(aCellOrdinal) : static_cast<Plato::Scalar>(0.0);

            // Compute Lagrange multiplier
            tTrialMultiplier(aCellOrdinal) = tLagrangeMultipliers(aCellOrdinal) + ( tAugLagPenalty * tTrueConstraint(aCellOrdinal) );
            tLagrangeMultipliers(aCellOrdinal) = Plato::max2(tTrialMultiplier(aCellOrdinal), static_cast<Plato::Scalar>(0.0));
        }, "Update Multipliers");
    }

    /******************************************************************************//**
     * \brief Compute structural mass (i.e. structural mass with ersatz densities set to one)
    **********************************************************************************/
    void computeStructuralMass()
    {
        auto tNumCells = mSpatialDomain.numCells();

        Plato::NodeCoordinate<mSpaceDim> tCoordinates(mSpatialDomain.Mesh);
        Plato::ScalarArray3D tConfig("configuration", tNumCells, mNumNodesPerCell, mSpaceDim);
        Plato::workset_config_scalar<mSpaceDim, mNumNodesPerCell>(tNumCells, tCoordinates, tConfig);
        Plato::ComputeCellVolume<mSpaceDim> tComputeCellVolume;

        Plato::ScalarVector tTotalMass("total mass", tNumCells);
        Plato::ScalarMultiVector tDensities("densities", tNumCells, mNumNodesPerCell);
        Kokkos::deep_copy(tDensities, 1.0);

        Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
        auto tCellMaterialDensity = mCellMaterialDensity;
        auto tCubWeight = tCubatureRule.getCubWeight();
        auto tBasisFunc = tCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            Plato::Scalar tCellVolume = 0;
            tComputeCellVolume(aCellOrdinal, tConfig, tCellVolume);
            auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(aCellOrdinal, tBasisFunc, tDensities);
            tTotalMass(aCellOrdinal) = tCellMass * tCellMaterialDensity * tCellVolume * tCubWeight;
        },"Compute Structural Mass");

        Plato::blas1::local_sum(tTotalMass, mMassNormalizationMultiplier);
    }
};
// class AugLagStressCriterionGeneral

}// namespace Plato

#include "Mechanics.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::AugLagStressCriterionGeneral<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::AugLagStressCriterionGeneral<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::AugLagStressCriterionGeneral<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::AugLagStressCriterionGeneral<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::AugLagStressCriterionGeneral<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::AugLagStressCriterionGeneral<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::AugLagStressCriterionGeneral<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::AugLagStressCriterionGeneral<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::AugLagStressCriterionGeneral<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::AugLagStressCriterionGeneral<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::AugLagStressCriterionGeneral<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::AugLagStressCriterionGeneral<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif
