#pragma once

#include "PlatoMathTypes.hpp"
#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Base class for micromorphic inertia material models
 *
 * \tparam SpatialDim spatial dimensions, options 1D, 2D, and 3D
 *
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class MicromorphicInertiaMaterial
{
protected:
    static constexpr auto mNumVoigtTerms   = (SpatialDim == 3) ? 6 :
                                             ((SpatialDim == 2) ? 3 :
                                            (((SpatialDim == 1) ? 1 : 0)));
    static constexpr auto mNumSkwTerms     = (SpatialDim == 3) ? 3 :
                                             ((SpatialDim == 2) ? 1 :
                                            (((SpatialDim == 1) ? 1 : 0)));


    static_assert(mNumVoigtTerms, "SpatialDim must be 1, 2, or 3."); /*!< number of stress-strain terms */

    Plato::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellInertiaTe;   /*!< cell inertia matrix Te, i.e. fourth-order inertia tensor for symmetric part of gradient micro-inertia */
    Plato::Matrix<mNumSkwTerms,mNumSkwTerms> mCellInertiaTc;   /*!< cell inertia matrix Tc, i.e. fourth-order inertia tensor for skew-symmetric part of gradient micro-inertia */
    Plato::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellInertiaJm;   /*!< cell inertia matrix Jm, i.e. fourth-order inertia tensor for symmetric part of free micro-inertia */
    Plato::Matrix<mNumSkwTerms,mNumSkwTerms> mCellInertiaJc;   /*!< cell inertia matrix Jc, i.e. fourth-order inertia tensor for symmetric part of free micro-inertia */

    Plato::Scalar mCellMacroscopicDensity; /*!< material density */
    Plato::Scalar mPressureScaling; /*!< pressure term scaling */
    Plato::Scalar mRayleighA; // mass coefficient

public:
    /******************************************************************************//**
     * \brief micromorphic inertia material model constructor.
    **********************************************************************************/
    MicromorphicInertiaMaterial();

    /******************************************************************************//**
     * \brief micromorphic inertia material model constructor.
     * \param [in] aParamList input parameter list
    **********************************************************************************/
    MicromorphicInertiaMaterial(const Teuchos::ParameterList& aParamList);

    /******************************************************************************//**
     * \brief Return material density (mass unit/volume unit).
     * \return material density
    **********************************************************************************/
    decltype(mCellMacroscopicDensity)     getMassDensity()     const {return mCellMacroscopicDensity;}

    /******************************************************************************//**
     * \brief Return cell inertia matrix Te.
     * \return cell stiffness matrix
    **********************************************************************************/
    decltype(mCellInertiaTe)   getInertiaMatrixTe() const {return mCellInertiaTe;}

    /******************************************************************************//**
     * \brief Return cell inertia matrix Tc.
     * \return cell stiffness matrix
    **********************************************************************************/
    decltype(mCellInertiaTc)   getInertiaMatrixTc() const {return mCellInertiaTc;}

    /******************************************************************************//**
     * \brief Return cell inertia matrix Jm.
     * \return cell stiffness matrix
    **********************************************************************************/
    decltype(mCellInertiaJm)   getInertiaMatrixJm() const {return mCellInertiaJm;}

    /******************************************************************************//**
     * \brief Return cell inertia matrix Jc.
     * \return cell stiffness matrix
    **********************************************************************************/
    decltype(mCellInertiaJc)   getInertiaMatrixJc() const {return mCellInertiaJc;}

    /******************************************************************************//**
     * \brief Return pressure term scaling. Used in the stabilized finite element formulation
     * \return pressure term scaling
    **********************************************************************************/
    decltype(mPressureScaling) getPressureScaling() const {return mPressureScaling;}

    decltype(mRayleighA)       getRayleighA()       const {return mRayleighA;}

private:
    /******************************************************************************//**
     * \brief Initialize member data to default values.
    **********************************************************************************/
    void initialize();

};
// class MicromorphicInertiaMaterial

}
// namespace Plato

