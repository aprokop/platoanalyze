#pragma once

#include <stdio.h>
#include <Teuchos_ParameterList.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Modified Solid Isotropic Material Penalization (MSIMP) model
**********************************************************************************/
class MSIMP
{
private:
    Plato::Scalar mMinValue; /*!< minimum ersatz material */
    Plato::Scalar mPenaltyParam; /*!< SIMP model penalty parameter */

public:
    /******************************************************************************//**
     * @brief Constructor
     * @param [in] aPenalty penalty parameter
     * @param [in] aMinValue minimum ersatz material
    **********************************************************************************/
    explicit MSIMP(const Plato::Scalar & aPenalty, const Plato::Scalar & aMinValue) :
            mPenaltyParam(aPenalty),
            mMinValue(aMinValue)
    {
    }

    /******************************************************************************//**
     * @brief Constructor
     * @param [in] aInputParams input parameters
    **********************************************************************************/
    explicit MSIMP(Teuchos::ParameterList & aInputParams)
    {
        mPenaltyParam = aInputParams.get<Plato::Scalar>("Exponent", 3.0);
        mMinValue = aInputParams.get<Plato::Scalar>("Minimum Value", 0.0);
    }

    /******************************************************************************//**
     * @brief Set SIMP model penalty
     * @param [in] aInput penalty
    **********************************************************************************/
    void setPenalty(const Plato::Scalar & aInput)
    {
        mPenaltyParam = aInput;
    }

    /******************************************************************************//**
     * @brief Set minimum ersatz material value
     * @param [in] aInput minimum value
    **********************************************************************************/
    void setMinimumErsatzMaterial(const Plato::Scalar & aInput)
    {
        mMinValue = aInput;
    }

    /******************************************************************************//**
     * @brief Constructor
     * @param [in] aInputParams input parameters
     * @return penalized ersatz material
    **********************************************************************************/
    template<typename ScalarType>
    DEVICE_TYPE inline ScalarType operator()( const ScalarType & aInput ) const
    {
        auto tOutput = mMinValue + ( (static_cast<ScalarType>(1.0) - mMinValue) * pow(aInput, mPenaltyParam) );
        return tOutput;
    }
};
// class MSIMP

} // namespace Plato
