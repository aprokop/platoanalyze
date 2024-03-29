#ifndef ABSTRACT_SCALAR_FUNCTION
#define ABSTRACT_SCALAR_FUNCTION


#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Geometric
{

/******************************************************************************//**
 * \brief Abstract scalar function (i.e. criterion) interface
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class AbstractScalarFunction
{
protected:
    const Plato::SpatialDomain & mSpatialDomain;   /*!< Plato spatial model */
          Plato::DataMap       & mDataMap;         /*!< Plato Analyze data map */
    const std::string            mFunctionName;    /*!< my abstract scalar function name */
          bool                   mHasBoundaryTerm; /*!< false if evaluate_boundary() is not implemented */
 
public:
    /******************************************************************************//**
     * \brief Abstract scalar function constructor
     * \param [in] aMesh mesh database
     * \param [in] aDataMap Plato Engine and Plato Analyze data map
     * \param [in] aName my abstract scalar function name
    **********************************************************************************/
    AbstractScalarFunction(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap,
        const std::string          & aName
    ) :
        mSpatialDomain   (aSpatialDomain),
        mDataMap         (aDataMap),
        mFunctionName    (aName),
        mHasBoundaryTerm (false)
    {
    }

    decltype(mHasBoundaryTerm) hasBoundaryTerm() const { return mHasBoundaryTerm; }

    /******************************************************************************//**
     * \brief Abstract scalar function destructor
    **********************************************************************************/
    virtual ~AbstractScalarFunction(){}

    /******************************************************************************//**
     * \brief Evaluate abstract scalar function
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    virtual void
    evaluate(
        const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    <typename EvaluationType::ConfigScalarType > & aConfig,
              Plato::ScalarVectorT     <typename EvaluationType::ResultScalarType > & aResult
    ) const = 0;

    /******************************************************************************//**
     * \brief Evaluate abstract scalar function
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    virtual void
    evaluate_boundary(
        const Plato::SpatialModel                                                   & aModel,
        const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    <typename EvaluationType::ConfigScalarType > & aConfig,
              Plato::ScalarVectorT     <typename EvaluationType::ResultScalarType > & aResult
    ) const {}

    /******************************************************************************//**
     * \brief Update physics-based data in between optimization iterations
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    virtual void updateProblem(const Plato::ScalarMultiVector & aControl,
                               const Plato::ScalarArray3D     & aConfig)
    { return; }

    /******************************************************************************//**
     * \brief Get abstract scalar function evaluation and total gradient
    **********************************************************************************/
    virtual void postEvaluate(Plato::ScalarVector, Plato::Scalar)
    { return; }

    /******************************************************************************//**
     * \brief Get abstract scalar function evaluation
     * \param [out] aOutput scalar function evaluation
    **********************************************************************************/
    virtual void postEvaluate(Plato::Scalar& aOutput)
    { return; }

    /******************************************************************************//**
     * \brief Return abstract scalar function name
     * \return name
    **********************************************************************************/
    const decltype(mFunctionName)& getName()
    {
        return mFunctionName;
    }
};

} // namespace Geometric

} // namespace Plato

#endif
