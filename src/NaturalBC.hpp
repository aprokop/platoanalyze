/*
 * NaturalBC.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include <sstream>

#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"
#include "SurfaceLoadIntegral.hpp"
#include "SurfacePressureIntegral.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Natural boundary condition type ENUM
*******************************************************************************/
struct Neumann
{
    enum bc_t
    {
        UNDEFINED = 0,
        UNIFORM = 1,
        UNIFORM_PRESSURE = 2,
        UNIFORM_COMPONENT = 3,
    };
};
// struct Neumann

/***************************************************************************//**
 * \brief Return natural boundary condition type
 * \param [in] aType natural boundary condition type string
 * \return natural boundary condition type enum
*******************************************************************************/
inline Plato::Neumann::bc_t natural_boundary_condition_type(const std::string& aType)
{
    if(aType == "Uniform")
    {
        return Plato::Neumann::UNIFORM;
    }
    else if(aType == "Uniform Pressure")
    {
        return Plato::Neumann::UNIFORM_PRESSURE;
    }
    else if(aType == "Uniform Component")
    {
        return Plato::Neumann::UNIFORM_COMPONENT;
    }
    else
    {
        std::stringstream tMsg;
        tMsg << "Natural Boundary Condition: 'Type' Parameter Keyword: '" << aType.c_str() << "' is not supported.";
        THROWERR(tMsg.str().c_str())
    }
}
// function natural_boundary_condition_type

/***************************************************************************//**
 * \brief Class for natural boundary conditions.
 *
 * \tparam SpatialDim   spatial dimension
 * \tparam NumDofs      number degrees of freedom per natural boundary condition force vector
 * \tparam DofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs=SpatialDim, Plato::OrdinalType DofsPerNode=NumDofs, Plato::OrdinalType DofOffset=0>
class NaturalBC
{
    const std::string mName;         /*!< user-defined load sublist name */
    const std::string mType;         /*!< natural boundary condition type */
    const std::string mSideSetName;  /*!< side set name */
    Omega_h::Vector<NumDofs> mFlux;  /*!< force vector values */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aLoadName user-defined name for natural boundary condition sublist
     * \param [in] aSubList  natural boundary condition input parameter sublist
    *******************************************************************************/
    NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>(const std::string & aLoadName, Teuchos::ParameterList &aSubList) :
        mName(aLoadName),
        mType(aSubList.get<std::string>("Type")),
        mSideSetName(aSubList.get<std::string>("Sides"))
    {
        auto tFlux = aSubList.get<Teuchos::Array<Plato::Scalar>>("Vector");
        for(Plato::OrdinalType tDof=0; tDof<NumDofs; tDof++)
        {
            mFlux(tDof) = tFlux[tDof];
        }
    }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    ~NaturalBC(){}

    /***************************************************************************//**
     * \brief Get the contribution to the assembled forcing vector.
     *
     * \tparam StateScalarType   state forward automatically differentiated (FAD) type
     * \tparam ControlScalarType control FAD type
     * \tparam ConfigScalarType  configuration FAD type
     * \tparam ResultScalarType  result FAD type
     *
     * \param [in]  aMesh     Omega_h mesh database.
     * \param [in]  aMeshSets Omega_h side set database.
     * \param [in]  aState    2-D view of state variables.
     * \param [in]  aControl  2-D view of control variables.
     * \param [in]  aConfig   3-D view of configuration variables.
     * \param [out] aResult   Assembled vector to which the boundary terms will be added
     * \param [in]  aScale    scalar multiplier
     *
     * The boundary terms are integrated on the parameterized surface, \f$\phi(\xi,\psi)\f$, according to:
     *  \f{eqnarray*}{
     *    \phi(\xi,\psi)=
     *       \left\{
     *        \begin{array}{ccc}
     *          N_I\left(\xi,\psi\right) x_I &
     *          N_I\left(\xi,\psi\right) y_I &
     *          N_I\left(\xi,\psi\right) z_I
     *        \end{array}
     *       \right\} \\
     *     f^{el}_{Ii} = \int_{\partial\Omega_{\xi}} N_I\left(\xi,\psi\right) t_i
     *          \left|\left|
     *            \frac{\partial\phi}{\partial\xi} \times \frac{\partial\phi}{\partial\psi}
     *          \right|\right| d\xi d\psi
     * \f}
    *******************************************************************************/
    template<typename StateScalarType,
             typename ControlScalarType,
             typename ConfigScalarType,
             typename ResultScalarType>
    void get(Omega_h::Mesh* aMesh,
             const Omega_h::MeshSets& aMeshSets,
             const Plato::ScalarMultiVectorT<  StateScalarType>&,
             const Plato::ScalarMultiVectorT<ControlScalarType>&,
             const Plato::ScalarArray3DT    < ConfigScalarType>&,
             const Plato::ScalarMultiVectorT< ResultScalarType>&,
             Plato::Scalar aScale) const;

    /***************************************************************************//**
     * \brief Return natural boundary condition sublist name
     * \return sublist name
    *******************************************************************************/
    decltype(mName) const& getSubListName() const { return mName; }

    /***************************************************************************//**
     * \brief Return side set name for this natural boundary condition
     * \return side set name
    *******************************************************************************/
    decltype(mSideSetName) const& getSideSetName() const { return mSideSetName; }

    /***************************************************************************//**
     * \brief Return force vector for this natural boundary condition
     * \return force vector values
    *******************************************************************************/
    decltype(mFlux) getValues() const { return mFlux; }

    /***************************************************************************//**
     * \brief Return natural boundary condition type
     * \return natural boundary condition type
    *******************************************************************************/
    decltype(mType) getType() const { return mType; }
};
// class NaturalBC

/***************************************************************************//**
 * \brief NaturalBC::get function definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>::get
(Omega_h::Mesh* aMesh,
 const Omega_h::MeshSets& aMeshSets,
 const Plato::ScalarMultiVectorT<  StateScalarType>& aState,
 const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
 const Plato::ScalarArray3DT    < ConfigScalarType>& aConfig,
 const Plato::ScalarMultiVectorT< ResultScalarType>& aResult,
 Plato::Scalar aScale) const
{
    auto tType = Plato::natural_boundary_condition_type(mType);
    switch(tType)
    {
        case Plato::Neumann::UNIFORM:
        case Plato::Neumann::UNIFORM_COMPONENT:
        {
            Plato::SurfaceLoadIntegral<SpatialDim, NumDofs, DofsPerNode, DofOffset> tSurfaceLoad(mSideSetName, mFlux);
            tSurfaceLoad(aMesh, aMeshSets, aState, aControl, aConfig, aResult, aScale);
            break;
        }
        case Plato::Neumann::UNIFORM_PRESSURE:
        {
            Plato::SurfacePressureIntegral<SpatialDim, NumDofs, DofsPerNode, DofOffset> tSurfacePress(mSideSetName, mFlux);
            tSurfacePress(aMesh, aMeshSets, aState, aControl, aConfig, aResult, aScale);
            break;
        }
        default:
        {
            std::stringstream tMsg;
            tMsg << "Natural Boundary Condition Type '" << mType.c_str() << "' is NOT supported.";
            THROWERR(tMsg.str().c_str())
        }
    }
}
// class NaturalBC::get

}
// namespace Plato
