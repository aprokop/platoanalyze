/*
 * SurfaceLoadIntegral.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>
#include <Omega_h_vector.hpp>

#include "UtilsOmegaH.hpp"
#include "SpatialModel.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "SurfaceIntegralUtilities.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Class for the evaluation of natural boundary condition surface integrals
 * of type: UNIFORM and UNIFORM COMPONENT.
 *
 * \tparam SpatialDim   spatial dimension
 * \tparam NumDofs      number degrees of freedom per natural boundary condition force vector
 * \tparam DofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs=SpatialDim, Plato::OrdinalType DofsPerNode=NumDofs, Plato::OrdinalType DofOffset=0>
class SurfaceLoadIntegral
{
private:
    /*!< number of spatial dimensions */
    static constexpr auto mNumSpatialDims = SpatialDim;
    /*!< number of spatial dimensions on face */
    static constexpr auto mNumSpatialDimsOnFace = mNumSpatialDims - static_cast<Plato::OrdinalType>(1);

    const std::string mSideSetName; /*!< side set name */
    const Omega_h::Vector<NumDofs> mFlux; /*!< force vector values */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace> mCubatureRule; /*!< integration rule */

public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    SurfaceLoadIntegral(const std::string & aSideSetName, const Omega_h::Vector<NumDofs>& aFlux);

    /***************************************************************************//**
     * \brief Evaluate natural boundary condition surface integrals.
     *
     * \tparam StateScalarType   state forward automatically differentiated (FAD) type
     * \tparam ControlScalarType control FAD type
     * \tparam ConfigScalarType  configuration FAD type
     * \tparam ResultScalarType  result FAD type
     *
     * \param [in]  aSpatialModel Plato spatial model
     * \param [in]  aState        2-D view of state variables.
     * \param [in]  aControl      2-D view of control variables.
     * \param [in]  aConfig       3-D view of configuration variables.
     * \param [out] aResult       Assembled vector to which the boundary terms will be added
     * \param [in]  aScale        scalar multiplier
     *
    *******************************************************************************/
    template<typename StateScalarType,
             typename ControlScalarType,
             typename ConfigScalarType,
             typename ResultScalarType>
    void operator()(
        const Plato::SpatialModel                          & aSpatialModel,
        const Plato::ScalarMultiVectorT<  StateScalarType> & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType> & aConfig,
        const Plato::ScalarMultiVectorT< ResultScalarType> & aResult,
              Plato::Scalar aScale) const;
};
// class SurfaceLoadIntegral

/***************************************************************************//**
 * \brief SurfaceLoadIntegral::SurfaceLoadIntegral constructor definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
SurfaceLoadIntegral<SpatialDim,NumDofs,DofsPerNode,DofOffset>::SurfaceLoadIntegral
(const std::string & aSideSetName, const Omega_h::Vector<NumDofs>& aFlux) :
    mSideSetName(aSideSetName),
    mFlux(aFlux),
    mCubatureRule()
{
}
// class SurfaceLoadIntegral::SurfaceLoadIntegral

/***************************************************************************//**
 * \brief SurfaceLoadIntegral::operator() function definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void SurfaceLoadIntegral<SpatialDim,NumDofs,DofsPerNode,DofOffset>::operator()(
    const Plato::SpatialModel                          & aSpatialModel,
    const Plato::ScalarMultiVectorT<  StateScalarType> & aState,
    const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
    const Plato::ScalarArray3DT    < ConfigScalarType> & aConfig,
    const Plato::ScalarMultiVectorT< ResultScalarType> & aResult,
          Plato::Scalar aScale
) const
{
    // get sideset faces
    auto tFaceLids = Plato::omega_h::side_set_face_ordinals(aSpatialModel.MeshSets, mSideSetName);
    auto tNumFaces = tFaceLids.size();

    // get mesh vertices
    auto tFace2Verts = aSpatialModel.Mesh.ask_verts_of(mNumSpatialDimsOnFace);
    auto tCell2Verts = aSpatialModel.Mesh.ask_elem_verts();

    auto tFace2eElems = aSpatialModel.Mesh.ask_up(mNumSpatialDimsOnFace, mNumSpatialDims);
    auto tFace2Elems_map   = tFace2eElems.a2ab;
    auto tFace2Elems_elems = tFace2eElems.ab2b;

    Plato::CalculateSurfaceArea<mNumSpatialDims> tCalculateSurfaceArea;
    Plato::CalculateSurfaceJacobians<mNumSpatialDims> tCalculateSurfaceJacobians;
    Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;
    Plato::ScalarArray3DT<ConfigScalarType> tJacobian("jacobian", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);

    auto tFlux = mFlux;
    auto tNodesPerFace = mNumSpatialDims;
    auto tCubatureWeight = mCubatureRule.getCubWeight();
    auto tBasisFunctions = mCubatureRule.getBasisFunctions();
    if(std::isfinite(tCubatureWeight) == false)
    {
        THROWERR("Natural Boundary Condition: A non-finite cubature weight was detected.")
    }
    auto tCubWeightTimesScale = aScale * tCubatureWeight;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
    {

      auto tFaceOrdinal = tFaceLids[aFaceI];

      // for each element that the face is connected to: (either 1 or 2)
      for( Plato::OrdinalType tLocalElemOrd = tFace2Elems_map[tFaceOrdinal]; tLocalElemOrd < tFace2Elems_map[tFaceOrdinal+1]; ++tLocalElemOrd )
      {
          // create a map from face local node index to elem local node index
          Plato::OrdinalType tLocalNodeOrd[mNumSpatialDims];
          auto tCellOrdinal = tFace2Elems_elems[tLocalElemOrd];
          tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrd);

          ConfigScalarType tSurfaceAreaTimesCubWeight(0.0);
          tCalculateSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrd, aConfig, tJacobian);
          tCalculateSurfaceArea(aFaceI, tCubWeightTimesScale, tJacobian, tSurfaceAreaTimesCubWeight);

          // project into aResult workset
          for( Plato::OrdinalType tNode=0; tNode<tNodesPerFace; tNode++)
          {
              for( Plato::OrdinalType tDof=0; tDof<NumDofs; tDof++)
              {
                  auto tCellDofOrdinal = tLocalNodeOrd[tNode] * DofsPerNode + tDof + DofOffset;
                  aResult(tCellOrdinal,tCellDofOrdinal) +=
                      tBasisFunctions(tNode) * tFlux[tDof] * tSurfaceAreaTimesCubWeight;
              }
          }
      }
    }, "surface load integral");
}
// class SurfaceLoadIntegral::operator()

}
// namespace Plato
