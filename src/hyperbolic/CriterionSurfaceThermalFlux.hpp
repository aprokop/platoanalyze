/*
 * CriterionSurfaceThermalFlux.hpp
 *
 *  Created on: Jul 21, 2021
 */

#pragma once

#include "UtilsOmegaH.hpp"
#include "SpatialModel.hpp"
#include "UtilsTeuchos.hpp"
#include "ExpInstMacros.hpp"
#include "SurfaceIntegralUtilities.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/SimplexFluids.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \tparam PhysicsT    Plato physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \class CriterionSurfaceThermalFlux
 *
 * \brief Evaluatie surface heat flux along the user-provided boundaries (i.e. 
 *        side-sets or entity-sets). The surface heat flux is given by:
 *
 *  \f[ \int_{\Gamma_e} -\frac{k^{\ast}}{k^{\ast}_f}\nabla{T}^n n_i d\Gamma_e \f],
 *
 * where \f$ n \f$ denotes the current time step, \f$ T \f$ is the temperature 
 * field, \f$ k^{\ast} \f$ is the dimensional thermal conductivity of the material 
 * where the flux is defined, \f$ k^{\ast}_f \f$ is the dimensional thermal 
 * conductivity of the fluid, and \f$ n_i \f$ is the outward unit normal.
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class CriterionSurfaceThermalFlux : public Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace;   /*!< number of spatial dimensions on face */

    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;        /*!< number of nodes per face */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of pressure dofs per node */

    using ResultT  = typename EvaluationT::ResultScalarType; /*!< result FAD type */
    using ConfigT  = typename EvaluationT::ConfigScalarType; /*!< configuration FAD type */
    using CurTempT = typename EvaluationT::CurrentEnergyScalarType; /*!< current temperature FAD evaluation type */

    /*!< temperature gradient FAD type */
    using TempGradT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurTempT, ConfigT>;

    // set local typenames
    using SurfaceCubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace>; /*!< local short name for cubature rule class */

    // member metadata
    Plato::DataMap& mDataMap; /*!< holds output metadata */
    SurfaceCubatureRule mSurfaceCubatureRule; /*!< surface cubature rule */
    const Plato::SpatialDomain& mSpatialDomain; /*!< holds mesh and entity sets metadata for a volume domain (i.e. element block) */

    // member parameters
    std::string mFuncName; /*!< scalar funciton name */
    std::vector<std::string> mSideSets; /*!< sideset names corresponding to the surfaces associated with the surface integral */
    std::vector<Plato::Scalar> mConductivityRatios; /*!< list with surface material over fluid conductivity ratios (k_{surface} / k_{fluid}) */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aName    scalar function name
     * \param [in] aDomain  holds mesh and entity sets (e.g. node and side sets) metadata
     * \param [in] aDataMap holds output metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    CriterionSurfaceThermalFlux
    (const std::string          & aName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSurfaceCubatureRule(SurfaceCubatureRule()),
         mSpatialDomain(aDomain),
         mFuncName(aName)
    {
        auto tMyCriteria = aInputs.sublist("Criteria").sublist(aName);
        mSideSets = Plato::teuchos::parse_array<std::string>("Sides", tMyCriteria);
        mConductivityRatios = Plato::teuchos::parse_array<Plato::Scalar>("Conductivity Ratios", tMyCriteria);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~CriterionSurfaceThermalFlux(){}

    /***************************************************************************//**
     * \fn std::string name
     * \brief Returns scalar function name
     * \return scalar function name
     ******************************************************************************/
    std::string name() const override { return mFuncName; }

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate scalar function inside the computational domain \f$ \Omega \f$.
     * \param [in] aWorkSets holds state work sets initialized with correct FAD types
     * \param [in] aResult   1D output work set of size number of cells
     ******************************************************************************/
    void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    { return; }

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate scalar function along a set of user-defined boudaries \f$ d\Gamma \f$.
     * \param [in] aSpatialModel mesh and entity sets (e.g. node and side sets) metadata
     * \param [in] aWorkSets     state work sets initialized with correct FAD types
     * \param [in] aResult       1D output work set of size number of cells
     ******************************************************************************/
    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel, 
     const Plato::WorkSets & aWorkSets, 
     Plato::ScalarVectorT<ResultT> & aResult) 
    const override
    {
        // set face to element graph
        auto tFace2eElems      = mSpatialDomain.Mesh.ask_up(mNumSpatialDimsOnFace, mNumSpatialDims);
        auto tFace2Elems_map   = tFace2eElems.a2ab;
        auto tFace2Elems_elems = tFace2eElems.ab2b;

        // get element to face map
        auto tElem2Faces = mSpatialDomain.Mesh.ask_down(mNumSpatialDims, mNumSpatialDimsOnFace).ab2b;

        // set mesh vertices
        auto tFace2Verts = mSpatialDomain.Mesh.ask_verts_of(mNumSpatialDimsOnFace);
        auto tCell2Verts = mSpatialDomain.Mesh.ask_elem_verts();

        // allocate local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::CalculateSurfaceArea<mNumSpatialDims> tCalculateSurfaceArea;
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(&(mSpatialDomain.Mesh));
        Plato::CalculateSurfaceJacobians<mNumSpatialDims> tCalculateSurfaceJacobians;
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;

        // set local input worksets
        auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurTempWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurTempT>>(aWorkSets.get("current temperature"));

        // transfer member data to device
        auto tCubatureWeight = mSurfaceCubatureRule.getCubWeight();
        auto tBasisFunctions = mSurfaceCubatureRule.getBasisFunctions();
        for(auto& tName : mSideSets)
        {
            // set surface material conductivity over fluid conductivity ratio 
            auto tSideNameIndex = &tName - &mSideSets[0];
            auto tSurfaceMaterialOverFluidConductivity = mConductivityRatios[tSideNameIndex];

            // set faces on this side set
            auto tFaceOrdinalsOnSideSet = Plato::omega_h::side_set_face_ordinals(mSpatialDomain.MeshSets, tName);
            auto tNumFaces = tFaceOrdinalsOnSideSet.size();
            Plato::ScalarArray3DT<ConfigT> tJacobians("face Jacobians", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);

            // set local worksets
            auto tNumCells = mSpatialDomain.Mesh.nelems();
            Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
            Plato::ScalarMultiVectorT<TempGradT> tTempGradGP("temperature gradient", tNumCells, mNumSpatialDims);
            Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
            {
                auto tFaceOrdinal = tFaceOrdinalsOnSideSet[aFaceI];
                // for all elements connected to this face, which is either 1 or 2 elements
                for( Plato::OrdinalType tElem = tFace2Elems_map[tFaceOrdinal]; tElem < tFace2Elems_map[tFaceOrdinal+1]; tElem++ )
                {
                    // create a map from face local node index to elem local node index
                    auto tCellOrdinal = tFace2Elems_elems[tElem];
                    Plato::OrdinalType tLocalNodeOrdinals[mNumSpatialDims];
                    tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrdinals);
                    
                    // calculate gradient workset
                    tComputeGradient(tCellOrdinal, tGradient, tConfigWS, tCellVolume);

                    // calculate surface Jacobian and surface integral weight
                    ConfigT tSurfaceAreaTimesCubWeight(0.0);
                    tCalculateSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrdinals, tConfigWS, tJacobians);
                    tCalculateSurfaceArea(aFaceI, tCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

                    // project temperature gradient onto surface
                    auto tElemFaceOrdinal = Plato::omega_h::get_face_ordinal<mNumSpatialDims>(tCellOrdinal, tFaceOrdinal, tElem2Faces);
                    auto tUnitNormalVec = Plato::omega_h::unit_normal_vector(tCellOrdinal, tElemFaceOrdinal, tCoords);

                    for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++)
                    {
                        for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                        {
                            tTempGradGP(tCellOrdinal, tDim) += tGradient(tCellOrdinal, tNode, tDim) * tCurTempWS(tCellOrdinal, tNode);
                        }   
                    }

                    // integrate thermal flux, defined as \int_{\Gamma_e} -N_p^a (\nabla{T}^n n_i) d\Gamma_e
                    for( Plato::OrdinalType tNode=0; tNode < mNumNodesPerFace; tNode++)
                    {
                        for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                        {
                            aResult(tCellOrdinal) += static_cast<Plato::Scalar>(-1.0) * tBasisFunctions(tNode) * tSurfaceAreaTimesCubWeight * 
                                ( tSurfaceMaterialOverFluidConductivity * tTempGradGP(tCellOrdinal, tDim) * tUnitNormalVec(tDim) );
                        }
                    }
                }
            }, "surface thermal flux");

        }
    }
};
// class CriterionSurfaceThermalFlux

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::CriterionSurfaceThermalFlux, Plato::IncompressibleFluids, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::CriterionSurfaceThermalFlux, Plato::IncompressibleFluids, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::CriterionSurfaceThermalFlux, Plato::IncompressibleFluids, Plato::SimplexFluids, 3, 1)
#endif