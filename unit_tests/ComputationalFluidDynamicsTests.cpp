/*
 * ComputationalFluidDynamicsTests.cpp
 *
 *  Created on: Oct 13, 2020
 */

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>
#include <unordered_map>

#include <Omega_h_mark.hpp>
#include <Omega_h_assoc.hpp>
#include <Omega_h_shape.hpp>
#include <Omega_h_array_ops.hpp>

#include "BLAS1.hpp"
#include "BLAS2.hpp"
#include "BLAS3.hpp"
#include "Simplex.hpp"
#include "Assembly.hpp"
#include "NaturalBCs.hpp"
#include "SpatialModel.hpp"
#include "EssentialBCs.hpp"
#include "ProjectToNode.hpp"
#include "PlatoUtilities.hpp"
#include "SimplexFadTypes.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoAbstractProblem.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "InterpolateFromNodal.hpp"
#include "SurfaceIntegralUtilities.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "alg/PlatoSolverFactory.hpp"

#include "PlatoTestHelpers.hpp"

namespace Plato
{

template<Plato::OrdinalType NumNodesPerCell,
         Plato::OrdinalType NumSpatialDims,
         typename ConfigT,
         typename InputT,
         typename ResultT>
DEVICE_TYPE inline void
calculate_scalar_field_gradient
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<InputT> & aField,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < NumSpatialDims; tDim++)
        {
            aResult(aCellOrdinal, tDim) += aGradient(aCellOrdinal, tNode, tDim) * aField(aCellOrdinal, tNode);
        }
    }
}

template<Plato::OrdinalType NumNodesPerCell,
         Plato::OrdinalType NumSpatialDims,
         typename ConfigT,
         typename InputT,
         typename ResultT>
DEVICE_TYPE inline void
calculate_vector_field_gradient
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<InputT> & aField,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < NumSpatialDims; tDim++)
        {
            auto tLocalCellDof = (tNode * NumSpatialDims) + tDim;
            aResult(aCellOrdinal, tDim) += aGradient(aCellOrdinal, tNode, tDim) * aField(aCellOrdinal, tLocalCellDof);
        }
    }
}

namespace blas2
{

template<Plato::OrdinalType LengthI,
         Plato::OrdinalType LengthJ,
         typename ScalarT,
         typename AViewTypeT,
         typename BViewTypeT>
DEVICE_TYPE inline void
scale(const Plato::OrdinalType & aCellOrdinal,
      const ScalarT & aScalar,
      const Plato::ScalarArray3DT<AViewTypeT> & aInputWS,
      const Plato::ScalarArray3DT<BViewTypeT> & aOutputWS)
{
    for(Plato::OrdinalType tDimI = 0; tDimI < LengthI; tDimI++)
    {
        for(Plato::OrdinalType tDimJ = 0; tDimJ < LengthJ; tDimJ++)
        {
            aOutputWS(aCellOrdinal, tDimI, tDimJ) = aScalar * aInputWS(aCellOrdinal, tDimI, tDimJ);
        }
    }
}

template<Plato::OrdinalType LengthI,
         Plato::OrdinalType LengthJ,
         typename AViewType,
         typename BViewType,
         typename CViewType>
DEVICE_TYPE inline void
dot(const Plato::OrdinalType & aCellOrdinal,
    const Plato::ScalarArray3DT<AViewType> & aTensorA,
    const Plato::ScalarArray3DT<BViewType> & aTensorB,
    const Plato::ScalarVectorT <CViewType> & aOutput)
{
    for(Plato::OrdinalType tDimI = 0; tDimI < LengthI; tDimI++)
    {
        for(Plato::OrdinalType tDimJ = 0; tDimJ < LengthJ; tDimJ++)
        {
            aOutput(aCellOrdinal) += aTensorA(aCellOrdinal, tDimI, tDimJ) * aTensorB(aCellOrdinal, tDimI, tDimJ);
        }
    }
}

}
// namespace blas3

namespace blas1
{

template<Plato::OrdinalType Length,
         typename ScalarT,
         typename AViewTypeT,
         typename BViewTypeT>
DEVICE_TYPE inline void
update(const Plato::OrdinalType & aCellOrdinal,
      const ScalarT & aAlpha,
      const Plato::ScalarMultiVectorT<AViewTypeT> & aInputWS,
      const ScalarT & aBeta,
      const Plato::ScalarMultiVectorT<BViewTypeT> & aOutputWS)
{
    for(Plato::OrdinalType tIndex = 0; tIndex < Length; tIndex++)
    {
        aOutputWS(aCellOrdinal, tIndex) =
            aBeta * aOutputWS(aCellOrdinal, tIndex) + aAlpha * aInputWS(aCellOrdinal, tIndex);
    }
}

template<Plato::OrdinalType Length,
         typename ScalarT,
         typename ResultT>
DEVICE_TYPE inline void
scale(const Plato::OrdinalType & aCellOrdinal,
      const ScalarT & aScalar,
      const Plato::ScalarMultiVectorT<ResultT> & aOutputWS)
{
    for(Plato::OrdinalType tIndex = 0; tIndex < Length; tIndex++)
    {
        aOutputWS(aCellOrdinal, tIndex) *= aScalar;
    }
}

template<Plato::OrdinalType Length,
         typename ScalarT,
         typename AViewTypeT,
         typename BViewTypeT>
DEVICE_TYPE inline void
scale(const Plato::OrdinalType & aCellOrdinal,
      const ScalarT & aScalar,
      const Plato::ScalarMultiVectorT<AViewTypeT> & aInputWS,
      const Plato::ScalarMultiVectorT<BViewTypeT> & aOutputWS)
{
    for(Plato::OrdinalType tIndex = 0; tIndex < Length; tIndex++)
    {
        aOutputWS(aCellOrdinal, tIndex) = aScalar * aInputWS(aCellOrdinal, tIndex);
    }
}

template<Plato::OrdinalType Length,
         typename AViewType,
         typename BViewType,
         typename CViewType>
DEVICE_TYPE inline void
dot(const Plato::OrdinalType & aCellOrdinal,
    const Plato::ScalarMultiVectorT<AViewType> & aVectorA,
    const Plato::ScalarMultiVectorT<BViewType> & aVectorB,
    const Plato::ScalarVectorT<CViewType>      & aOutput)
{
    for(Plato::OrdinalType tIndex = 0; tIndex < Length; tIndex++)
    {
        aOutput(aCellOrdinal) += aVectorA(aCellOrdinal, tIndex) * aVectorB(aCellOrdinal, tIndex);
    }
}

}
// namespace blas1


template <typename Type>
inline void print_fad_val_values
(const Plato::ScalarVectorT<Type> & aInput,
 const std::string & aName)
{
    std::cout << "\nStart: Print ScalarVector '" << aName << "'.\n";
    const auto tLength = aInput.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        printf("Input(%d) = %f\n", aOrdinal, aInput(aOrdinal).val());
    }, "print_fad_val_values");
    std::cout << "End: Print ScalarVector '" << aName << "'.\n";
}

template <Plato::OrdinalType NumNodesPerCell,
          Plato::OrdinalType NumDofsPerNode,
          typename Type>
inline void print_fad_dx_values
(const Plato::ScalarVectorT<Type> & aInput,
 const std::string & aName)
{
    std::cout << "\nStart: Print ScalarVector '" << aName << "'.\n";
    const auto tLength = aInput.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        for(Plato::OrdinalType tNode=0; tNode < NumNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDof=0; tDof < NumDofsPerNode; tDof++)
            {
                printf("Input(Cell=%d,Node=%d,Dof=%d) = %f\n", aOrdinal, tNode, tDof, aInput(aOrdinal).dx(tNode * NumDofsPerNode + tDof));
            }
        }
    }, "print_fad_dx_values");
    std::cout << "End: Print ScalarVector '" << aName << "'.\n";
}

namespace omega_h
{

template<typename Type>
inline Omega_h::LOs
copy(const ScalarVectorT<Type> & aInput)
{
    auto tLength = aInput.size();
    Omega_h::Write<Type> tWrite(tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tWrite[aOrdinal] = aInput(aOrdinal);
    }, "copy");

    return (Omega_h::LOs(tWrite));
}

template<typename ArrayT>
void print(const ArrayT & aInput, const std::string & aName)
{
    std::cout << "Start Printing Array with Name '" << aName << "'\n";
    auto tLength = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        printf("Array(%d)=%d\n",aOrdinal,aInput[aOrdinal]);
    }, "print");
    std::cout << "Finished Printing Array with Name '" << aName << "'\n";
}

template<Plato::OrdinalType NumSpatialDims,
         Plato::OrdinalType NumNodesPerCell>
DEVICE_TYPE inline
Plato::Scalar
calculate_element_size
(const Plato::OrdinalType & aCellOrdinal,
 const Omega_h::LOs & tCells2Nodes,
 const Omega_h::Reals & tCoords)
{
    Omega_h::Few<Omega_h::Vector<NumSpatialDims>, NumNodesPerCell> tElemCoords;
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        const Plato::OrdinalType tVertexIndex = tCells2Nodes[aCellOrdinal*NumNodesPerCell + tNode];
        for(Plato::OrdinalType tDim = 0; tDim < NumSpatialDims; tDim++)
        {
            tElemCoords[tNode][tDim] = tCoords[tVertexIndex*NumSpatialDims + tDim];
        }
    }
    auto tSphere = Omega_h::get_inball(tElemCoords);

    return (static_cast<Plato::Scalar>(2.0) * tSphere.r);
}

}
// namespace omega_h

template<Omega_h::SetType EntitySet>
inline Omega_h::LOs
entity_ordinals
(const Omega_h::MeshSets& aMeshSets,
 const std::string& aSetName,
 bool aThrow = true)
{
    auto& tEntitySets = aMeshSets[EntitySet];
    auto tEntitySetMapIterator = tEntitySets.find(aSetName);
    if( (tEntitySetMapIterator == tEntitySets.end()) && (aThrow) )
    {
        THROWERR(std::string("DID NOT FIND NODE SET WITH NAME '") + aSetName + "'. NODE SET '"
                 + aSetName + "' IS NOT DEFINED IN INPUT MESH FILE, I.E. INPUT EXODUS FILE");
    }
    auto tFaceLids = (tEntitySetMapIterator->second);
    return tFaceLids;
}
// function node_set_face_ordinals

inline void is_material_defined
(const std::string & aMaterialName,
 const Teuchos::ParameterList & aInputs)
{
    if(!aInputs.sublist("Material Models").isSublist(aMaterialName))
    {
        THROWERR(std::string("Material with tag '") + aMaterialName + "' is not defined in 'Material Models' block")
    }
}

template<Omega_h::SetType EntitySet>
inline bool
is_entity_set_defined
(const Omega_h::MeshSets& aMeshSets,
 const std::string& aSetName)
{
    auto& tNodeSets = aMeshSets[EntitySet];
    auto tNodeSetMapItr = tNodeSets.find(aSetName);
    auto tIsNodeSetDefined = tNodeSetMapItr != tNodeSets.end() ? true : false;
    return tIsNodeSetDefined;
}

template<Omega_h::SetType EntitySet>
inline Omega_h::LOs
get_entity_ordinals
(const Omega_h::MeshSets& aMeshSets,
 const std::string& aSetName,
 bool aThrow = true)
{

    if( Plato::is_entity_set_defined<EntitySet>(aMeshSets, aSetName) )
    {
        auto tEntityFaceOrdinals = Plato::entity_ordinals<EntitySet>(aMeshSets, aSetName);
        return tEntityFaceOrdinals;
    }
    else
    {
        THROWERR(std::string("Entity set, i.e. side or node set, with name '") + aSetName + "' is not defined.")
    }
}

inline Plato::OrdinalType
get_num_entities
(const Omega_h::Int aEntityDim,
 const Omega_h::Mesh & aMesh)
{
    if(aEntityDim == Omega_h::VERT)
    {
        return aMesh.nverts();
    }
    else if(aEntityDim == Omega_h::EDGE)
    {
        return aMesh.nedges();
    }
    else if(aEntityDim == Omega_h::FACE)
    {
        return aMesh.nfaces();
    }
    else if(aEntityDim == Omega_h::REGION)
    {
        return aMesh.nelems();
    }
    else
    {
        THROWERR("Entity is not supported. Supported entities: Omega_h::VERT, Omega_h::EDGE, Omega_h::FACE, and Omega_h::REGION")
    }
}

DEVICE_TYPE
inline bool equal
(const Plato::Scalar & aX,
 const Plato::Scalar & aY,
 Plato::OrdinalType aULP = 2)
{
    return (fabs(aX-aY) <= DBL_EPSILON * fabs(aX-aY) * aULP);
}

template<Plato::OrdinalType NumPoints, Plato::OrdinalType SpaceDim>
Plato::ScalarVectorT<Plato::OrdinalType>
find_node_ids_on_face_set
(const Omega_h::Mesh & aMesh,
 const Omega_h::MeshSets & aMeshSets,
 const std::string & aEntitySetName,
 const Plato::ScalarMultiVector & aPoints)
{
    Plato::ScalarVectorT<Plato::OrdinalType> tNodeIds;
    if(Plato::is_entity_set_defined<Omega_h::SIDE_SET>(aMeshSets, aEntitySetName) == false)
    {
        return tNodeIds;
    }

    auto tAllCoords = aMesh.coords();
    auto tFaceLocalIds = Plato::get_entity_ordinals<Omega_h::SIDE_SET>(aMeshSets, aEntitySetName);
    auto tFaceToNodeIds = aMesh.get_adj(Omega_h::FACE, Omega_h::VERT).ab2b;
    const auto tNumNodesOnSet = tFaceToNodeIds.size();

    Plato::ScalarVectorT<Plato::OrdinalType> tMatch("1=match & 0=no match", NumPoints);
    Plato::ScalarVectorT<Plato::OrdinalType> tNodeIdMatch("matching node ids", NumPoints);
    const auto tNumSetFaces = tFaceLocalIds.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumSetFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        auto tNumNodes = SpaceDim;
        Plato::OrdinalType tNodes[SpaceDim];
        const auto tFace = tFaceLocalIds[aOrdinal];
        for (Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
        {
            tNodes[tNode] = tFaceToNodeIds[tNumNodes*tFace+tNode];
        }

        for(Plato::OrdinalType tPoint = 0; tPoint < NumPoints; tPoint++)
        {
            for(Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
            {
                Plato::OrdinalType tSum = 0;
                for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
                {
                    tSum += Plato::equal(tAllCoords[SpaceDim*tNodes[tNode] + tDim], aPoints(tPoint,tDim)) ?
                        static_cast<Plato::OrdinalType>(1) : static_cast<Plato::OrdinalType>(0);
                }

                if(tSum == SpaceDim) 
                { //Found Match 
                    tMatch(tPoint) = 1; 
                    tNodeIdMatch(tPoint) = tNodes[tNode];
                }
            }
        }
    }, "find_node_ids_on_face_set");

    auto tHostMatch = Kokkos::create_mirror(tMatch);
    Kokkos::deep_copy(tHostMatch, tMatch);
    auto tHostNodeIdMatch = Kokkos::create_mirror(tNodeIdMatch);
    Kokkos::deep_copy(tHostNodeIdMatch, tNodeIdMatch);

    auto tLength = tNodeIdMatch.size();
    std::vector<Plato::OrdinalType> tIds;
    for(decltype(tLength) tIndex = 0; tIndex < tLength; tIndex++)
    {
        if(tHostMatch(tIndex) == 1)
        {
            tIds.push_back(tHostNodeIdMatch(tIndex));
        }
    }

    Kokkos::resize(tNodeIds, tIds.size());
    auto tHostNodeIds = Kokkos::create_mirror(tNodeIds);
    for(auto& tId : tIds)
    {
        auto tIndex = &tId - &tIds[0];
        tHostNodeIds(tIndex) = tId;
    }
    Kokkos::deep_copy(tNodeIds, tHostNodeIds);

    return tNodeIds;
}

template
<Omega_h::Int EntityDim,
 Omega_h::SetType EntitySet>
inline Omega_h::LOs
entities_on_non_prescribed_boundary
(const std::vector<std::string> & aEntitySetNames,
       Omega_h::Mesh            & aMesh,
       Omega_h::MeshSets        & aMeshSets)
{
    // returns all the boundary faces, excluding faces within the domain
    auto tEntitiesAreOnNonPrescribedBoundary = Omega_h::mark_by_class_dim(&aMesh, EntityDim, EntityDim);
    // loop over all the side sets to get non-prescribed boundary faces
    auto tNumEntities = Plato::get_num_entities(EntityDim, aMesh);
    for(auto& tEntitySetName : aEntitySetNames)
    {
        // return entity ids on prescribed side set
        auto tEntitiesOnPrescribedBoundary = Plato::get_entity_ordinals<EntitySet>(aMeshSets, tEntitySetName);
        // return boolean array (entity on prescribed side set=1, entity not on prescribed side set=0)
        auto tEntitiesAreOnPrescribedBoundary = Omega_h::mark_image(tEntitiesOnPrescribedBoundary, tNumEntities);
        // return boolean array with 1's for all entities not on prescribed side set and 0's otherwise
        auto tEntitiesAreNotOnPrescribedBoundary = Omega_h::invert_marks(tEntitiesAreOnPrescribedBoundary);
        // return boolean array (entity on the non-prescribed boundary=1, entity not on the non-prescribed boundary=0)
        tEntitiesAreOnNonPrescribedBoundary = Omega_h::land_each(tEntitiesAreOnNonPrescribedBoundary, tEntitiesAreNotOnPrescribedBoundary);
    }
    // return identification numbers of all the entities on the non-prescribed boundary
    auto tIDsOfEntitiesOnNonPrescribedBoundary = Omega_h::collect_marked(tEntitiesAreOnNonPrescribedBoundary);
    return tIDsOfEntitiesOnNonPrescribedBoundary;
}

inline std::string is_valid_function(const std::string& aInput)
{
    std::vector<std::string> tValidKeys = {"scalar function", "vector function"};
    auto tLowerKey = Plato::tolower(aInput);
    if(std::find(tValidKeys.begin(), tValidKeys.end(), tLowerKey) == tValidKeys.end())
    {
        THROWERR(std::string("Input key with tag '") + tLowerKey + "' is not a valid vector function.")
    }
    return tLowerKey;
}

inline std::vector<std::string>
sideset_names(Teuchos::ParameterList & aInputs)
{
    std::vector<std::string> tOutput;
    for (Teuchos::ParameterList::ConstIterator tItr = aInputs.begin(); tItr != aInputs.end(); ++tItr)
    {
        const Teuchos::ParameterEntry &tEntry = aInputs.entry(tItr);
        if (!tEntry.isList())
        {
            THROWERR("sideset_names: Parameter list block is not valid.  Expect lists only.")
        }

        const std::string &tName = aInputs.name(tItr);
        if(aInputs.isSublist(tName) == false)
        {
            THROWERR(std::string("Parameter sublist with name '") + tName.c_str() + "' is not defined.")
        }

        Teuchos::ParameterList &tSubList = aInputs.sublist(tName);
        if(tSubList.isParameter("Sides") == false)
        {
            THROWERR(std::string("Keyword 'Sides' is not define in Parameter Sublist with name '") + tName.c_str() + "'.")
        }
        const auto tValue = tSubList.get<std::string>("Sides");
        tOutput.push_back(tValue);
    }
    return tOutput;
}

template <typename T>
inline std::vector<T>
parse_array
(const std::string & aTag,
 const Teuchos::ParameterList & aInputs)
{
    if(!aInputs.isParameter(aTag))
    {
        std::vector<T> tOutput;
        return tOutput;
    }
    auto tSideSets = aInputs.get< Teuchos::Array<T> >(aTag);

    auto tLength = tSideSets.size();
    std::vector<T> tOutput(tLength);
    for(auto & tName : tOutput)
    {
        auto tIndex = &tName - &tOutput[0];
        tOutput[tIndex] = tSideSets[tIndex];
    }
    return tOutput;
}

template <typename T>
inline T parse_parameter
(const std::string            & aTag,
 const std::string            & aBlock,
 const Teuchos::ParameterList & aInputs)
{
    if( !aInputs.isSublist(aBlock) )
    {
        THROWERR(std::string("Parameter Sublist '") + aBlock + "' within Paramater List '" 
            + aInputs.name() + "' is not defined.")
    }
    auto tSublist = aInputs.sublist(aBlock);

    if( !tSublist.isParameter(aTag) )
    {
        THROWERR(std::string("Parameter with '") + aTag + "' is not defined in Parameter Sublist with name '" + aBlock + "'.")
    }
    auto tOutput = tSublist.get<T>(aTag);
    return tOutput;
}

/***************************************************************************//**
 *  \brief Base class for simplex-based fluid mechanics problems
 *  \tparam SpaceDim    (integer) spatial dimensions
 *  \tparam NumControls (integer) number of design variable fields (default = 1)
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexFluids: public Plato::Simplex<SpaceDim>
{
public:
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;  /*!< number of spatial dimensions */
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per simplex cell */

    // optimizable quantities of interest
    static constexpr Plato::OrdinalType mNumConfigDofsPerNode  = mNumSpatialDims; /*!< number of configuration degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumControlDofsPerNode = NumControls;     /*!< number of controls per node */
    static constexpr Plato::OrdinalType mNumConfigDofsPerCell  = mNumConfigDofsPerNode * mNumNodesPerCell;  /*!< number of configuration degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumControlDofsPerCell = mNumControlDofsPerNode * mNumNodesPerCell; /*!< number of controls per cell */

    // physical quantities of interest
    static constexpr Plato::OrdinalType mNumMassDofsPerNode     = 1; /*!< number of continuity degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumMassDofsPerCell     = mNumMassDofsPerNode * mNumNodesPerCell; /*!< number of continuity degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumEnergyDofsPerNode   = 1; /*!< number energy degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumEnergyDofsPerCell   = mNumEnergyDofsPerNode * mNumNodesPerCell; /*!< number of energy degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumMomentumDofsPerNode = mNumSpatialDims; /*!< number of momentum degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumMomentumDofsPerCell = mNumMomentumDofsPerNode * mNumNodesPerCell; /*!< number of momentum degrees of freedom per cell */

};
// class SimplexFluidDynamics

struct Solutions
{
private:
    std::unordered_map<std::string, Plato::ScalarMultiVector> mSolution;

public:
    void set(const std::string& aTag, const Plato::ScalarMultiVector& aData)
    {
        auto tLowerTag = Plato::tolower(aTag);
        mSolution[tLowerTag] = aData;
    }

    Plato::ScalarMultiVector get(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mSolution.find(tLowerTag);
        if(tItr == mSolution.end())
        {
            THROWERR(std::string("Solution with tag '") + aTag + "' is not defined in the solution map.")
        }
        return tItr->second;
    }
};
// struct Solutions


class MetaDataBase
{
public:
    virtual ~MetaDataBase() = 0;
};
inline MetaDataBase::~MetaDataBase(){}

template<class Type>
class MetaData : public MetaDataBase
{
public:
    explicit MetaData(const Type &aData) : mData(aData) {}
    MetaData() {}
    Type mData;
};

template<class Type>
inline Type metadata(const std::shared_ptr<Plato::MetaDataBase> & aInput)
{
    return (dynamic_cast<Plato::MetaData<Type>&>(aInput.operator*()).mData);
}

struct WorkSets
{
private:
    std::unordered_map<std::string, std::shared_ptr<Plato::MetaDataBase>> mData;

public:
    WorkSets(){}

    void set(const std::string & aName, const std::shared_ptr<Plato::MetaDataBase> & aData)
    {
        auto tLowerKey = Plato::tolower(aName);
        mData[tLowerKey] = aData;
    }

    const std::shared_ptr<Plato::MetaDataBase> & get(const std::string & aName) const
    {
        auto tLowerKey = Plato::tolower(aName);
        auto tItr = mData.find(tLowerKey);
        if(tItr != mData.end())
        {
            return tItr->second;
        }
        else
        {
            THROWERR(std::string("Did not find 'MetaData' with tag '") + aName + "'.")
        }
    }

    std::vector<std::string> tags() const
    {
        std::vector<std::string> tOutput;
        for(auto& tPair : mData)
        {
            tOutput.push_back(tPair.first);
        }
        return tOutput;
    }

    bool defined(const std::string & aTag) const
    {
        auto tLowerKey = Plato::tolower(aTag);
        auto tItr = mData.find(tLowerKey);
        auto tFound = tItr != mData.end();
        if(tFound)
        { return true; }
        else
        { return false; }
    }
};

template <typename PhysicsT>
struct LocalOrdinalMaps
{
    Plato::NodeCoordinate<PhysicsT::SimplexT::mNumSpatialDims> mNodeCoordinate;
    Plato::VectorEntryOrdinal<PhysicsT::SimplexT::mNumSpatialDims, 1 /*scalar dofs per node*/>                 mScalarFieldOrdinalsMap;
    Plato::VectorEntryOrdinal<PhysicsT::SimplexT::mNumSpatialDims, PhysicsT::SimplexT::mNumSpatialDims>        mVectorFieldOrdinalsMap;
    Plato::VectorEntryOrdinal<PhysicsT::SimplexT::mNumSpatialDims, PhysicsT::SimplexT::mNumControlDofsPerNode> mControlOrdinalsMap;

    LocalOrdinalMaps(Omega_h::Mesh & aMesh) :
        mNodeCoordinate(&aMesh),
        mScalarFieldOrdinalsMap(&aMesh),
        mVectorFieldOrdinalsMap(&aMesh),
        mControlOrdinalsMap(&aMesh)
    { return; }
};

struct Variables
{
private:
    std::unordered_map<std::string, Plato::Scalar> mScalars;
    std::unordered_map<std::string, Plato::ScalarVector> mVectors;

public:
    Plato::Scalar scalar(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mScalars.find(tLowerTag);
        if(tItr == mScalars.end())
        {
            THROWERR(std::string("Scalar with tag '") + aTag + "' is not defined in the variables map.")
        }
        return tItr->second;
    }

    void scalar(const std::string& aTag, const Plato::Scalar& aInput)
    {
        auto tLowerTag = Plato::tolower(aTag);
        mScalars[tLowerTag] = aInput;
    }

    Plato::ScalarVector vector(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mVectors.find(tLowerTag);
        if(tItr == mVectors.end())
        {
            THROWERR(std::string("Vector with tag '") + aTag + "' is not defined in the variables map.")
        }
        return tItr->second;
    }

    void vector(const std::string& aTag, const Plato::ScalarVector& aInput)
    {
        auto tLowerTag = Plato::tolower(aTag);
        mVectors[tLowerTag] = aInput;
    }

    bool isVectorMapEmpty() const
    {
        return mVectors.empty();
    }

    bool isScalarMapEmpty() const
    {
        return mScalars.empty();
    }

    bool defined(const std::string & aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tScalarMapItr = mScalars.find(tLowerTag);
        auto tFoundScalarTag = tScalarMapItr != mScalars.end();
        auto tVectorMapItr = mVectors.find(tLowerTag);
        auto tFoundVectorTag = tVectorMapItr != mVectors.end();

        if(tFoundScalarTag || tFoundVectorTag)
        { return true; }
        else
        { return false; }
    }
};
typedef Variables Dual;
typedef Variables Primal;


namespace Fluids
{

template<typename SimplexPhysics>
struct SimplexFadTypes
{
    using ConfigFad   = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumConfigDofsPerCell>;
    using ControlFad  = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumNodesPerCell>;
    using MassFad     = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumMassDofsPerCell>;
    using EnergyFad   = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumEnergyDofsPerCell>;
    using MomentumFad = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumMomentumDofsPerCell>;
};

// is_fad<TypesT, T>::value is true if T is of any AD type defined TypesT.
//
template <typename SimplexFadTypesT, typename ScalarType>
struct is_fad {
  static constexpr bool value = std::is_same< ScalarType, typename SimplexFadTypesT::MassFad     >::value ||
                                std::is_same< ScalarType, typename SimplexFadTypesT::ControlFad  >::value ||
                                std::is_same< ScalarType, typename SimplexFadTypesT::ConfigFad   >::value ||
                                std::is_same< ScalarType, typename SimplexFadTypesT::EnergyFad   >::value ||
                                std::is_same< ScalarType, typename SimplexFadTypesT::MomentumFad >::value;
};


// which_fad<TypesT,T1,T2>::type returns:
// -- compile error  if T1 and T2 are both AD types defined in TypesT,
// -- T1             if only T1 is an AD type in TypesT,
// -- T2             if only T2 is an AD type in TypesT,
// -- T2             if neither are AD types.
//
template <typename TypesT, typename T1, typename T2>
struct which_fad {
  static_assert( !(is_fad<TypesT,T1>::value && is_fad<TypesT,T2>::value), "Only one template argument can be an AD type.");
  using type = typename std::conditional< is_fad<TypesT,T1>::value, T1, T2 >::type;
};


// fad_type_t<PhysicsT,T1,T2,T3,...,TN> returns:
// -- compile error  if more than one of T1,...,TN is an AD type in SimplexFadTypes<PhysicsT>,
// -- type TI        if only TI is AD type in SimplexFadTypes<PhysicsT>,
// -- TN             if none of TI are AD type in SimplexFadTypes<PhysicsT>.
//
template <typename TypesT, typename ...P> struct fad_type;
template <typename TypesT, typename T> struct fad_type<TypesT, T> { using type = T; };
template <typename TypesT, typename T, typename ...P> struct fad_type<TypesT, T, P ...> {
  using type = typename which_fad<TypesT, T, typename fad_type<TypesT, P...>::type>::type;
};
template <typename PhysicsT, typename ...P> using fad_type_t = typename fad_type<SimplexFadTypes<PhysicsT>,P...>::type;

/***************************************************************************//**
 *  \brief Base class for automatic differentiation types used in fluid problems
 *  \tparam SpaceDim    (integer) spatial dimensions
 *  \tparam SimplexPhysicsT simplex fluid dynamic physics type
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct EvaluationTypes
{
    static constexpr Plato::OrdinalType mNumSpatialDims        = SimplexPhysicsT::mNumSpatialDims;        /*!< number of spatial dimensions */
    static constexpr Plato::OrdinalType mNumNodesPerCell       = SimplexPhysicsT::mNumNodesPerCell;       /*!< number of nodes per simplex cell */
    static constexpr Plato::OrdinalType mNumControlDofsPerNode = SimplexPhysicsT::mNumControlDofsPerNode; /*!< number of design variable fields */
};

template <typename SimplexPhysicsT>
struct ResultTypes : EvaluationTypes<SimplexPhysicsT>
{
    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = Plato::Scalar;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradCurrentMomentumTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = FadType;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradCurrentEnergyTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::EnergyFad;

    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = FadType;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradCurrentMassTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MassFad;

    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = FadType;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradPreviousMomentumTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = FadType;

    using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradPreviousEnergyTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::EnergyFad;

    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = FadType;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradPreviousMassTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MassFad;

    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = FadType;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradMomentumPredictorTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = FadType;
};

template <typename SimplexPhysicsT>
struct GradConfigTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename SimplexFadTypes<SimplexPhysicsT>::ConfigFad;

    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = FadType;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradControlTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename SimplexFadTypes<SimplexPhysicsT>::ControlFad;

    using ControlScalarType           = FadType;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct Evaluation
{
    using Residual         = ResultTypes<SimplexPhysicsT>;
    using GradConfig       = GradConfigTypes<SimplexPhysicsT>;
    using GradControl      = GradControlTypes<SimplexPhysicsT>;

    using GradCurMass      = GradCurrentMassTypes<SimplexPhysicsT>;
    using GradPrevMass     = GradPreviousMassTypes<SimplexPhysicsT>;

    using GradCurEnergy    = GradCurrentEnergyTypes<SimplexPhysicsT>;
    using GradPrevEnergy   = GradPreviousEnergyTypes<SimplexPhysicsT>;

    using GradCurMomentum  = GradCurrentMomentumTypes<SimplexPhysicsT>;
    using GradPrevMomentum = GradPreviousMomentumTypes<SimplexPhysicsT>;
    using GradPredictor    = GradMomentumPredictorTypes<SimplexPhysicsT>;
};


template<typename PhysicsT>
struct WorkSetBuilder
{
private:
    using SimplexPhysicsT = typename PhysicsT::SimplexT;

    using ConfigLocalOridnalMap   = Plato::NodeCoordinate<SimplexPhysicsT::mNumSpatialDims>;

    using MassLocalOridnalMap     = Plato::VectorEntryOrdinal<SimplexPhysicsT::mNumSpatialDims, SimplexPhysicsT::mNumMassDofsPerNode>;
    using EnergyLocalOridnalMap   = Plato::VectorEntryOrdinal<SimplexPhysicsT::mNumSpatialDims, SimplexPhysicsT::mNumEnergyDofsPerNode>;
    using MomentumLocalOridnalMap = Plato::VectorEntryOrdinal<SimplexPhysicsT::mNumSpatialDims, SimplexPhysicsT::mNumMomentumDofsPerNode>;
    using ControlLocalOridnalMap  = Plato::VectorEntryOrdinal<SimplexPhysicsT::mNumSpatialDims, SimplexPhysicsT::mNumControlDofsPerNode>;

    using ConfigFad   = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::ConfigFad;
    using ControlFad  = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::ControlFad;
    using MassFad     = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MassFad;
    using EnergyFad   = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::EnergyFad;
    using MomentumFad = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

public:
    void buildMomentumWorkSet
    (const Plato::SpatialDomain               & aDomain,
     const MomentumLocalOridnalMap            & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumMomentumDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aDomain, aMap, aInput, aOutput);
    }

    void buildMomentumWorkSet
    (const Plato::OrdinalType                 & aNumCells,
     const MomentumLocalOridnalMap            & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumMomentumDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aNumCells, aMap, aInput, aOutput);
    }

    void buildMomentumWorkSet
    (const Plato::SpatialDomain             & aDomain,
     const MomentumLocalOridnalMap          & aMap,
     const Plato::ScalarVector              & aInput,
     Plato::ScalarMultiVectorT<MomentumFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumMomentumDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            MomentumFad>
        (aDomain, aMap, aInput, aOutput);
    }

    void buildMomentumWorkSet
    (const Plato::OrdinalType               & aNumCells,
     const MomentumLocalOridnalMap          & aMap,
     const Plato::ScalarVector              & aInput,
     Plato::ScalarMultiVectorT<MomentumFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumMomentumDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            MomentumFad>
        (aNumCells, aMap, aInput, aOutput);
    }

    void buildEnergyWorkSet
    (const Plato::SpatialDomain               & aDomain,
     const EnergyLocalOridnalMap              & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumEnergyDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aDomain, aMap, aInput, aOutput);
    }

    void buildEnergyWorkSet
    (const Plato::OrdinalType                 & aNumCells,
     const EnergyLocalOridnalMap              & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumEnergyDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aNumCells, aMap, aInput, aOutput);
    }

    void buildEnergyWorkSet
    (const Plato::SpatialDomain              & aDomain,
     const EnergyLocalOridnalMap             & aMap,
     const Plato::ScalarVector               & aInput,
     Plato::ScalarMultiVectorT<EnergyFad>    & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumEnergyDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            EnergyFad>
        (aDomain, aMap, aInput, aOutput);
    }

    void buildEnergyWorkSet
    (const Plato::OrdinalType                & aNumCells,
     const EnergyLocalOridnalMap             & aMap,
     const Plato::ScalarVector               & aInput,
     Plato::ScalarMultiVectorT<EnergyFad>    & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumEnergyDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            EnergyFad>
        (aNumCells, aMap, aInput, aOutput);
    }

    void buildMassWorkSet
    (const Plato::SpatialDomain               & aDomain,
     const MassLocalOridnalMap                & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumMassDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aDomain, aMap, aInput, aOutput);
    }

    void buildMassWorkSet
    (const Plato::OrdinalType                 & aNumCells,
     const MassLocalOridnalMap                & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumMassDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aNumCells, aMap, aInput, aOutput);
    }

    void buildMassWorkSet
    (const Plato::SpatialDomain         & aDomain,
     const MassLocalOridnalMap          & aMap,
     const Plato::ScalarVector          & aInput,
     Plato::ScalarMultiVectorT<MassFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumMassDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            MassFad>
        (aDomain, aMap, aInput, aOutput);
    }

    void buildMassWorkSet
    (const Plato::OrdinalType           & aNumCells,
     const MassLocalOridnalMap          & aMap,
     const Plato::ScalarVector          & aInput,
     Plato::ScalarMultiVectorT<MassFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumMassDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            MassFad>
        (aNumCells, aMap, aInput, aOutput);
    }

    void buildControlWorkSet
    (const Plato::SpatialDomain               & aDomain,
     const ControlLocalOridnalMap             & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumControlDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aDomain, aMap, aInput, aOutput);
    }

    void buildControlWorkSet
    (const Plato::OrdinalType                 & aNumCells,
     const ControlLocalOridnalMap             & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumControlDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aNumCells, aMap, aInput, aOutput);
    }

    void buildControlWorkSet
    (const Plato::SpatialDomain            & aDomain,
     const ControlLocalOridnalMap          & aMap,
     const Plato::ScalarVector             & aInput,
     Plato::ScalarMultiVectorT<ControlFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumControlDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            ControlFad>
        (aDomain, aMap, aInput, aOutput);
    }

    void buildControlWorkSet
    (const Plato::OrdinalType              & aNumCells,
     const ControlLocalOridnalMap          & aMap,
     const Plato::ScalarVector             & aInput,
     Plato::ScalarMultiVectorT<ControlFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumControlDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            ControlFad>
        (aNumCells, aMap, aInput, aOutput);
    }

    void buildConfigWorkSet
    (const Plato::SpatialDomain           & aDomain,
     const ConfigLocalOridnalMap          & aMap,
     Plato::ScalarArray3DT<Plato::Scalar> & aOutput)
    {
        Plato::workset_config_scalar<
            SimplexPhysicsT::mNumConfigDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aDomain, aMap, aOutput);
    }

    void buildConfigWorkSet
    (const Plato::OrdinalType             & aNumCells,
     const ConfigLocalOridnalMap          & aMap,
     Plato::ScalarArray3DT<Plato::Scalar> & aOutput)
    {
        Plato::workset_config_scalar<
            SimplexPhysicsT::mNumConfigDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aNumCells, aMap, aOutput);
    }

    void buildConfigWorkSet
    (const Plato::SpatialDomain       & aDomain,
     const ConfigLocalOridnalMap      & aMap,
     Plato::ScalarArray3DT<ConfigFad> & aOutput)
    {
        Plato::workset_config_fad<
            SimplexPhysicsT::mNumSpatialDims,
            SimplexPhysicsT::mNumNodesPerCell,
            SimplexPhysicsT::mNumConfigDofsPerNode,
            ConfigFad>
        (aDomain, aMap, aOutput);
    }

    void buildConfigWorkSet
    (const Plato::OrdinalType         & aNumCells,
     const ConfigLocalOridnalMap      & aMap,
     Plato::ScalarArray3DT<ConfigFad> & aOutput)
    {
        Plato::workset_config_fad<
            SimplexPhysicsT::mNumSpatialDims,
            SimplexPhysicsT::mNumNodesPerCell,
            SimplexPhysicsT::mNumConfigDofsPerNode,
            ConfigFad>
        (aNumCells, aMap, aOutput);
    }
};


template
<typename EvaluationT,
 typename PhysicsT>
inline void
build_scalar_function_worksets
(const Plato::SpatialDomain              & aDomain,
 const Plato::ScalarVector               & aControls,
 const Plato::Variables                  & aVariables,
 const Plato::LocalOrdinalMaps<PhysicsT> & aMaps,
       Plato::WorkSets                   & aWorkSets)
{
    auto tNumCells = aDomain.numCells();
    Plato::Fluids::WorkSetBuilder<PhysicsT> tWorkSetBuilder;

    using CurrentVelocityT = typename EvaluationT::CurrentMomentumScalarType;
    auto tCurVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentVelocityT> > >
        ( Plato::ScalarMultiVectorT<CurrentVelocityT>("current velocity", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current velocity"), tCurVelWS->mData);
    aWorkSets.set("current velocity", tCurVelWS);

    using CurrentPressureT = typename EvaluationT::CurrentMassScalarType;
    auto tCurPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPressureT> > >
        ( Plato::ScalarMultiVectorT<CurrentPressureT>("current pressure", tNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current pressure"), tCurPressWS->mData);
    aWorkSets.set("current pressure", tCurPressWS);

    using CurrentTemperatureT = typename EvaluationT::CurrentEnergyScalarType;
    auto tCurTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentTemperatureT> > >
        ( Plato::ScalarMultiVectorT<CurrentTemperatureT>("current temperature", tNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current temperature"), tCurTempWS->mData);
    aWorkSets.set("current temperature", tCurTempWS);

    using ControlT = typename EvaluationT::ControlScalarType;
    auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    tWorkSetBuilder.buildControlWorkSet(aDomain, aMaps.mControlOrdinalsMap, aControls, tControlWS->mData);
    aWorkSets.set("control", tControlWS);

    auto tTimeStepsWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("time steps", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    Plato::workset_control_scalar_scalar<PhysicsT::SimplexT::mNumNodesPerCell>
        (aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("time steps"), tTimeStepsWS->mData);
    aWorkSets.set("time steps", tTimeStepsWS);

    using ConfigT = typename EvaluationT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell, PhysicsT::SimplexT::mNumConfigDofsPerNode) );
    tWorkSetBuilder.buildConfigWorkSet(aDomain, aMaps.mNodeCoordinate, tConfig->mData);
    aWorkSets.set("configuration", tConfig);
}

template
<typename EvaluationT,
 typename PhysicsT>
inline void
build_scalar_function_worksets
(const Plato::OrdinalType                & aNumCells,
 const Plato::ScalarVector               & aControls,
 const Plato::Variables                  & aVariables,
 const Plato::LocalOrdinalMaps<PhysicsT> & aMaps,
       Plato::WorkSets                   & aWorkSets)
{
    Plato::Fluids::WorkSetBuilder<PhysicsT> tWorkSetBuilder;

    using CurrentVelocityT = typename EvaluationT::CurrentMomentumScalarType;
    auto tCurVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentVelocityT> > >
        ( Plato::ScalarMultiVectorT<CurrentVelocityT>("current velocity", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current velocity"), tCurVelWS->mData);
    aWorkSets.set("current velocity", tCurVelWS);

    using CurrentPressureT = typename EvaluationT::CurrentMassScalarType;
    auto tCurPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPressureT> > >
        ( Plato::ScalarMultiVectorT<CurrentPressureT>("current pressure", aNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current pressure"), tCurPressWS->mData);
    aWorkSets.set("current pressure", tCurPressWS);

    using CurrentTemperatureT = typename EvaluationT::CurrentEnergyScalarType;
    auto tCurTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentTemperatureT> > >
        ( Plato::ScalarMultiVectorT<CurrentTemperatureT>("current temperature", aNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current temperature"), tCurTempWS->mData);
    aWorkSets.set("current temperature", tCurTempWS);

    using ControlT = typename EvaluationT::ControlScalarType;
    auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    tWorkSetBuilder.buildControlWorkSet(aNumCells, aMaps.mControlOrdinalsMap, aControls, tControlWS->mData);
    aWorkSets.set("control", tControlWS);

    auto tTimeStepsWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("time steps", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    Plato::workset_control_scalar_scalar<PhysicsT::SimplexT::mNumNodesPerCell>
        (aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("time steps"), tTimeStepsWS->mData);
    aWorkSets.set("time steps", tTimeStepsWS);

    using ConfigT = typename EvaluationT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell, PhysicsT::SimplexT::mNumConfigDofsPerNode) );
    tWorkSetBuilder.buildConfigWorkSet(aNumCells, aMaps.mNodeCoordinate, tConfig->mData);
    aWorkSets.set("configuration", tConfig);
}

template
<typename EvaluationT,
 typename PhysicsT>
inline void
build_vector_function_worksets
(const Plato::SpatialDomain              & aDomain,
 const Plato::ScalarVector               & aControls,
 const Plato::Variables                  & aVariables,
 const Plato::LocalOrdinalMaps<PhysicsT> & aMaps,
       Plato::WorkSets                   & aWorkSets)

{
    auto tNumCells = aDomain.numCells();
    Plato::Fluids::WorkSetBuilder<PhysicsT> tWorkSetBuilder;

    using CurrentPredictorT = typename EvaluationT::MomentumPredictorScalarType;
    auto tPredictorWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPredictorT> > >
        ( Plato::ScalarMultiVectorT<CurrentPredictorT>("current predictor", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current predictor"), tPredictorWS->mData);
    aWorkSets.set("current predictor", tPredictorWS);

    using CurrentVelocityT = typename EvaluationT::CurrentMomentumScalarType;
    auto tCurVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentVelocityT> > >
        ( Plato::ScalarMultiVectorT<CurrentVelocityT>("current velocity", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current velocity"), tCurVelWS->mData);
    aWorkSets.set("current velocity", tCurVelWS);

    using CurrentPressureT = typename EvaluationT::CurrentMassScalarType;
    auto tCurPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPressureT> > >
        ( Plato::ScalarMultiVectorT<CurrentPressureT>("current pressure", tNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current pressure"), tCurPressWS->mData);
    aWorkSets.set("current pressure", tCurPressWS);

    using CurrentTemperatureT = typename EvaluationT::CurrentEnergyScalarType;
    auto tCurTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentTemperatureT> > >
        ( Plato::ScalarMultiVectorT<CurrentTemperatureT>("current temperature", tNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current temperature"), tCurTempWS->mData);
    aWorkSets.set("current temperature", tCurTempWS);

    using PreviousVelocityT = typename EvaluationT::PreviousMomentumScalarType;
    auto tPrevVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousVelocityT> > >
        ( Plato::ScalarMultiVectorT<PreviousVelocityT>("previous velocity", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("previous velocity"), tPrevVelWS->mData);
    aWorkSets.set("previous velocity", tPrevVelWS);

    using PreviousPressureT = typename EvaluationT::PreviousMassScalarType;
    auto tPrevPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousPressureT> > >
        ( Plato::ScalarMultiVectorT<PreviousPressureT>("previous pressure", tNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("previous pressure"), tPrevPressWS->mData);
    aWorkSets.set("previous pressure", tPrevPressWS);

    using PreviousTemperatureT = typename EvaluationT::PreviousEnergyScalarType;
    auto tPrevTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousTemperatureT> > >
        ( Plato::ScalarMultiVectorT<PreviousTemperatureT>("previous temperature", tNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("previous temperature"), tPrevTempWS->mData);
    aWorkSets.set("previous temperature", tPrevTempWS);

    using ControlT = typename EvaluationT::ControlScalarType;
    auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    tWorkSetBuilder.buildControlWorkSet(aDomain, aMaps.mControlOrdinalsMap, aControls, tControlWS->mData);
    aWorkSets.set("control", tControlWS);

    using ConfigT = typename EvaluationT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell, PhysicsT::SimplexT::mNumConfigDofsPerNode) );
    tWorkSetBuilder.buildConfigWorkSet(aDomain, aMaps.mNodeCoordinate, tConfig->mData);
    aWorkSets.set("configuration", tConfig);

    auto tTimeStepsWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("time steps", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    Plato::workset_control_scalar_scalar<PhysicsT::SimplexT::mNumNodesPerCell>
        (aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("time steps"), tTimeStepsWS->mData);
    aWorkSets.set("time steps", tTimeStepsWS);

    auto tVelBCsWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("previous velocity", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("prescribed velocity"), tVelBCsWS->mData);
    aWorkSets.set("prescribed velocity", tVelBCsWS);

    if(aVariables.defined("artificial compressibility"))
    {
        auto tArtificialCompressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
            ( Plato::ScalarMultiVector("artificial compressibility", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
        Plato::workset_control_scalar_scalar<PhysicsT::SimplexT::mNumNodesPerCell>
            (aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("artificial compressibility"), tArtificialCompressWS->mData);
        aWorkSets.set("artificial compressibility", tArtificialCompressWS);
    }
}

template
<typename EvaluationT,
 typename PhysicsT>
inline void
build_vector_function_worksets
(const Plato::OrdinalType                & aNumCells,
 const Plato::ScalarVector               & aControls,
 const Plato::Variables                  & aVariables,
 const Plato::LocalOrdinalMaps<PhysicsT> & aMaps,
       Plato::WorkSets                   & aWorkSets)
{
    Plato::Fluids::WorkSetBuilder<PhysicsT> tWorkSetBuilder;

    using CurrentPredictorT = typename EvaluationT::MomentumPredictorScalarType;
    auto tPredictorWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPredictorT> > >
        ( Plato::ScalarMultiVectorT<CurrentPredictorT>("current predictor", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current predictor"), tPredictorWS->mData);
    aWorkSets.set("current predictor", tPredictorWS);

    using CurrentVelocityT = typename EvaluationT::CurrentMomentumScalarType;
    auto tCurVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentVelocityT> > >
        ( Plato::ScalarMultiVectorT<CurrentVelocityT>("current velocity", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current velocity"), tCurVelWS->mData);
    aWorkSets.set("current velocity", tCurVelWS);

    using CurrentPressureT = typename EvaluationT::CurrentMassScalarType;
    auto tCurPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPressureT> > >
        ( Plato::ScalarMultiVectorT<CurrentPressureT>("current pressure", aNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current pressure"), tCurPressWS->mData);
    aWorkSets.set("current pressure", tCurPressWS);

    using CurrentTemperatureT = typename EvaluationT::CurrentEnergyScalarType;
    auto tCurTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentTemperatureT> > >
        ( Plato::ScalarMultiVectorT<CurrentTemperatureT>("current temperature", aNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current temperature"), tCurTempWS->mData);
    aWorkSets.set("current temperature", tCurTempWS);

    using PreviousVelocityT = typename EvaluationT::PreviousMomentumScalarType;
    auto tPrevVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousVelocityT> > >
        ( Plato::ScalarMultiVectorT<PreviousVelocityT>("previous velocity", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("previous velocity"), tPrevVelWS->mData);
    aWorkSets.set("previous velocity", tPrevVelWS);

    using PreviousPressureT = typename EvaluationT::PreviousMassScalarType;
    auto tPrevPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousPressureT> > >
        ( Plato::ScalarMultiVectorT<PreviousPressureT>("previous pressure", aNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("previous pressure"), tPrevPressWS->mData);
    aWorkSets.set("previous pressure", tPrevPressWS);

    using PreviousTemperatureT = typename EvaluationT::PreviousEnergyScalarType;
    auto tPrevTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousTemperatureT> > >
        ( Plato::ScalarMultiVectorT<PreviousTemperatureT>("previous temperature", aNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("previous temperature"), tPrevTempWS->mData);
    aWorkSets.set("previous temperature", tPrevTempWS);

    using ControlT = typename EvaluationT::ControlScalarType;
    auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    tWorkSetBuilder.buildControlWorkSet(aNumCells, aMaps.mControlOrdinalsMap, aControls, tControlWS->mData);
    aWorkSets.set("control", tControlWS);

    using ConfigT = typename EvaluationT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell, PhysicsT::SimplexT::mNumConfigDofsPerNode) );
    tWorkSetBuilder.buildConfigWorkSet(aNumCells, aMaps.mNodeCoordinate, tConfig->mData);
    aWorkSets.set("configuration", tConfig);

    auto tTimeStepsWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("time steps", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    Plato::workset_control_scalar_scalar<PhysicsT::SimplexT::mNumNodesPerCell>
        (aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("time steps"), tTimeStepsWS->mData);
    aWorkSets.set("time steps", tTimeStepsWS);

    auto tVelBCsWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("previous velocity", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("prescribed velocity"), tVelBCsWS->mData);
    aWorkSets.set("prescribed velocity", tVelBCsWS);

    if(aVariables.defined("artificial compressibility"))
    {
        auto tArtificialCompressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
            ( Plato::ScalarMultiVector("artificial compressibility", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
        Plato::workset_control_scalar_scalar<PhysicsT::SimplexT::mNumNodesPerCell>
            (aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("artificial compressibility"), tArtificialCompressWS->mData);
        aWorkSets.set("artificial compressibility", tArtificialCompressWS);
    }
}



// todo: abstract scalar function
template<typename PhysicsT, typename EvaluationT>
class AbstractScalarFunction
{
private:
    using ResultT = typename EvaluationT::ResultScalarType;

public:
    AbstractScalarFunction(){}
    virtual ~AbstractScalarFunction(){}

    virtual std::string name() const = 0;
    virtual void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const = 0;
    virtual void evaluateBoundary(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const = 0;
};
// class AbstractScalarFunction

template<typename PhysicsT, typename EvaluationT>
class AverageSurfacePressure : public Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace;   /*!< number of spatial dimensions on face */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;        /*!< number of nodes per face */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of pressure dofs per node */

    using ResultT   = typename EvaluationT::ResultScalarType;
    using ConfigT   = typename EvaluationT::ConfigScalarType;
    using PressureT = typename EvaluationT::CurrentMassScalarType;

    // set local typenames
    using CubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace>;

    // member metadata
    Plato::DataMap& mDataMap;
    CubatureRule mCubatureRule;
    const Plato::SpatialDomain& mSpatialDomain;

    // member parameters
    std::string mFuncName;
    std::vector<std::string> mSideSets;

public:
    AverageSurfacePressure
    (const std::string          & aName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mCubatureRule(CubatureRule()),
         mSpatialDomain(aDomain),
         mFuncName(aName)
    {
        auto tMyCriteria = aInputs.sublist("Criteria").sublist(aName);
        mSideSets = Plato::parse_array<std::string>("Sides", tMyCriteria);
    }

    virtual ~AverageSurfacePressure(){}

    std::string name() const override { return mFuncName; }

    void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    { return; }

    void evaluateBoundary(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    {
        // set face to element graph
        auto tFace2eElems      = mSpatialDomain.Mesh.ask_up(mNumSpatialDimsOnFace, mNumSpatialDims);
        auto tFace2Elems_map   = tFace2eElems.a2ab;
        auto tFace2Elems_elems = tFace2eElems.ab2b;

        // set mesh vertices
        auto tFace2Verts = mSpatialDomain.Mesh.ask_verts_of(mNumSpatialDimsOnFace);
        auto tCell2Verts = mSpatialDomain.Mesh.ask_elem_verts();

        // allocate local functors
        Plato::CalculateSurfaceArea<mNumSpatialDims> tCalculateSurfaceArea;
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(&(mSpatialDomain.Mesh));
        Plato::CalculateSurfaceJacobians<mNumSpatialDims> tCalculateSurfaceJacobians;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumPressDofsPerNode> tIntrplScalarField;
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;

        // set local data
        auto tCubatureWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();

        // set local worksets
        auto tNumCells = mSpatialDomain.Mesh.nelems();
        Plato::ScalarVectorT<PressureT> tCurrentPressGP("current pressure at Gauss point", tNumCells);

        // set input worksets
        auto tConfigurationWS   = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurrentPressureWS = Plato::metadata<Plato::ScalarMultiVectorT<PressureT>>(aWorkSets.get("current pressure"));
        for(auto& tName : mSideSets)
        {
            // get faces on this side set
            auto tFaceOrdinalsOnSideSet = Plato::side_set_face_ordinals(mSpatialDomain.MeshSets, tName);
            auto tNumFaces = tFaceOrdinalsOnSideSet.size();
            Plato::ScalarArray3DT<ConfigT> tJacobians("face Jacobians", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);

            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
            {
                auto tFaceOrdinal = tFaceOrdinalsOnSideSet[aFaceI];
                // for all elements connected to this face, which is either 1 or 2 elements
                for( Plato::OrdinalType tElem = tFace2Elems_map[tFaceOrdinal]; tElem < tFace2Elems_map[tFaceOrdinal+1]; tElem++ )
                {
                    // create a map from face local node index to elem local node index
                    Plato::OrdinalType tLocalNodeOrdinals[mNumSpatialDims];
                    auto tCellOrdinal = tFace2Elems_elems[tElem];
                    tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrdinals);

                    // calculate surface Jacobian and surface integral weight
                    ConfigT tSurfaceAreaTimesCubWeight(0.0);
                    tCalculateSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrdinals, tConfigurationWS, tJacobians);
                    tCalculateSurfaceArea(aFaceI, tCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

                    // evaluate surface scalar function
                    tIntrplScalarField(tCellOrdinal, tBasisFunctions, tCurrentPressureWS, tCurrentPressGP);

                    // calculate surface integral, which is defined as
                    // \int_{\Gamma_e}N_p^a p^h d\Gamma_e
                    for( Plato::OrdinalType tNode=0; tNode < mNumNodesPerFace; tNode++)
                    {
                        for( Plato::OrdinalType tDof=0; tDof < mNumPressDofsPerNode; tDof++)
                        {
                            aResult(tCellOrdinal) += tBasisFunctions(tNode) *
                                tCurrentPressGP(tCellOrdinal) * tSurfaceAreaTimesCubWeight;
                        }
                    }
                }
            }, "average surface pressure integral");

        }
    }
};
// class AverageSurfacePressure

template<typename PhysicsT, typename EvaluationT>
class AverageSurfaceTemperature : public Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace;   /*!< number of spatial dimensions on face */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;        /*!< number of nodes per face */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of temperature dofs per node */

    using TempT    = typename EvaluationT::CurrentEnergyScalarType;
    using ResultT  = typename EvaluationT::ResultScalarType;
    using ConfigT  = typename EvaluationT::ConfigScalarType;

    // set local typenames
    using CubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace>;

    // member metadata
    Plato::DataMap& mDataMap;
    CubatureRule mCubatureRule;
    const Plato::SpatialDomain& mSpatialDomain;

    // member parameters
    std::string mFuncName;
    std::vector<std::string> mWallSets;

public:
    AverageSurfaceTemperature
    (const std::string          & aName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mCubatureRule(CubatureRule()),
         mSpatialDomain(aDomain),
         mFuncName(aName)
    {
        auto tMyCriteria = aInputs.sublist("Criteria").sublist(aName);
        mWallSets = Plato::parse_array<std::string>("Sides", tMyCriteria);
    }

    virtual ~AverageSurfaceTemperature(){}

    std::string name() const override { return mFuncName; }

    void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    { return; }

    void evaluateBoundary(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    {
        // set face to element graph
        auto tFace2eElems      = mSpatialDomain.Mesh.ask_up(mNumSpatialDimsOnFace, mNumSpatialDims);
        auto tFace2Elems_map   = tFace2eElems.a2ab;
        auto tFace2Elems_elems = tFace2eElems.ab2b;

        // set mesh vertices
        auto tFace2Verts = mSpatialDomain.Mesh.ask_verts_of(mNumSpatialDimsOnFace);
        auto tCell2Verts = mSpatialDomain.Mesh.ask_elem_verts();

        // allocate local functors
        Plato::CalculateSurfaceArea<mNumSpatialDims> tCalculateSurfaceArea;
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(&(mSpatialDomain.Mesh));
        Plato::CalculateSurfaceJacobians<mNumSpatialDims> tCalculateSurfaceJacobians;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumPressDofsPerNode> tIntrplScalarField;
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;

        // set local data
        auto tCubatureWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();

        // set local worksets
        auto tNumCells = mSpatialDomain.Mesh.nelems();
        Plato::ScalarVectorT<TempT> tCurrentTempGP("current temperature at Gauss point", tNumCells);

        // set input worksets
        auto tCurrentTempWS   = Plato::metadata<Plato::ScalarMultiVectorT<TempT>>(aWorkSets.get("current temperature"));
        auto tConfigurationWS = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));

        for(auto& tName : mWallSets)
        {
            // get faces on this side set
            auto tFaceOrdinalsOnSideSet = Plato::side_set_face_ordinals(mSpatialDomain.MeshSets, tName);
            auto tNumFaces = tFaceOrdinalsOnSideSet.size();
            Plato::ScalarArray3DT<ConfigT> tJacobians("face Jacobians", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);

            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
            {
                auto tFaceOrdinal = tFaceOrdinalsOnSideSet[aFaceI];
                // for all elements connected to this face, which is either 1 or 2 elements
                for( Plato::OrdinalType tElem = tFace2Elems_map[tFaceOrdinal]; tElem < tFace2Elems_map[tFaceOrdinal+1]; tElem++ )
                {
                    // create a map from face local node index to elem local node index
                    Plato::OrdinalType tLocalNodeOrdinals[mNumSpatialDims];
                    auto tCellOrdinal = tFace2Elems_elems[tElem];
                    tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrdinals);

                    // calculate surface Jacobian and surface integral weight
                    ConfigT tSurfaceAreaTimesCubWeight(0.0);
                    tCalculateSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrdinals, tConfigurationWS, tJacobians);
                    tCalculateSurfaceArea(aFaceI, tCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

                    // evaluate surface scalar function
                    tIntrplScalarField(tCellOrdinal, tBasisFunctions, tCurrentTempWS, tCurrentTempGP);

                    // calculate surface integral, which is defined as
                    // \int_{\Gamma_e}N_p^a p^h d\Gamma_e
                    for( Plato::OrdinalType tNode=0; tNode < mNumNodesPerFace; tNode++)
                    {
                        for( Plato::OrdinalType tDof=0; tDof < mNumPressDofsPerNode; tDof++)
                        {
                            aResult(tCellOrdinal) += tBasisFunctions(tNode) *
                                tCurrentTempGP(tCellOrdinal) * tSurfaceAreaTimesCubWeight;
                        }
                    }
                }
            }, "average surface temperature integral");

        }
    }
};
// class AverageSurfaceTemperature

// todo: continue to build class abstractions
template<Plato::OrdinalType NumNodesPerCell, typename ControlT>
DEVICE_TYPE inline ControlT
simp_penalization
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar      & aPhysicalParam,
 const Plato::Scalar      & aPenaltyParam,
 const Plato::ScalarMultiVectorT<ControlT> & aControlWS)
{
    ControlT tDensity = Plato::cell_density<NumNodesPerCell>(aCellOrdinal, aControlWS);
    ControlT tPenalizedDensity = pow(tDensity, aPenaltyParam);
    ControlT tPenalizedParam = tPenalizedDensity * aPhysicalParam;
    return tPenalizedParam;
}

template<Plato::OrdinalType NumNodesPerCell, typename ControlT>
DEVICE_TYPE inline ControlT
msimp_penalization
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar      & aPhysicalParam,
 const Plato::Scalar      & aPenaltyParam,
 const Plato::Scalar      & aMinErsatzParam,
 const Plato::ScalarMultiVectorT<ControlT> & aControlWS)
{
    ControlT tDensity = Plato::cell_density<NumNodesPerCell>(aCellOrdinal, aControlWS);
    ControlT tPenalizedDensity = aMinErsatzParam +
        ( (static_cast<Plato::Scalar>(1.0) - aMinErsatzParam) * pow(tDensity, aPenaltyParam) );
    ControlT tPenalizedParam = tPenalizedDensity * aPhysicalParam;
    return tPenalizedParam;
}

template<Plato::OrdinalType NumNodesPerCell, typename ControlT>
DEVICE_TYPE inline ControlT
ramp_penalization
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar      & aPhysicalParam,
 const Plato::Scalar      & aConvexityParam,
 const Plato::ScalarMultiVectorT<ControlT> & aControlWS)
{
    ControlT tDensity = Plato::cell_density<NumNodesPerCell>(aCellOrdinal, aControlWS);
    ControlT tPenalizedPhysicalParam =
        ( tDensity * ( aPhysicalParam * (static_cast<Plato::Scalar>(1.0) - aConvexityParam)
            - static_cast<Plato::Scalar>(1.0) ) + static_cast<Plato::Scalar>(1.0) )
            / ( aPhysicalParam * (static_cast<Plato::Scalar>(1.0) + aConvexityParam * tDensity) );
    return tPenalizedPhysicalParam;
}

template<Plato::OrdinalType NumNodesPerCell, typename ControlT>
DEVICE_TYPE inline ControlT
brinkman_penalization
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar      & aPhysicalParam,
 const Plato::Scalar      & aConvexityParam,
 const Plato::ScalarMultiVectorT<ControlT> & aControlWS)
{
    ControlT tDensity = Plato::cell_density<NumNodesPerCell>(aCellOrdinal, aControlWS);
    ControlT tPenalizedPhysicalParam = aPhysicalParam * (static_cast<Plato::Scalar>(1.0) - tDensity)
        / (static_cast<Plato::Scalar>(1.0) + (aConvexityParam * tDensity));
    return tPenalizedPhysicalParam;
}

// calculate strain rate for incompressible flows, which is defined as
// \frac{1}{2}\left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right)
template<Plato::OrdinalType NumNodesPerCell,
         Plato::OrdinalType NumSpaceDim,
         typename AViewTypeT,
         typename BViewTypeT,
         typename CViewTypeT>
DEVICE_TYPE inline void
strain_rate
(const Plato::OrdinalType & aCellOrdinal,
 const AViewTypeT & aStateWS,
 const BViewTypeT & aGradient,
 const CViewTypeT & aStrainRate)
{
    // calculate strain rate for incompressible flows, which is defined as
    // \frac{1}{2}\left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right)
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < NumSpaceDim; tDimI++)
        {
            for(Plato::OrdinalType tDimJ = 0; tDimJ < NumSpaceDim; tDimJ++)
            {
                auto tLocalDimI = tNode * NumSpaceDim + tDimI;
                auto tLocalDimJ = tNode * NumSpaceDim + tDimJ;
                aStrainRate(aCellOrdinal, tDimI, tDimJ) += static_cast<Plato::Scalar>(0.5) *
                    ( ( aGradient(aCellOrdinal, tNode, tDimJ) * aStateWS(aCellOrdinal, tLocalDimI) )
                    + ( aGradient(aCellOrdinal, tNode, tDimI) * aStateWS(aCellOrdinal, tLocalDimJ) ) );
            }
        }
    }
}

template<Plato::OrdinalType NumSpaceDim,
         typename ControlT,
         typename StrainT,
         typename StressT>
DEVICE_TYPE inline void
deviatoric_stress
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aPrandtlNum,
 const Plato::ScalarArray3DT<StrainT> & aStrain,
 const Plato::ScalarArray3DT<StressT> & aStress)
{
    for(Plato::OrdinalType tDimI = 0; tDimI < NumSpaceDim; tDimI++)
    {
        for(Plato::OrdinalType tDimJ = 0; tDimJ < NumSpaceDim; tDimJ++)
        {
            aStress(aCellOrdinal, tDimI, tDimJ) +=
                static_cast<Plato::Scalar>(2.0) * aPrandtlNum * aStrain(aCellOrdinal, tDimI, tDimJ);
        }
    }
}



// todo: internal energy
// calculate internal energy, which is defined as
//   \f$ \int_{\Omega_e}\left[ \tau_{ij}(\theta):\tau_{ij}(\theta) + \alpha(\theta)u_i^2 \right] d\Omega_e, \f$
// where \f$\theta\f$ denotes the controls, \f$\alpha\f$ denotes the Brinkman penalization parameter.
template<typename PhysicsT, typename EvaluationT>
class InternalDissipationEnergyIncompressible : public Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims    = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell   = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerNode = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of velocity dofs per node */

    // local forward automatic differentiation typenames
    using ResultT  = typename EvaluationT::ResultScalarType;
    using CurVelT  = typename EvaluationT::CurrentMomentumScalarType;
    using ConfigT  = typename EvaluationT::ConfigScalarType;
    using ControlT = typename EvaluationT::ControlScalarType;
    using StrainT  = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurVelT, ConfigT>;

    // set local typenames
    using CubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>;

    // member parameters
    std::string mFuncName;
    Plato::Scalar mPrNum = 1.0;
    Plato::Scalar mDaNum = 1.0;
    Plato::Scalar mBrinkmanConvexityParam = 0.5;

    // member metadata
    Plato::DataMap& mDataMap;
    CubatureRule mCubatureRule;
    const Plato::SpatialDomain& mSpatialDomain;

public:
    InternalDissipationEnergyIncompressible
    (const std::string          & aName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mFuncName(aName),
         mDataMap(aDataMap),
         mCubatureRule(CubatureRule()),
         mSpatialDomain(aDomain)
    {
        mDaNum = Plato::parse_parameter<Plato::Scalar>("Darcy Number", "Dimensionless Properties", aInputs);
        mPrNum = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", aInputs);
        this->setPenaltyModels(aInputs);
    }

    virtual ~InternalDissipationEnergyIncompressible(){}

    std::string name() const override { return mFuncName; }

    void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    {
        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // set local worksets
        auto tNumCells = mSpatialDomain.numCells();
        Plato::ScalarVectorT<ConfigT> tVolumeTimesWeight("volume times gauss weight", tNumCells);
        Plato::ScalarVectorT<CurVelT> tCurVelDotCurVel("current velocity dot current velocity", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::ScalarArray3DT<StrainT> tStrainRate("strain rate", tNumCells, mNumSpatialDims, mNumSpatialDims);
        Plato::ScalarArray3DT<ResultT> tDevStress("deviatoric stress", tNumCells, mNumSpatialDims, mNumSpatialDims);
        Plato::ScalarMultiVectorT<CurVelT> tCurVelGP("current velocity at Gauss point", tNumCells, mNumSpatialDims);

        // set input worksets
        auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurVelWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurVelT>>(aWorkSets.get("current velocity"));

        // transfer member data to device
        auto tPrNum = mPrNum;
        auto tDaNum = mDaNum;
        auto tBrinkConvexParam = mBrinkmanConvexityParam;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tVolumeTimesWeight);
            tVolumeTimesWeight(aCellOrdinal) *= tCubWeight;

            // calculate deviatoric stress contribution to internal energy
            Plato::Fluids::strain_rate<mNumNodesPerCell, mNumSpatialDims>(aCellOrdinal, tCurVelWS, tGradient, tStrainRate);
            auto tTwoTimesPrNum = static_cast<Plato::Scalar>(2.0) * tPrNum;
            Plato::blas2::scale<mNumSpatialDims, mNumSpatialDims>(aCellOrdinal, tTwoTimesPrNum, tStrainRate, tDevStress);
            Plato::blas2::dot<mNumSpatialDims, mNumSpatialDims>(aCellOrdinal, tDevStress, tDevStress, aResult);

            // calculate fictitious material model (i.e. brinkman model) contribution to internal energy
            auto tPermeability = tPrNum / tDaNum;
            ControlT tPenalizedPermeability =
                Plato::Fluids::brinkman_penalization<mNumNodesPerCell>(aCellOrdinal, tPermeability, tBrinkConvexParam, tControlWS);
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurVelWS, tCurVelGP);
            Plato::blas1::dot<mNumSpatialDims>(aCellOrdinal, tCurVelGP, tCurVelGP, tCurVelDotCurVel);
            aResult(aCellOrdinal) += tPenalizedPermeability * tCurVelDotCurVel(aCellOrdinal);

            // apply gauss weight times volume multiplier
            aResult(aCellOrdinal) *= tVolumeTimesWeight(aCellOrdinal);

        }, "internal energy");
    }

    void evaluateBoundary(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    { return; }

private:
    void setPenaltyModels(Teuchos::ParameterList & aInputs)
    {
        auto tMyCriterionInputs = aInputs.sublist("Criteria").sublist(mFuncName);
        if(tMyCriterionInputs.isSublist("Penalty Function"))
        {
            auto tPenaltyFuncInputs = tMyCriterionInputs.sublist("Penalty Function");
            mBrinkmanConvexityParam = tPenaltyFuncInputs.get<Plato::Scalar>("Brinkman Convexity Parameter", 0.5);
        }
    }
};
// class InternalDissipationEnergyIncompressible


/******************************************************************************/
/*! scalar function class

   This class takes as a template argument a scalar function in the form:

   \f$ J = J(\phi, U^k, P^k, T^k, X) \f$

   and manages the evaluation of the function and derivatives with respect to
   control, \f$\phi\f$, momentum state, \f$ U^k \f$, mass state, \f$ P^k \f$,
   energy state, \f$ T^k \f$, and configuration, \f$ X \f$.

*/
/******************************************************************************/
class CriterionBase
{
public:
    virtual ~CriterionBase(){}

    /******************************************************************************//**
     * \brief Return function name
     * \return user defined function name
     **********************************************************************************/
    virtual std::string name() const = 0;

    virtual Plato::Scalar value
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const = 0;

    virtual Plato::ScalarVector gradientConfig
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const = 0;

    virtual Plato::ScalarVector gradientControl
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const = 0;

    virtual Plato::ScalarVector gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const = 0;

    virtual Plato::ScalarVector gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const = 0;

    virtual Plato::ScalarVector gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const = 0;
};
// class ScalarFunctionBase


// todo: physics scalar function
template<typename PhysicsT>
class PhysicsScalarFunction : public Plato::Fluids::CriterionBase
{
private:
    std::string mFuncName;

    static constexpr auto mNumSpatialDims         = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell        = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumMassDofsPerCell     = PhysicsT::SimplexT::mNumMassDofsPerCell;     /*!< number of mass dofs per cell */
    static constexpr auto mNumEnergyDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumMomentumDofsPerCell = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumMassDofsPerNode     = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumEnergyDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumMomentumDofsPerNode = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumConfigDofsPerCell   = PhysicsT::SimplexT::mNumConfigDofsPerCell;   /*!< number of configuration degrees of freedom per cell */
    static constexpr auto mNumControlDofsPerNode  = PhysicsT::SimplexT::mNumControlDofsPerNode;  /*!< number of design variables per node */

    // forward automatic differentiation evaluation types
    using ResidualEvalT     = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradConfigEvalT   = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradConfig;
    using GradControlEvalT  = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradControl;
    using GradCurVelEvalT   = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurMomentum;
    using GradCurTempEvalT  = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurEnergy;
    using GradCurPressEvalT = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurMass;

    // element scalar functions types
    using ResidualFunc     = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, ResidualEvalT>>;
    using GradConfigFunc   = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, GradConfigEvalT>>;
    using GradControlFunc  = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, GradControlEvalT>>;
    using GradCurVelFunc   = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, GradCurVelEvalT>>;
    using GradCurTempFunc  = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, GradCurTempEvalT>>;
    using GradCurPressFunc = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, GradCurPressEvalT>>;

    // element scalar functions per element block, i.e. domain
    std::unordered_map<std::string, ResidualFunc>     mResidualFuncs;
    std::unordered_map<std::string, GradConfigFunc>   mGradConfigFuncs;
    std::unordered_map<std::string, GradControlFunc>  mGradControlFuncs;
    std::unordered_map<std::string, GradCurVelFunc>   mGradCurrentVelocityFuncs;
    std::unordered_map<std::string, GradCurPressFunc> mGradCurrentPressureFuncs;
    std::unordered_map<std::string, GradCurTempFunc>  mGradCurrentTemperatureFuncs;

    Plato::DataMap& mDataMap;
    const Plato::SpatialModel& mSpatialModel;
    Plato::LocalOrdinalMaps<PhysicsT> mLocalOrdinalMaps;

public:
    PhysicsScalarFunction
    (Plato::SpatialModel    & aModel,
     Plato::DataMap         & aDataMap,
     Teuchos::ParameterList & aInputs,
     std::string            & aName):
        mFuncName(aName),
        mSpatialModel(aModel),
        mDataMap(aDataMap),
        mLocalOrdinalMaps(aModel.Mesh)
    {
        this->initialize(aInputs);
    }

    virtual ~PhysicsScalarFunction(){}

    std::string name() const
    {
        return mFuncName;
    }

    Plato::Scalar value
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const
    {
        using ResultScalarT = typename ResidualEvalT::ResultScalarType;
        ResultScalarT tReturnValue(0.0);

        // evaluate internal domain
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_scalar_function_worksets<ResidualEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mResidualFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            tReturnValue += Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_scalar_function_worksets<ResidualEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mResidualFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            tReturnValue += Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
        }

        return tReturnValue;
    }

    Plato::ScalarVector gradientConfig
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const
    {
        using ResultScalarT = typename GradConfigEvalT::ResultScalarType;
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradient("gradient wrt configuration", mNumSpatialDims * tNumNodes);

        // evaluate internal domain
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_scalar_function_worksets<GradConfigEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mGradConfigFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>
                (tDomain, mLocalOrdinalMaps.mVectorFieldOrdinalsMap, tResultWS, tGradient);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_scalar_function_worksets<GradConfigEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mGradConfigFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>
                (tNumCells, mLocalOrdinalMaps.mVectorFieldOrdinalsMap, tResultWS, tGradient);
        }

        return tGradient;
    }

    Plato::ScalarVector gradientControl
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const
    {
        using ResultScalarT = typename GradControlEvalT::ResultScalarType;
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradient("gradient wrt control", mNumControlDofsPerNode * tNumNodes);

        // evaluate internal domain
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_scalar_function_worksets<GradControlEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mGradControlFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumControlDofsPerNode>
                (tDomain, mLocalOrdinalMaps.mControlOrdinalsMap, tResultWS, tGradient);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_scalar_function_worksets<GradControlEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mGradControlFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumControlDofsPerNode>
                (tNumCells, mLocalOrdinalMaps.mControlOrdinalsMap, tResultWS, tGradient);
        }

        return tGradient;
    }

    Plato::ScalarVector gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const
    {
        using ResultScalarT = typename GradCurPressEvalT::ResultScalarType;
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradient("gradient wrt current pressure state", mNumMassDofsPerNode * tNumNodes);

        // evaluate internal domain
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_scalar_function_worksets<GradCurPressEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mGradCurrentPressureFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumMassDofsPerNode>
                (tDomain, mLocalOrdinalMaps.mScalarFieldOrdinalsMap, tResultWS, tGradient);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_scalar_function_worksets<GradCurPressEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mGradCurrentPressureFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumMassDofsPerNode>
                (tNumCells, mLocalOrdinalMaps.mScalarFieldOrdinalsMap, tResultWS, tGradient);
        }

        return tGradient;
    }

    Plato::ScalarVector gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const
    {
        using ResultScalarT = typename GradCurTempEvalT::ResultScalarType;
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradient("gradient wrt current temperature state", mNumEnergyDofsPerNode * tNumNodes);

        // evaluate internal domain
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_scalar_function_worksets<GradCurTempEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mGradCurrentTemperatureFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumEnergyDofsPerNode>
                (tDomain, mLocalOrdinalMaps.mScalarFieldOrdinalsMap, tResultWS, tGradient);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_scalar_function_worksets<GradCurTempEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mGradCurrentTemperatureFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumEnergyDofsPerNode>
                (tNumCells, mLocalOrdinalMaps.mScalarFieldOrdinalsMap, tResultWS, tGradient);
        }

        return tGradient;
    }

    Plato::ScalarVector gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const
    {
        using ResultScalarT = typename GradCurVelEvalT::ResultScalarType;
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradient("gradient wrt current velocity state", mNumMomentumDofsPerNode * tNumNodes);

        // evaluate internal domain
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_scalar_function_worksets<GradCurVelEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mGradCurrentVelocityFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumMomentumDofsPerNode>
                (tDomain, mLocalOrdinalMaps.mVectorFieldOrdinalsMap, tResultWS, tGradient);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_scalar_function_worksets<GradCurVelEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mGradCurrentVelocityFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumMomentumDofsPerNode>
                (tNumCells, mLocalOrdinalMaps.mVectorFieldOrdinalsMap, tResultWS, tGradient);
        }

        return tGradient;
    }

private:
    void initialize(Teuchos::ParameterList & aInputs)
    {
        typename PhysicsT::FunctionFactory tScalarFuncFactory;
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mResidualFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<PhysicsT, ResidualEvalT>
                    (mFuncName, tDomain, mDataMap, aInputs);

            mGradConfigFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<PhysicsT, GradConfigEvalT>
                    (mFuncName, tDomain, mDataMap, aInputs);

            mGradControlFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<PhysicsT, GradControlEvalT>
                    (mFuncName, tDomain, mDataMap, aInputs);

            mGradCurrentPressureFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<PhysicsT, GradCurPressEvalT>
                    (mFuncName, tDomain, mDataMap, aInputs);

            mGradCurrentTemperatureFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<PhysicsT, GradCurTempEvalT>
                    (mFuncName, tDomain, mDataMap, aInputs);

            mGradCurrentVelocityFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<PhysicsT, GradCurVelEvalT>
                    (mFuncName, tDomain, mDataMap, aInputs);
        }
    }
};
// class PhysicsScalarFunction










template<typename PhysicsT, typename EvaluationT>
class PressureSurfaceForces
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;       /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;      /*!< number of nodes per face */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;   /*!< number of pressure dofs per node */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace; /*!< number of spatial dimensions on face */

    // forward automatic differentiation types
    using ResultT    = typename EvaluationT::ResultScalarType;
    using ConfigT    = typename EvaluationT::ConfigScalarType;
    using PrevPressT = typename EvaluationT::PreviousMassScalarType;

    // set local type names
    using VolumeCubatureRule = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>;
    using SurfaceCubatureRule = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace>;

    const std::string mSideSetName = "not defined"; /*!< side set name */

    VolumeCubatureRule mVolumeCubatureRule; /*!< volume integration rule */
    SurfaceCubatureRule mSurfaceCubatureRule; /*!< surface integration rule */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */

public:
    PressureSurfaceForces
    (const Plato::SpatialDomain & aSpatialDomain,
     std::string aSideSetName = "not defined") :
         mSideSetName(aSideSetName),
         mSpatialDomain(aSpatialDomain)
    {
        auto tLowerTag = Plato::tolower(mSideSetName);
        if(tLowerTag == "not defined")
        {
            THROWERR("Side set with name '" + mSideSetName + "' is not defined.")
        }
    }

    void operator()
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult,
     Plato::Scalar aMultiplier = 1.0) const
    {
        // get mesh vertices
        auto tFace2Verts = mSpatialDomain.Mesh.ask_verts_of(mNumSpatialDimsOnFace);
        auto tCell2Verts = mSpatialDomain.Mesh.ask_elem_verts();

        // get face to element graph
        auto tFace2eElems = mSpatialDomain.Mesh.ask_up(mNumSpatialDimsOnFace, mNumSpatialDims);
        auto tFace2Elems_map   = tFace2eElems.a2ab;
        auto tFace2Elems_elems = tFace2eElems.ab2b;

        // get element to face map
        auto tElem2Faces = mSpatialDomain.Mesh.ask_down(mNumSpatialDims, mNumSpatialDimsOnFace).ab2b;

        // set local functors
        Plato::CalculateSurfaceArea<mNumSpatialDims> tCalculateSurfaceArea;
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(&(mSpatialDomain.Mesh));
        Plato::CalculateSurfaceJacobians<mNumSpatialDims> tCalculateSurfaceJacobians;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumPressDofsPerNode> tIntrplScalarField;
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;

        // get sideset faces
        auto tFaceLocalOrdinals = Plato::side_set_face_ordinals(mSpatialDomain.MeshSets, mSideSetName);
        auto tNumFaces = tFaceLocalOrdinals.size();
        Plato::ScalarArray3DT<ConfigT> tSurfaceJacobians("jacobian", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);
        auto tNumCells = mSpatialDomain.Mesh.nelems();
        Plato::ScalarVectorT<PrevPressT> tPrevPressGP("previous pressure at Gauss point", tNumCells);

        // set input state worksets
        auto tConfigWS    = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevPressWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevPressT>>(aWorkSets.get("previous pressure"));

        // calculate surface integral
        auto tVolumeBasisFunctions = mVolumeCubatureRule.getBasisFunctions();
        auto tSurfaceCubatureWeight = mSurfaceCubatureRule.getCubWeight();
        auto tSurfaceBasisFunctions = mSurfaceCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
        {

          auto tFaceOrdinal = tFaceLocalOrdinals[aFaceI];
          // for each element that the face is connected to: (either 1 or 2 elements)
          for( Plato::OrdinalType tElem = tFace2Elems_map[tFaceOrdinal]; tElem < tFace2Elems_map[tFaceOrdinal + 1]; tElem++ )
          {
              // create a map from face local node index to elem local node index
              Plato::OrdinalType tLocalNodeOrd[mNumSpatialDims];
              auto tCellOrdinal = tFace2Elems_elems[tElem];
              tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrd);

              // calculate surface jacobians
              ConfigT tSurfaceAreaTimesCubWeight(0.0);
              tCalculateSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrd, tConfigWS, tSurfaceJacobians);
              tCalculateSurfaceArea(aFaceI, tSurfaceCubatureWeight, tSurfaceJacobians, tSurfaceAreaTimesCubWeight);

              // compute unit normal vector
              auto tElemFaceOrdinal = Plato::get_face_ordinal<mNumSpatialDims>(tCellOrdinal, tFaceOrdinal, tElem2Faces);
              auto tUnitNormalVec = Plato::unit_normal_vector(tCellOrdinal, tElemFaceOrdinal, tCoords);

              // project into aResult workset
              tIntrplScalarField(tCellOrdinal, tVolumeBasisFunctions, tPrevPressWS, tPrevPressGP);
              for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++ )
              {
                  for( Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++ )
                  {
                      auto tLocalCellNode = tLocalNodeOrd[tNode];
                      auto tLocalCellDof = (tLocalCellNode * mNumSpatialDims) + tDim;
                      aResult(tCellOrdinal, tLocalCellDof) += aMultiplier * tSurfaceBasisFunctions(tNode)
                          * tSurfaceAreaTimesCubWeight * (tPrevPressGP(tCellOrdinal) * tUnitNormalVec(tDim));
                  }
              }
          }
        }, "calculate surface pressure integral");
    }
};
// class PressureSurfaceForces






/***************************************************************************//**
 * \brief Class used to integrate deviatoric stresses on a surface.  The integral
 * is defined as: \int_{\Gamma - \Gamma_t} N_i^a \left(\tau_{ij}n_j\right) d\Gamma.
 * The aforedefined integral is used by the momentum predictor residual evaluations.
 *
 * Note: The implementation works well for linear tetrahedral elements since the
 * gradient is constant across the element. However, the graident calculation, i.e
 * the calculation of the derivative of the shape functions, will require the
 * implementation of a map from a 3-D quantity into a 2 dimensional surface. Thus,
 * the gradients will need to be calculated using this map, which is not the case
 * for linear tetrahedral elements.
 *
 * \tparam PhysicsT    physics type
 * \tparam EvaluationT forward automatic differentiation type
 *
*******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class DeviatoricSurfaceForces
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;       /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;      /*!< number of nodes per face */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;      /*!< number of nodes per cell */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace; /*!< number of spatial dimensions on face */

    // forward automatic differentiation types
    using ResultT  = typename EvaluationT::ResultScalarType;
    using ConfigT  = typename EvaluationT::ConfigScalarType;
    using ControlT = typename EvaluationT::ControlScalarType;
    using PrevVelT = typename EvaluationT::PreviousMomentumScalarType;
    using StrainT  = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>;

    // set local typenames
    using SurfaceCubatureRule = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace>;

    std::string   mSideSetName; /*!< side set name */
    Plato::Scalar mPrNum = 1.0; /*!< prandtl number */

    Omega_h::LOs mFaceOrdinalsOnBoundary;
    SurfaceCubatureRule mSurfaceCubatureRule; /*!< surface integration rule */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */

public:
    DeviatoricSurfaceForces
    (const Plato::SpatialDomain & aSpatialDomain,
     Teuchos::ParameterList & aInputs,
     std::string aSideSetName = "") :
         mSideSetName(aSideSetName),
         mSpatialDomain(aSpatialDomain)
    {
        this->setParameters(aInputs);
        this->setFacesOnNonPrescribedBoundary(aInputs);
    }

    void operator()
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult,
     Plato::Scalar aMultiplier = 1.0) const
    {
        // get mesh vertices
        auto tFace2Verts = mSpatialDomain.Mesh.ask_verts_of(mNumSpatialDimsOnFace);
        auto tCell2Verts = mSpatialDomain.Mesh.ask_elem_verts();

        // get face to element graph
        auto tFace2eElems = mSpatialDomain.Mesh.ask_up(mNumSpatialDimsOnFace, mNumSpatialDims);
        auto tFace2Elems_map   = tFace2eElems.a2ab;
        auto tFace2Elems_elems = tFace2eElems.ab2b;

        // get element to face map
        auto tElem2Faces = mSpatialDomain.Mesh.ask_down(mNumSpatialDims, mNumSpatialDimsOnFace).ab2b;

        // set local functors
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(&(mSpatialDomain.Mesh));
        Plato::ComputeGradientWorkset<mNumSpatialDims> tCalculateGradients;
        Plato::CalculateSurfaceJacobians<mNumSpatialDims> tCalculateSurfaceJacobians;
        Plato::CalculateSurfaceArea<mNumSpatialDims> tCalculateSurfaceAreaTimesCubWeight;
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;

        // transfer member data to device
        auto tPrNum = mPrNum;
        auto tFaceOrdinalsOnBoundary = mFaceOrdinalsOnBoundary;

        // set local data structures
        auto tNumCells = mSpatialDomain.Mesh.nelems();
        Plato::ScalarVectorT<ConfigT>  tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradients("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::ScalarArray3DT<StrainT> tStrainRate("cell strain rate", tNumCells, mNumSpatialDims, mNumSpatialDims);
        auto tNumFaces = tFaceOrdinalsOnBoundary.size();
        Plato::ScalarArray3DT<ConfigT> tJacobians("surface jacobians", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);

        // set input state worksets
        auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));

        // calculate surface integral
        auto tSurfaceCubatureWeight = mSurfaceCubatureRule.getCubWeight();
        auto tSurfaceBasisFunctions = mSurfaceCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
        {
            auto tFaceOrdinal = tFaceOrdinalsOnBoundary[aFaceI];
            // for each element that the face is connected to: (either 1 or 2 elements)
            for( Plato::OrdinalType tElem = tFace2Elems_map[tFaceOrdinal]; tElem < tFace2Elems_map[tFaceOrdinal + 1]; tElem++ )
            {
                // create a map from face local node index to elem local node index
                Plato::OrdinalType tLocalNodeOrd[mNumSpatialDims];
                auto tCellOrdinal = tFace2Elems_elems[tElem];
                tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrd);

                // calculate surface jacobians
                ConfigT tSurfaceAreaTimesCubWeight(0.0);
                tCalculateSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrd, tConfigWS, tJacobians);
                tCalculateSurfaceAreaTimesCubWeight(aFaceI, tSurfaceCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

                // compute unit normal vector
                auto tElemFaceOrdinal = Plato::get_face_ordinal<mNumSpatialDims>(tCellOrdinal, tFaceOrdinal, tElem2Faces);
                auto tUnitNormalVec = Plato::unit_normal_vector(tCellOrdinal, tElemFaceOrdinal, tCoords);

                // calculate strain rate
                tCalculateGradients(tCellOrdinal, tGradients, tConfigWS, tCellVolume);
                Plato::Fluids::strain_rate<mNumNodesPerCell, mNumSpatialDims>
                    (tCellOrdinal, tPrevVelWS, tGradients, tStrainRate);

                // calculate deviatoric traction forces, which are defined as,
                // \int_{\Gamma_e} N_u^a \left( \tau^h_{ij}n_j \right) d\Gamma_e
                for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++)
                {
                    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
                    {
                        auto tLocalCellDofI = (mNumSpatialDims * tNode) + tDimI;
                        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
                        {
                            aResult(tCellOrdinal, tLocalCellDofI) += aMultiplier * tSurfaceBasisFunctions(tNode)
                            * tSurfaceAreaTimesCubWeight * ( ( static_cast<Plato::Scalar>(2.0) * tPrNum *
                                    tStrainRate(tCellOrdinal, tDimI, tDimJ) ) * tUnitNormalVec(tDimJ) );
                        }
                    }
                }
            }

        }, "calculate deviatoric stress surface integral");
    }

private:
    void setParameters
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        auto tHyperParamList = aInputs.sublist("Hyperbolic");

        mPrNum = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperParamList);
        this->setPenaltyModel(tHyperParamList);
    }

    void setPenaltyModel
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Momentum Conservation"))
        {
            auto tMomentumParamList = aInputs.sublist("Momentum Conservation");
            if (tMomentumParamList.isSublist("Penalty Function"))
            {
                auto tPenaltyFuncList = tMomentumParamList.sublist("Penalty Function");
            }
        }
    }

    void setFacesOnNonPrescribedBoundary(Teuchos::ParameterList& aInputs)
    {
        if (mSideSetName.empty())
        {
            if (aInputs.isSublist("Momentum Natural Boundary Conditions"))
            {
                auto tNaturalBCs = aInputs.sublist("Momentum Natural Boundary Conditions");
                auto tPrescribedSideSetNames = Plato::sideset_names(tNaturalBCs);
                mFaceOrdinalsOnBoundary =
                    Plato::entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>
                        (tPrescribedSideSetNames, mSpatialDomain.Mesh, mSpatialDomain.MeshSets);
            }
            else
            {
                // integral is performed along all the fluid surfaces
                std::vector<std::string> tEmptyPrescribedBCs;
                mFaceOrdinalsOnBoundary =
                    Plato::entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>
                        (tEmptyPrescribedBCs, mSpatialDomain.Mesh, mSpatialDomain.MeshSets);
            }
        }
        else
        {
            mFaceOrdinalsOnBoundary = Plato::side_set_face_ordinals(mSpatialDomain.MeshSets, mSideSetName);
        }
    }
};
// class DeviatoricSurfaceForces

// todo: abstract vector function
template<typename PhysicsT, typename EvaluationT>
class AbstractVectorFunction
{
private:
    using ResultT = typename EvaluationT::ResultScalarType;

public:
    AbstractVectorFunction(){}
    virtual ~AbstractVectorFunction(){}

    virtual void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const = 0;

    virtual void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const = 0;

    virtual void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const = 0;
};
// class AbstractVectorFunction


// calculate f_i^{convective}=\frac{\partial}{\partial x_j}(u^{n-1}_j u^{n-1}_i), where
// \frac{\partial}{\partial x_j}(u^{n-1}_j u^{n-1}_i) = \frac{\partial u^{n-1}_j}{\partial x_j}u^{n-1}_i + u^{n-1}_j\frac{\partial u^{n-1}_i}{\partial x_j}
template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ConfigT,
 typename PrevVelT>
DEVICE_TYPE inline void
calculate_advected_internal_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelWS,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < SpaceDim; tDimI++)
        {
            auto tCellDofI = (SpaceDim * tNode) + tDimI;
            for(Plato::OrdinalType tDimJ = 0; tDimJ < SpaceDim; tDimJ++)
            {
                auto tCellDofJ = (SpaceDim * tNode) + tDimJ;
                aResult(aCellOrdinal, tDimI) +=
                    ( ( aGradient(aCellOrdinal, tNode, tDimJ) * aPrevVelWS(aCellOrdinal, tCellDofJ) ) * aPrevVelGP(aCellOrdinal, tDimI) ) +
                    ( aPrevVelGP(aCellOrdinal, tDimJ) * ( aGradient(aCellOrdinal, tNode, tDimJ) * aPrevVelWS(aCellOrdinal, tCellDofI) ) );
            }
        }
    }
}

// calculate viscous force integral, which is defined as,
// \int_{\Omega_e}\frac{\partial N_u^a}{\partial x_j}\tau^h_{ij} d\Omega_e
template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ConfigT,
 typename ControlT,
 typename StrainT>
DEVICE_TYPE inline void
integrate_viscous_forces
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aPrandtlNumber,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarArray3DT<StrainT> & aStrainRate,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < SpaceDim; tDimI++)
        {
            auto tDofIndex = (SpaceDim * tNode) + tDimI;
            for(Plato::OrdinalType tDimJ = 0; tDimJ < SpaceDim; tDimJ++)
            {
                aResult(aCellOrdinal, tDofIndex) += aCellVolume(aCellOrdinal) * aGradient(aCellOrdinal, tNode, tDimJ)
                    * ( static_cast<Plato::Scalar>(2.0) * aPrandtlNumber * aStrainRate(aCellOrdinal, tDimI, tDimJ) );
            }
        }
    }
}

// calculate stabilized natural convective force integral, which is defined as F_i = Gr_i Pr^2 T^h
template
<Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ControlT,
 typename PrevTempT>
DEVICE_TYPE inline void
calculate_natural_convective_forces
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aPenalizedPrNumSquared,
 const Plato::ScalarVector & aGrashofNum,
 const Plato::ScalarVectorT<PrevTempT> & aPrevTempGP,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for (Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
    {
        aResult(aCellOrdinal, tDim) += aGrashofNum(tDim) * aPenalizedPrNumSquared * aPrevTempGP(aCellOrdinal);
    }
}

// calculate stabilized brinkman force, which is defined as F_i = \frac{Pr}{Da} u^{n-1}_i
template
<Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ControlT,
 typename PrevVelT>
DEVICE_TYPE inline void
calculate_brinkman_forces
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aPenalizedBrinkmanCoeff,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for (Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
    {
        aResult(aCellOrdinal, tDim) += aPenalizedBrinkmanCoeff * aPrevVelGP(aCellOrdinal, tDim);
    }
}

template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ConfigT,
 typename PrevVelT>
DEVICE_TYPE inline
void divergence
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelWS,
 const Plato::ScalarVectorT<ResultT> & aResult)
{
    for (Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for (Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            auto tCellDof = (SpaceDim * tNode) + tDim;
            aResult(aCellOrdinal) += aGradient(aCellOrdinal, tNode, tDim) * aPrevVelWS(aCellOrdinal, tCellDof);
        }
    }
}

// calculate stabilizing force integral, which are defined as
// \int_{\Omega_e} \left( \frac{\partial N_u^a}{\partial x_k} u^{n-1}_k \right) F_i^{stab} d\Omega_e
// where the stanilized force, F_i^{stab}, is defined as
// F_i^{stab} = \frac{\partial}{\partial x_j}(u^{n-1}_j u^{n-1}_i) + Gr_i Pr^2 T^h + \frac{Pr}{Da} u^{n-1}_i
template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename DivVelT,
 typename ResultT,
 typename ConfigT,
 typename PrevVelT,
 typename StabForceT>
DEVICE_TYPE inline void
integrate_stabilizing_vector_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarVectorT<DivVelT> & aDivPrevVel,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarMultiVectorT<StabForceT> & aStabForce,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
 {
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < SpaceDim; tDimI++)
        {
            auto tLocalCellDofI = (SpaceDim * tNode) + tDimI;
            for(Plato::OrdinalType tDimK = 0; tDimK < SpaceDim; tDimK++)
            {
                aResult(aCellOrdinal, tLocalCellDofI) += aCellVolume(aCellOrdinal) * aStabForce(aCellOrdinal, tDimI)
                    * ( aGradient(aCellOrdinal, tNode, tDimK) * aPrevVelGP(aCellOrdinal, tDimK) );
            }
            aResult(aCellOrdinal, tLocalCellDofI) += aCellVolume(aCellOrdinal) * aStabForce(aCellOrdinal, tDimI)
                * ( aBasisFunctions(tNode) * aDivPrevVel(aCellOrdinal) );
        }
    }
 }

template
<Plato::OrdinalType NumNodesPerCell,
 Plato::OrdinalType NumDofPerNode,
 typename ResultT>
DEVICE_TYPE inline void
multiply_time_step
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar & aMultiplier,
 const Plato::ScalarMultiVector & aTimeStepWS,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDof = 0; tDof < NumDofPerNode; tDof++)
        {
            auto tLocalCellDof = (NumDofPerNode * tNode) + tDof;
            aResult(aCellOrdinal, tLocalCellDof) *= aTimeStepWS(aCellOrdinal, tNode) * aMultiplier;
        }
    }
}

// calculate inertial force integral, which are defined as
// \int_{Omega_e} N_u^a \left( u^\ast_i - u^{n-1}_i \right) d\Omega_e
// or it can be used to calculate
// \int_{Omega_e} N_u^a \left( u^{n}_i - u^\ast_i \right) d\Omega_e
template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ConfigT,
 typename PredVelT,
 typename PrevVelT>
DEVICE_TYPE inline void
calculate_inertial_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<PredVelT> & aVecA_GP,
 const Plato::ScalarMultiVectorT<PrevVelT> & aVecB_GP,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < SpaceDim; tDimI++)
        {
            auto tDofIndex = (SpaceDim * tNode) + tDimI;
            aResult(aCellOrdinal, tDofIndex) += aCellVolume(aCellOrdinal) * aBasisFunctions(tNode) *
                ( aVecA_GP(aCellOrdinal, tDimI) - aVecB_GP(aCellOrdinal, tDimI) );
        }
    }
}

template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ConfigT,
 typename ForceT>
DEVICE_TYPE inline
void integrate_vector_field
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<ForceT> & aForce,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            auto tLocalCellDof = (SpaceDim * tNode) + tDim;
            aResult(aCellOrdinal, tLocalCellDof) +=
                aCellVolume(aCellOrdinal) * aBasisFunctions(tNode) * aForce(aCellOrdinal, tDim);
        }
    }
}

struct SupportedApplyWeightTags
{
private:
    std::vector<std::string> mData = {"convexity", "minimum ersatz", "penalty"};

public:
    bool supported(const std::string & aTag)
    {
        if(std::find(mData.begin(), mData.end(), aTag) == mData.end())
        {
            THROWERR(std::string("Tag '" + aTag + "' is not a valid keyword."))
        }
        return true;
    }
};

struct ApplyWeightData
{
private:
    Plato::Fluids::SupportedApplyWeightTags mValidTags;
    std::unordered_map<std::string, Plato::Scalar> mData;

public:
    void set(const std::string & aTag, const Plato::Scalar & aInput)
    {
        auto tLowerTag = Plato::tolower(aTag);
        mValidTags.supported(tLowerTag);
        mData[tLowerTag] = aInput;
    }

    Plato::Scalar get(const std::string & aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mData.find(tLowerTag);
        if(tItr == mData.end())
        {
            THROWERR(std::string("Parameter with tag '" + aTag + "' is not defined."))
        }
        return tItr->second;
    }
};

template<typename EvaluationT>
class NoWeight
{
private:
    using ControlT = typename EvaluationT::ControlScalarType;

public:
    void set(const Plato::Fluids::ApplyWeightData & aInputs){ return; }

    DEVICE_TYPE inline ControlT operator()
    (const Plato::OrdinalType & aCellOrdinal,
     const Plato::Scalar & aPhysicalProperty,
     const Plato::ScalarMultiVectorT<ControlT> & aControlWS) const
    {
        ControlT tOutput = static_cast<Plato::Scalar>(1.0) * aPhysicalProperty;
        return tOutput;
    }
};

template<typename PhysicsT, typename EvaluationT>
class WeightSIMP
{
private:
    static constexpr auto mNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    using ControlT = typename EvaluationT::ControlScalarType;

    Plato::Scalar mPenalty;

public:
    void set(const Plato::Fluids::ApplyWeightData & aInputs)
    {
        mPenalty = aInputs.get("penalty");
    }

    DEVICE_TYPE inline ControlT operator()
    (const Plato::OrdinalType & aCellOrdinal,
     const Plato::Scalar & aPhysicalProperty,
     const Plato::ScalarMultiVectorT<ControlT> & aControlWS) const
    {
        ControlT tOutput =
            Plato::Fluids::simp_penalization<mNumNodesPerCell>(aCellOrdinal, aPhysicalProperty, mPenalty, aControlWS);
        return tOutput;
    }
};

template<typename PhysicsT, typename EvaluationT>
class WeightMSIMP
{
private:
    static constexpr auto mNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    using ControlT = typename EvaluationT::ControlScalarType;

    Plato::Scalar mPenalty;
    Plato::Scalar mMinErsatz;

public:
    void set(const Plato::Fluids::ApplyWeightData & aInputs)
    {
        mPenalty = aInputs.get("penalty");
        mMinErsatz = aInputs.get("minimum ersatz");
    }

    DEVICE_TYPE inline ControlT operator()
    (const Plato::OrdinalType & aCellOrdinal,
     const Plato::Scalar & aPhysicalProperty,
     const Plato::ScalarMultiVectorT<ControlT> & aControlWS) const
    {
        ControlT tOutput =
            Plato::Fluids::msimp_penalization<mNumNodesPerCell>(aCellOrdinal, aPhysicalProperty, mPenalty, mMinErsatz, aControlWS);
        return tOutput;
    }
};

template<typename PhysicsT, typename EvaluationT>
class WeightBrinkman
{
private:
    static constexpr auto mNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    using ControlT = typename EvaluationT::ControlScalarType;

    Plato::Scalar mConvexity;

public:
    void set(const Plato::Fluids::ApplyWeightData & aInputs)
    {
        mConvexity = aInputs.get("convexity");
    }

    DEVICE_TYPE inline ControlT operator()
    (const Plato::OrdinalType & aCellOrdinal,
     const Plato::Scalar & aPhysicalProperty,
     const Plato::ScalarMultiVectorT<ControlT> & aControlWS) const
    {
        ControlT tOutput =
            Plato::Fluids::brinkman_penalization<mNumNodesPerCell>(aCellOrdinal, aPhysicalProperty, mConvexity, aControlWS);
        return tOutput;
    }
};

template<typename PhysicsT, typename EvaluationT>
class WeightRAMP
{
private:
    static constexpr auto mNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    using ControlT = typename EvaluationT::ControlScalarType;

    Plato::Scalar mConvexity;

public:
    void set(const Plato::Fluids::ApplyWeightData & aInputs)
    {
        mConvexity = aInputs.get("convexity");
    }

    DEVICE_TYPE inline ControlT operator()
    (const Plato::OrdinalType & aCellOrdinal,
     const Plato::Scalar & aPhysicalProperty,
     const Plato::ScalarMultiVectorT<ControlT> & aControlWS) const
    {
        ControlT tOutput =
            Plato::Fluids::ramp_penalization<mNumNodesPerCell>(aCellOrdinal, aPhysicalProperty, mConvexity, aControlWS);
        return tOutput;
    }
};

template<typename Method, typename EvaluationT>
class ApplyWeight
{
private:
    using ControlT = typename EvaluationT::ControlScalarType;
    Method mMethod;

public:
    ApplyWeight(const Plato::Fluids::ApplyWeightData & aInputs)
    {
        mMethod.set(aInputs);
    }

    DEVICE_TYPE inline ControlT operator()
    (const Plato::OrdinalType & aCellOrdinal,
     const Plato::Scalar & aPhysicalProperty,
     const Plato::ScalarMultiVectorT<ControlT> & aControlWS) const
    {
        ControlT tOutput = mMethod(aCellOrdinal, aPhysicalProperty, aControlWS);
        return tOutput;
    }
};

template<typename PhysicsT, typename EvaluationT>
class VelocityPredictorResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumTempDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell;   /*!< number of configuration degrees of freedom per cell */

    // set local ad types
    using ResultT   = typename EvaluationT::ResultScalarType;
    using ConfigT   = typename EvaluationT::ConfigScalarType;
    using ControlT  = typename EvaluationT::ControlScalarType;
    using PrevVelT  = typename EvaluationT::PreviousMomentumScalarType;
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType;
    using PredVelT  = typename EvaluationT::MomentumPredictorScalarType;
    using StrainT   = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>;

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature rule evaluator */

    // set local type names
    using PressureForces   = Plato::Fluids::PressureSurfaceForces<PhysicsT, EvaluationT>;
    using DeviatoricForces = Plato::Fluids::DeviatoricSurfaceForces<PhysicsT, EvaluationT>;

    // set external force evaluators
    std::unordered_map<std::string, std::shared_ptr<PressureForces>>     mPressureBCs;   /*!< prescribed pressure forces */
    std::unordered_map<std::string, std::shared_ptr<DeviatoricForces>>   mDeviatoricBCs; /*!< stabilized deviatoric boundary forces */
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mPrescribedBCs; /*!< prescribed boundary conditions, e.g. tractions */

    // set member scalar data
    Plato::Scalar mBuoyancy = 1.0; /*!< buoyancy dimensionless number */
    Plato::Scalar mViscocity = 1.0; /*!< viscocity dimensionless number */
    Plato::ScalarVector mGrNum; /*!< grashof dimensionless number */

public:
    VelocityPredictorResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>()),
         mGrNum("grashof number", mNumSpatialDims)
    {
        this->setDimensionlessProperties(aInputs);
        this->setNaturalBoundaryConditions(aInputs);
        this->setStabilizedNaturalBoundaryConditions(aInputs);
    }

    virtual ~VelocityPredictorResidual(){}

    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        Plato::ScalarVectorT<ConfigT>   tCellVolume("cell weight", tNumCells);
        Plato::ScalarVectorT<StrainT>   tDivPrevVel("divergence previous velocity", tNumCells);
        Plato::ScalarVectorT<PrevTempT> tPrevTempGP("previous temperature at Gauss point", tNumCells);

        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::ScalarArray3DT<StrainT> tStrainRate("cell strain rate", tNumCells, mNumSpatialDims, mNumSpatialDims);

        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultT>  tViscousForces("viscous forces", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultT>  tStabForces("stabilizing forces", tNumCells, mNumDofsPerCell);
        Plato::ScalarMultiVectorT<ResultT>  tInternalForces("internal forces", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PredVelT> tPredictorGP("predicted velocity at Gauss point", tNumCells, mNumSpatialDims);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode> tIntrplScalarField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // set input state worksets
        auto tTimeStepWS  = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
        auto tControlWS   = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tConfigWS    = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tPrevTempWS  = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));
        auto tPredictorWS = Plato::metadata<Plato::ScalarMultiVectorT<PredVelT>>(aWorkSets.get("current predictor"));

        // transfer member data to device
        auto tGrNum = mGrNum;
        auto tBuoyancy = mBuoyancy;
        auto tViscocity = mViscocity;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. add internal force contribution
            Plato::Fluids::strain_rate<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tPrevVelWS, tGradient, tStrainRate);
            Plato::Fluids::integrate_viscous_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tViscocity, tCellVolume, tGradient, tStrainRate, aResultWS);

            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            Plato::Fluids::calculate_advected_internal_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevVelWS, tPrevVelGP, tInternalForces);

            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tPrevTempWS, tPrevTempGP);
            Plato::Fluids::calculate_natural_convective_forces<mNumSpatialDims>
                (aCellOrdinal, tBuoyancy, tGrNum, tPrevTempGP, tInternalForces);

            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tInternalForces, aResultWS);

            // 2. add stabilizing internal force contribution
            Plato::Fluids::divergence<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevVelWS, tDivPrevVel);
            Plato::Fluids::integrate_stabilizing_vector_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tDivPrevVel, tGradient, tPrevVelGP, tInternalForces, tStabForces);
            Plato::Fluids::multiply_time_step<mNumNodesPerCell, mNumDofsPerNode>
                (aCellOrdinal, 0.5, tTimeStepWS, tStabForces);
            Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, 1.0, tStabForces, 1.0, aResultWS);

            // 3. apply time step to sum of internal and stabilizing forces, F = \Delta{t}\left( F_i^{int} + F_i^{stab} \right)
            Plato::Fluids::multiply_time_step<mNumNodesPerCell, mNumDofsPerNode>
                (aCellOrdinal, 1.0, tTimeStepWS, aResultWS);

            // 4. add inertial force contribution
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPredictorWS, tPredictorGP);
            Plato::Fluids::calculate_inertial_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPredictorGP, tPrevVelGP, aResultWS);
        }, "predictor residual");
    }

   void evaluateBoundary
   (const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets & aWorkSets,
    Plato::ScalarMultiVectorT<ResultT> & aResult)
   const override
   {
       // calculate boundary integral, which is defined as
       // \int_{\Gamma-\Gamma_t} N_u^a\left(\tau_{ij}n_j\right) d\Gamma
       auto tNumCells = aResult.extent(0);
       Plato::ScalarMultiVectorT<ResultT> tResultWS("deviatoric forces", tNumCells, mNumDofsPerCell);
       for(auto& tPair : mDeviatoricBCs)
       {
           tPair.second->operator()(aWorkSets, tResultWS);
       }

       // multiply force vector by the corresponding nodal time steps
       auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
       Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
       {
           Plato::Fluids::multiply_time_step<mNumNodesPerCell, mNumDofsPerNode>
               (aCellOrdinal, -1.0, tTimeStepWS, tResultWS);
           Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, 1.0, tResultWS, 1.0, aResult);
       }, "deviatoric traction forces");
   }

   void evaluatePrescribed
   (const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets & aWorkSets,
    Plato::ScalarMultiVectorT<ResultT> & aResult)
   const override
   {
       if( mPrescribedBCs != nullptr )
       {
           // set input worksets
           auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
           auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
           auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));

           // calculate deviatoric traction forces, which are defined as
           // \int_{\Gamma_t} N_u^a\left(t_i + p^{n-1}n_i\right) d\Gamma
           auto tNumCells = aResult.extent(0);
           Plato::ScalarMultiVectorT<ResultT> tResultWS("traction forces", tNumCells, mNumDofsPerCell);
           mPrescribedBCs->get( aSpatialModel, tPrevVelWS, tControlWS, tConfigWS, tResultWS); // traction forces
           for(auto& tPair : mPressureBCs)
           {
               tPair.second->operator()(aWorkSets, tResultWS); // pressure forces on traction side sets
           }

           // multiply force vector by the corresponding nodal time steps
           auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
           Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
           {
               Plato::Fluids::multiply_time_step<mNumNodesPerCell, mNumDofsPerNode>
                   (aCellOrdinal, -1.0, tTimeStepWS, tResultWS);
               Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, 1.0, tResultWS, 1.0, aResult);
           }, "prescribed traction forces");
       }
   }

private:
   void setViscosity
   (Teuchos::ParameterList & aInputs)
   {
       auto tHyperbolic = aInputs.sublist("Hyperbolic");
       auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
       auto tHeatTransfer = Plato::tolower(tTag);

       if(tHeatTransfer == "forced" || tHeatTransfer == "none")
       {
           auto tReNum = Plato::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolic);
           mViscocity = static_cast<Plato::Scalar>(1) / tReNum;
       }
       else if(tHeatTransfer == "natural")
       {
           mViscocity = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
       }
       else
       {
           THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
       }
   }

   void setGrashofNumber
   (Teuchos::ParameterList & aInputs)
   {
       auto tHyperbolic = aInputs.sublist("Hyperbolic");
       auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
       auto tHeatTransfer = Plato::tolower(tTag);
       auto tCalculateHeatTransfer = tHeatTransfer == "none" ? false : true;

        if(tCalculateHeatTransfer)
        {
            auto tGrNum = Plato::parse_parameter<Teuchos::Array<Plato::Scalar>>("Grashof Number", "Dimensionless Properties", tHyperbolic);
            if(tGrNum.size() != mNumSpatialDims)
            {
                THROWERR(std::string("'Grashof Number' array length should match the number of physical spatial dimensions. ")
                    + "Array length is '" + std::to_string(tGrNum.size()) + "' and the number of physical spatial dimensions is '"
                    + std::to_string(mNumSpatialDims) + "'.")
            }

            auto tLength = mGrNum.size();
            auto tHostGrNum = Kokkos::create_mirror(mGrNum);
            for(decltype(tLength) tDim = 0; tDim < tLength; tDim++)
            {
                tHostGrNum(tDim) = tGrNum[tDim];
            }
            Kokkos::deep_copy(mGrNum, tHostGrNum);
        }
        else
        {
            Plato::blas1::fill(0.0, mGrNum);
        }
   }

   void setBuoyancyConstant
   (Teuchos::ParameterList & aInputs)
   {
       auto tHyperbolic = aInputs.sublist("Hyperbolic");
       auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
       auto tHeatTransfer = Plato::tolower(tTag);

       if(tHeatTransfer == "forced")
       {
           auto tReNum = Plato::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolic);
           mBuoyancy = static_cast<Plato::Scalar>(1) / (tReNum*tReNum);
       }
       else if(tHeatTransfer == "natural")
       {
           auto tPrNum = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
           mBuoyancy = tPrNum*tPrNum;
       }
       else if(tHeatTransfer == "none")
       {
           mBuoyancy = 0;
       }
       else
       {
           THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
       }
   }

   void setDimensionlessProperties
   (Teuchos::ParameterList & aInputs)
   {
       if(aInputs.isSublist("Hyperbolic") == false)
       {
           THROWERR("'Hyperbolic' Parameter List is not defined.")
       }
       this->setViscosity(aInputs);
       this->setGrashofNumber(aInputs);
       this->setBuoyancyConstant(aInputs);
   }

   void setNaturalBoundaryConditions(Teuchos::ParameterList& aInputs)
   {
       if(aInputs.isSublist("Momentum Natural Boundary Conditions"))
       {
           auto tInputsNaturalBCs = aInputs.sublist("Momentum Natural Boundary Conditions");
           mPrescribedBCs = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>>(tInputsNaturalBCs);

           for (Teuchos::ParameterList::ConstIterator tItr = tInputsNaturalBCs.begin(); tItr != tInputsNaturalBCs.end(); ++tItr)
           {
               const Teuchos::ParameterEntry &tEntry = tInputsNaturalBCs.entry(tItr);
               if (!tEntry.isList())
               {
                   THROWERR(std::string("Error parsing Parameter List '") + tInputsNaturalBCs.name()
                       + "'. Parameter List block is not valid. Expects Parameter Lists only.")
               }

               const std::string &tName = tInputsNaturalBCs.name(tItr);
               if(tInputsNaturalBCs.isSublist(tName) == false)
               {
                   THROWERR(std::string("Error parsing Parameter List '") + tInputsNaturalBCs.name()
                       + "'. Parameter Sublist '" + tName.c_str() + "' is not defined.")
               }

               Teuchos::ParameterList &tSubList = tInputsNaturalBCs.sublist(tName);
               if(tSubList.isParameter("Sides") == false)
               {
                   THROWERR(std::string("Error parsing Parameter List '") + tSubList.name() + "'. 'Sides' keyword "
                       +"is not define in Parameter Sublist '" + tName.c_str() + "'. The 'Sides' keyword is used "
                       + "to define the 'side set' names where 'Natrual Boundary Conditions' are applied.")
               }
               const auto tSideSetName = tSubList.get<std::string>("Sides");
               auto tMapItr = mPressureBCs.find(tSideSetName);
               if(tMapItr == mPressureBCs.end())
               {
                   mPressureBCs[tSideSetName] = std::make_shared<PressureForces>(mSpatialDomain, tSideSetName);
               }
           }
       }
   }

   void setStabilizedNaturalBoundaryConditions(Teuchos::ParameterList& aInputs)
   {
       if(aInputs.isSublist("Balancing Momentum Natural Boundary Conditions"))
       {
           auto tInputsNaturalBCs = aInputs.sublist("Balancing Momentum Natural Boundary Conditions");
           for (Teuchos::ParameterList::ConstIterator tItr = tInputsNaturalBCs.begin(); tItr != tInputsNaturalBCs.end(); ++tItr)
           {
               const Teuchos::ParameterEntry &tEntry = tInputsNaturalBCs.entry(tItr);
               if (!tEntry.isList())
               {
                   THROWERR(std::string("Error parsing Parameter List '") + tInputsNaturalBCs.name()
                       + "'. Parameter List block is not valid. Expects Parameter Lists only.")
               }

               const std::string &tName = tInputsNaturalBCs.name(tItr);
               if(tInputsNaturalBCs.isSublist(tName) == false)
               {
                   THROWERR(std::string("Error parsing Parameter List '") + tInputsNaturalBCs.name()
                       + "'. Parameter Sublist '" + tName.c_str() + "' is not defined.")
               }

               Teuchos::ParameterList &tSubList = tInputsNaturalBCs.sublist(tName);
               if(tSubList.isParameter("Sides") == false)
               {
                   THROWERR(std::string("Error parsing Parameter List '") + tSubList.name() + "'. 'Sides' keyword "
                       +"is not define in Parameter Sublist '" + tName.c_str() + "'. The 'Sides' keyword is used "
                       + "to define the 'side set' names where 'Natural Boundary Conditions' are applied.")
               }
               const auto tSideSetName = tSubList.get<std::string>("Sides");
               auto tMapItr = mDeviatoricBCs.find(tSideSetName);
               if(tMapItr == mDeviatoricBCs.end())
               {
                   mDeviatoricBCs[tSideSetName] = std::make_shared<DeviatoricForces>(mSpatialDomain, aInputs, tSideSetName);
               }
           }
       }
       else
       {
           mDeviatoricBCs["automated"] = std::make_shared<DeviatoricForces>(mSpatialDomain, aInputs);
       }
   }
};
// class VelocityPredictorResidual

namespace Brinkman
{

template<typename PhysicsT, typename EvaluationT>
class VelocityPredictorResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumTempDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell;   /*!< number of configuration degrees of freedom per cell */

    // set local ad types
    using ResultT   = typename EvaluationT::ResultScalarType;
    using ConfigT   = typename EvaluationT::ConfigScalarType;
    using ControlT  = typename EvaluationT::ControlScalarType;
    using PrevVelT  = typename EvaluationT::PreviousMomentumScalarType;
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType;
    using PredVelT  = typename EvaluationT::MomentumPredictorScalarType;
    using StrainT   = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>;

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature rule evaluator */

    // set local type names
    using PressureForces   = Plato::Fluids::PressureSurfaceForces<PhysicsT, EvaluationT>;
    using DeviatoricForces = Plato::Fluids::DeviatoricSurfaceForces<PhysicsT, EvaluationT>;

    // set external force evaluators
    std::unordered_map<std::string, std::shared_ptr<PressureForces>>     mPressureBCs;   /*!< prescribed pressure forces */
    std::unordered_map<std::string, std::shared_ptr<DeviatoricForces>>   mDeviatoricBCs; /*!< stabilized deviatoric boundary forces */
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mPrescribedBCs; /*!< prescribed boundary conditions, e.g. tractions */

    // set member scalar data
    Plato::Scalar mBuoyancy = 1.0; /*!< buoyancy dimensionless number */
    Plato::Scalar mViscocity = 1.0; /*!< viscocity dimensionless number */
    Plato::Scalar mPermeability = 1.0; /*!< permeability dimensionless number */
    Plato::Scalar mBrinkmanConvexityParam = 0.5;  /*!< brinkman model convexity parameter */
    Plato::ScalarVector mGrNum; /*!< grashof dimensionless number */

public:
    VelocityPredictorResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>()),
         mGrNum("grashof number", mNumSpatialDims)
    {
        this->setParameters(aInputs);
        this->setNaturalBoundaryConditions(aInputs);
        this->setStabilizedNaturalBoundaryConditions(aInputs);
    }

    virtual ~VelocityPredictorResidual(){}

    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        Plato::ScalarVectorT<ConfigT>   tCellVolume("cell weight", tNumCells);
        Plato::ScalarVectorT<StrainT>   tDivPrevVel("divergence previous velocity", tNumCells);
        Plato::ScalarVectorT<PrevTempT> tPrevTempGP("previous temperature at Gauss point", tNumCells);

        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::ScalarArray3DT<StrainT> tStrainRate("cell strain rate", tNumCells, mNumSpatialDims, mNumSpatialDims);

        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultT>  tViscousForces("viscous forces", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultT>  tStabForces("stabilizing forces", tNumCells, mNumDofsPerCell);
        Plato::ScalarMultiVectorT<ResultT>  tInternalForces("internal forces", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PredVelT> tPredictorGP("predicted velocity at Gauss point", tNumCells, mNumSpatialDims);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode> tIntrplScalarField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // set input state worksets
        auto tTimeStepWS  = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
        auto tControlWS   = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tConfigWS    = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tPrevTempWS  = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));
        auto tPredictorWS = Plato::metadata<Plato::ScalarMultiVectorT<PredVelT>>(aWorkSets.get("current predictor"));

        // transfer member data to device
        auto tGrNum = mGrNum;
        auto tBuoyancy = mBuoyancy;
        auto tViscocity = mViscocity;
        auto tPermeability = mPermeability;
        auto tBrinkmanConvexityParam = mBrinkmanConvexityParam;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. add internal force contribution
            Plato::Fluids::strain_rate<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tPrevVelWS, tGradient, tStrainRate);
            Plato::Fluids::integrate_viscous_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tViscocity, tCellVolume, tGradient, tStrainRate, aResultWS);

            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            Plato::Fluids::calculate_advected_internal_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevVelWS, tPrevVelGP, tInternalForces);

            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tPrevTempWS, tPrevTempGP);
            Plato::Fluids::calculate_natural_convective_forces<mNumSpatialDims>
                (aCellOrdinal, tBuoyancy, tGrNum, tPrevTempGP, tInternalForces);

            ControlT tPenalizedPermeability = Plato::Fluids::brinkman_penalization<mNumNodesPerCell>
                (aCellOrdinal, tPermeability, tBrinkmanConvexityParam, tControlWS);
            Plato::Fluids::calculate_brinkman_forces<mNumSpatialDims>
                (aCellOrdinal, tPenalizedPermeability, tPrevVelGP, tInternalForces);

            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tInternalForces, aResultWS);

            // 2. add stabilizing internal force contribution
            Plato::Fluids::divergence<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevVelWS, tDivPrevVel);
            Plato::Fluids::integrate_stabilizing_vector_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tDivPrevVel, tGradient, tPrevVelGP, tInternalForces, tStabForces);
            Plato::Fluids::multiply_time_step<mNumNodesPerCell, mNumDofsPerNode>
                (aCellOrdinal, 0.5, tTimeStepWS, tStabForces);
            Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, 1.0, tStabForces, 1.0, aResultWS);

            // 3. apply time step to sum of internal and stabilizing forces, F = \Delta{t}\left( F_i^{int} + F_i^{stab} \right)
            Plato::Fluids::multiply_time_step<mNumNodesPerCell, mNumDofsPerNode>
                (aCellOrdinal, 1.0, tTimeStepWS, aResultWS);

            // 4. add inertial force contribution
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPredictorWS, tPredictorGP);
            Plato::Fluids::calculate_inertial_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPredictorGP, tPrevVelGP, aResultWS);
        }, "predictor residual");
    }

   void evaluateBoundary
   (const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets & aWorkSets,
    Plato::ScalarMultiVectorT<ResultT> & aResult)
   const override
   {
       // calculate boundary integral, which is defined as
       // \int_{\Gamma-\Gamma_t} N_u^a\left(\tau_{ij}n_j\right) d\Gamma
       auto tNumCells = aResult.extent(0);
       Plato::ScalarMultiVectorT<ResultT> tResultWS("deviatoric forces", tNumCells, mNumDofsPerCell);
       for(auto& tPair : mDeviatoricBCs)
       {
           tPair.second->operator()(aWorkSets, tResultWS);
       }

       // multiply force vector by the corresponding nodal time steps
       auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
       Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
       {
           Plato::Fluids::multiply_time_step<mNumNodesPerCell, mNumDofsPerNode>
               (aCellOrdinal, -1.0, tTimeStepWS, tResultWS);
           Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, 1.0, tResultWS, 1.0, aResult);
       }, "deviatoric traction forces");
   }

   void evaluatePrescribed
   (const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets & aWorkSets,
    Plato::ScalarMultiVectorT<ResultT> & aResult)
   const override
   {
       if( mPrescribedBCs != nullptr )
       {
           // set input worksets
           auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
           auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
           auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));

           // calculate deviatoric traction forces, which are defined as
           // \int_{\Gamma_t} N_u^a\left(t_i + p^{n-1}n_i\right) d\Gamma
           auto tNumCells = aResult.extent(0);
           Plato::ScalarMultiVectorT<ResultT> tResultWS("traction forces", tNumCells, mNumDofsPerCell);
           mPrescribedBCs->get( aSpatialModel, tPrevVelWS, tControlWS, tConfigWS, tResultWS); // traction forces
           for(auto& tPair : mPressureBCs)
           {
               tPair.second->operator()(aWorkSets, tResultWS); // pressure forces on traction side sets
           }

           // multiply force vector by the corresponding nodal time steps
           auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
           Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
           {
               Plato::Fluids::multiply_time_step<mNumNodesPerCell, mNumDofsPerNode>
                   (aCellOrdinal, -1.0, tTimeStepWS, tResultWS);
               Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, 1.0, tResultWS, 1.0, aResult);
           }, "prescribed traction forces");
       }
   }

private:
   void setParameters
   (Teuchos::ParameterList & aInputs)
   {
       if(aInputs.isSublist("Hyperbolic") == false)
       {
           THROWERR("'Hyperbolic' Parameter List is not defined.")
       }
       this->setDimensionlessProperties(aInputs);
       auto tHyperbolic = aInputs.sublist("Hyperbolic");
       if(tHyperbolic.isSublist("Momentum Conservation"))
       {
           auto tMomentumParamList = tHyperbolic.sublist("Momentum Conservation");
           this->setPenaltyModel(tMomentumParamList);
       }
   }

   void setPenaltyModel
   (Teuchos::ParameterList & aInputs)
   {
        if (aInputs.isSublist("Penalty Function"))
        {
            auto tPenaltyFuncList = aInputs.sublist("Penalty Function");
            mBrinkmanConvexityParam = tPenaltyFuncList.get<Plato::Scalar>("Brinkman Convexity Parameter", 0.5);
        }
   }

   void setPermeability
   (Teuchos::ParameterList & aInputs)
   {
       auto tHyperbolic = aInputs.sublist("Hyperbolic");
       auto tDaNum = Plato::parse_parameter<Plato::Scalar>("Darcy Number", "Dimensionless Properties", tHyperbolic);
       auto tPrNum = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
       mPermeability = tPrNum / tDaNum;
   }

   void setViscosity
   (Teuchos::ParameterList & aInputs)
   {
       auto tHyperbolic = aInputs.sublist("Hyperbolic");
       auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
       auto tHeatTransfer = Plato::tolower(tTag);

       if(tHeatTransfer == "forced" || tHeatTransfer == "none")
       {
           auto tReNum = Plato::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolic);
           mViscocity = static_cast<Plato::Scalar>(1) / tReNum;
       }
       else if(tHeatTransfer == "natural")
       {
           mViscocity = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
       }
       else
       {
           THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
       }
   }

   void setGrashofNumber
   (Teuchos::ParameterList & aInputs)
   {
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
        auto tHeatTransfer = Plato::tolower(tTag);
        auto tCalculateHeatTransfer = tHeatTransfer == "none" ? false : true;

        if(tCalculateHeatTransfer)
        {
            auto tGrNum = Plato::parse_parameter<Teuchos::Array<Plato::Scalar>>("Grashof Number", "Dimensionless Properties", tHyperbolic);
            if(tGrNum.size() != mNumSpatialDims)
            {
                THROWERR(std::string("'Grashof Number' array length should match the number of physical spatial dimensions. ")
                    + "Array length is '" + std::to_string(tGrNum.size()) + "' and the number of physical spatial dimensions is '"
                    + std::to_string(mNumSpatialDims) + "'.")
            }

            auto tLength = mGrNum.size();
            auto tHostGrNum = Kokkos::create_mirror(mGrNum);
            for(decltype(tLength) tDim = 0; tDim < tLength; tDim++)
            {
                tHostGrNum(tDim) = tGrNum[tDim];
            }
            Kokkos::deep_copy(mGrNum, tHostGrNum);
        }
        else
        {
            Plato::blas1::fill(0.0, mGrNum);
        }
   }

   void setBuoyancyConstant
   (Teuchos::ParameterList & aInputs)
   {
       auto tHyperbolic = aInputs.sublist("Hyperbolic");
       auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
       auto tHeatTransfer = Plato::tolower(tTag);

       if(tHeatTransfer == "forced")
       {
           auto tReNum = Plato::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolic);
           mBuoyancy = static_cast<Plato::Scalar>(1) / (tReNum*tReNum);
       }
       else if(tHeatTransfer == "natural")
       {
           auto tPrNum = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
           mBuoyancy = tPrNum*tPrNum;
       }
       else if(tHeatTransfer == "none")
       {
           mBuoyancy = 0;
       }
       else
       {
           THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
       }
   }

   void setDimensionlessProperties
   (Teuchos::ParameterList & aInputs)
   {
       if(aInputs.isSublist("Hyperbolic") == false)
       {
           THROWERR("'Hyperbolic' Parameter List is not defined.")
       }
       this->setViscosity(aInputs);
       this->setPermeability(aInputs);
       this->setGrashofNumber(aInputs);
       this->setBuoyancyConstant(aInputs);
   }

    void setNaturalBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Momentum Natural Boundary Conditions"))
        {
            auto tInputsNaturalBCs = aInputs.sublist("Momentum Natural Boundary Conditions");
            mPrescribedBCs = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>>(tInputsNaturalBCs);

            for (Teuchos::ParameterList::ConstIterator tItr = tInputsNaturalBCs.begin(); tItr != tInputsNaturalBCs.end(); ++tItr)
            {
                const Teuchos::ParameterEntry &tEntry = tInputsNaturalBCs.entry(tItr);
                if (!tEntry.isList())
                {
                    THROWERR(std::string("Error parsing Parameter List '") + tInputsNaturalBCs.name() 
                        + "'. Parameter List block is not valid. Expects Parameter Lists only.")
                }

                const std::string &tName = tInputsNaturalBCs.name(tItr);
                if(tInputsNaturalBCs.isSublist(tName) == false)
                {
                    THROWERR(std::string("Error parsing Parameter List '") + tInputsNaturalBCs.name() 
                        + "'. Parameter Sublist '" + tName.c_str() + "' is not defined.")
                }

                Teuchos::ParameterList &tSubList = tInputsNaturalBCs.sublist(tName);
                if(tSubList.isParameter("Sides") == false)
                {
                    THROWERR(std::string("Error parsing Parameter List '") + tSubList.name() + "'. 'Sides' keyword " 
                        +"is not define in Parameter Sublist '" + tName.c_str() + "'. The 'Sides' keyword is used " 
                        + "to define the 'side set' names where 'Natrual Boundary Conditions' are applied.")
                }
                const auto tSideSetName = tSubList.get<std::string>("Sides");
                auto tMapItr = mPressureBCs.find(tSideSetName);
                if(tMapItr == mPressureBCs.end())
                {
                    mPressureBCs[tSideSetName] = std::make_shared<PressureForces>(mSpatialDomain, tSideSetName);
                }
            }
        }
    }

    void setStabilizedNaturalBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Balancing Momentum Natural Boundary Conditions"))
        {
            auto tInputsNaturalBCs = aInputs.sublist("Balancing Momentum Natural Boundary Conditions");
            for (Teuchos::ParameterList::ConstIterator tItr = tInputsNaturalBCs.begin(); tItr != tInputsNaturalBCs.end(); ++tItr)
            {
                const Teuchos::ParameterEntry &tEntry = tInputsNaturalBCs.entry(tItr);
                if (!tEntry.isList())
                {
                    THROWERR(std::string("Error parsing Parameter List '") + tInputsNaturalBCs.name()
                        + "'. Parameter List block is not valid. Expects Parameter Lists only.")
                }

                const std::string &tName = tInputsNaturalBCs.name(tItr);
                if(tInputsNaturalBCs.isSublist(tName) == false)
                {
                    THROWERR(std::string("Error parsing Parameter List '") + tInputsNaturalBCs.name()
                        + "'. Parameter Sublist '" + tName.c_str() + "' is not defined.")
                }

                Teuchos::ParameterList &tSubList = tInputsNaturalBCs.sublist(tName);
                if(tSubList.isParameter("Sides") == false)
                {
                    THROWERR(std::string("Error parsing Parameter List '") + tSubList.name() + "'. 'Sides' keyword "
                        +"is not define in Parameter Sublist '" + tName.c_str() + "'. The 'Sides' keyword is used "
                        + "to define the 'side set' names where 'Natural Boundary Conditions' are applied.")
                }
                const auto tSideSetName = tSubList.get<std::string>("Sides");
                auto tMapItr = mDeviatoricBCs.find(tSideSetName);
                if(tMapItr == mDeviatoricBCs.end())
                {
                    mDeviatoricBCs[tSideSetName] = std::make_shared<DeviatoricForces>(mSpatialDomain, aInputs, tSideSetName);
                }
            }
        }
        else
        {
            mDeviatoricBCs["automated"] = std::make_shared<DeviatoricForces>(mSpatialDomain, aInputs);
        }
    }
};
// class VelocityPredictorResidual

}
// namespace Brinkman

// calculate pressure gradient integral, which is defined as \frac{\partial p^{n+\theta_2}}{\partial x_i}, where
// p^{n+\theta_2} = \frac{\partial p^{n-1}}{\partial x_i} + \theta_2\frac{\partial\delta{p}}{\partial x_i}
// and \delta{p} = p^n - p^{n-1}
template<Plato::OrdinalType NumNodes,
         Plato::OrdinalType SpaceDim,
         typename ConfigT,
         typename CurPressT,
         typename PrevPressT,
         typename PressGradT>
DEVICE_TYPE inline void
calculate_pressure_gradient
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar & aTheta,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<CurPressT> & aCurPress,
 const Plato::ScalarMultiVectorT<PrevPressT> & aPrevPress,
 const Plato::ScalarMultiVectorT<PressGradT> & aPressGrad)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aPressGrad(aCellOrdinal, tDim) += ( (static_cast<Plato::Scalar>(1.0) - aTheta)
                * aGradient(aCellOrdinal, tNode, tDim) * aPrevPress(aCellOrdinal, tNode) )
                + ( aTheta * aGradient(aCellOrdinal, tNode, tDim) * aCurPress(aCellOrdinal, tNode) );
        }
    }
}


template<typename PhysicsT, typename EvaluationT>
class VelocityCorrectorResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode    = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell    = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */
    static constexpr auto mNumNodesPerCell   = PhysicsT::SimplexT::mNumNodesPerCell; /*!< number of nodes per cell */
    static constexpr auto mNumSpatialDims    = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */

    using ResultT    = typename EvaluationT::ResultScalarType;
    using ConfigT    = typename EvaluationT::ConfigScalarType;
    using CurVelT    = typename EvaluationT::CurrentMomentumScalarType;
    using CurPressT  = typename EvaluationT::CurrentMassScalarType;
    using PrevVelT   = typename EvaluationT::PreviousMomentumScalarType;
    using PrevPressT = typename EvaluationT::PreviousMassScalarType;
    using PredictorT = typename EvaluationT::MomentumPredictorScalarType;
    using GradientT  = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>;

    Plato::DataMap& mDataMap;                   /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< integration rule */

    Plato::Scalar mThetaTwo = 0.0;

public:
    VelocityCorrectorResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mThetaTwo = tTimeIntegration.get<Plato::Scalar>("Artificial Damping Two", 0.0);
        }
    }

    virtual ~VelocityCorrectorResidual(){}

    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        // set local data structures
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
        Plato::ScalarVectorT<GradientT> tDivPrevVel("divergence previous velocity", tNumCells);

        Plato::ScalarMultiVectorT<ResultT> tStabForces("stabilizing forces", tNumCells, mNumDofsPerCell);
        Plato::ScalarMultiVectorT<ResultT> tInternalForces("internal forces", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<CurVelT> tCurVelGP("current velocity at Gauss point", tNumCells, mNumDofsPerNode);
        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity at Gauss point", tNumCells, mNumDofsPerNode);
        Plato::ScalarMultiVectorT<PredictorT> tPredictorGP("velocity predictor at Gauss point", tNumCells, mNumDofsPerNode);

        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        // set input state worksets
        auto tTimeStepWS  = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
        auto tConfigWS    = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurVelWS    = Plato::metadata<Plato::ScalarMultiVectorT<CurVelT>>(aWorkSets.get("current velocity"));
        auto tPrevVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tCurPressWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurPressT>>(aWorkSets.get("current pressure"));
        auto tPrevPressWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevPressT>>(aWorkSets.get("previous pressure"));
        auto tPredictorWS = Plato::metadata<Plato::ScalarMultiVectorT<PredictorT>>(aWorkSets.get("current predictor"));

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        auto tThetaTwo = mThetaTwo;
        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. calculate internal forces
            Plato::Fluids::calculate_pressure_gradient<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tThetaTwo, tGradient, tCurPressWS, tPrevPressWS, tInternalForces);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tInternalForces, aResultWS);

            // 2. calculate stabilizing forces
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            Plato::Fluids::divergence<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevVelWS, tDivPrevVel);
            Plato::Fluids::integrate_stabilizing_vector_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tDivPrevVel, tGradient, tPrevVelGP, tInternalForces, tStabForces);
            Plato::Fluids::multiply_time_step<mNumNodesPerCell, mNumDofsPerNode>
                (aCellOrdinal, 0.5, tTimeStepWS, tStabForces);
            Plato::blas1::update<mNumNodesPerCell>(aCellOrdinal, 1.0, tStabForces, 1.0, aResultWS);

            // 3. apply time step to sum of internal and stabilizing forces, i.e.
            // F = \Delta{t} * \left( F_i^{int} + F_i^{stab} \right)
            Plato::Fluids::multiply_time_step<mNumNodesPerCell, mNumDofsPerNode>
                (aCellOrdinal, 1.0, tTimeStepWS, aResultWS);

            // 4. add inertial force contribution
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurVelWS, tCurVelGP);
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPredictorWS, tPredictorGP);
            Plato::Fluids::calculate_inertial_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tCurVelGP, tPredictorGP, aResultWS);
        }, "velocity increment residual, i.e. 'corrector step'");
    }

    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return; /* boundary integral equals zero */ }

    void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return; /* prescribed force integral equals zero */ }
};
// class VelocityCorrectorResidual



template<Plato::OrdinalType NumNodes,
         Plato::OrdinalType SpaceDim,
         typename ConfigT,
         typename PrevVelT,
         typename PrevTempT,
         typename ResultT>
DEVICE_TYPE inline void
calculate_convective_forces
(const Plato::OrdinalType & aCellOrdinals,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarMultiVectorT<PrevTempT> & aPrevTemp,
 const Plato::ScalarVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aResult(aCellOrdinals) += aPrevVelGP(aCellOrdinals, tDim)
                * ( aGradient(aCellOrdinals, tNode, tDim) * aPrevTemp(aCellOrdinals, tNode) );
        }
    }
}

template<Plato::OrdinalType NumNodes,
         typename ConfigT,
         typename SourceT,
         typename ResultT>
DEVICE_TYPE inline void
integrate_scalar_field
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarVectorT<SourceT> & aSource,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        aResult(aCellOrdinal, tNode) += aBasisFunctions(tNode) * aSource(aCellOrdinal) * aCellVolume(aCellOrdinal);
    }
}

template<Plato::OrdinalType NumNodes,
         Plato::OrdinalType SpaceDim,
         typename ConfigT,
         typename SourceT,
         typename ResultT>
DEVICE_TYPE inline void
calculate_flux_divergence
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<SourceT> & aFlux,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aResult(aCellOrdinal, tNode) +=
                ( aGradient(aCellOrdinal, tNode, tDim) * aFlux(aCellOrdinal, tDim) ) * aCellVolume(aCellOrdinal);
        }
    }
}

template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename FluxT,
 typename ConfigT,
 typename StateT>
DEVICE_TYPE inline void
calculate_flux
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<StateT> & aState,
 const Plato::ScalarMultiVectorT<FluxT> & aThermalFlux)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aThermalFlux(aCellOrdinal, tDim) += aGradient(aCellOrdinal, tNode, tDim) * aState(aCellOrdinal, tNode);
        }
    }
}

// TODO: FIX THIS, IT SHOULD BE DIMENSIONLESS
template<Plato::OrdinalType NumNodesPerCell,
         typename ControlT>
DEVICE_TYPE inline ControlT
penalize_thermal_diffusivity
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar & aFluidThermalDiff,
 const Plato::Scalar & aSolidThermalDiff,
 const Plato::Scalar & aPenaltyExponent,
 const Plato::ScalarMultiVectorT<ControlT> & aControl)
{
    ControlT tDensity = Plato::cell_density<NumNodesPerCell>(aCellOrdinal, aControl);
    ControlT tPenalizedDensity = pow(tDensity, aPenaltyExponent);
    auto tSolidOverFluidThermalDiff = aSolidThermalDiff / aFluidThermalDiff;
    ControlT tPenalizedThermalDiff =
        tSolidOverFluidThermalDiff + ( (static_cast<Plato::Scalar>(1.0) - tSolidOverFluidThermalDiff) * tPenalizedDensity);
    return tPenalizedThermalDiff;
}

// TODO: FIX THIS, PUT IT IN TERMS OF APPLY_WEIGHT CLASS
template<Plato::OrdinalType NumNodesPerCell,
         typename ControlT>
DEVICE_TYPE inline ControlT
penalize_heat_source_constant
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar & aHeatSourceConstant,
 const Plato::Scalar & aPenaltyExponent,
 const Plato::ScalarMultiVectorT<ControlT> & aControl)
{
    ControlT tDensity = Plato::cell_density<NumNodesPerCell>(aCellOrdinal, aControl);
    ControlT tPenalizedDensity = pow(tDensity, aPenaltyExponent);
    auto tPenalizedProperty = (static_cast<Plato::Scalar>(1) - tPenalizedDensity) * aHeatSourceConstant;
    return tPenalizedProperty;
}

template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename DivVelT,
 typename ResultT,
 typename ConfigT,
 typename PrevVelT,
 typename StabForceT>
DEVICE_TYPE inline void
integrate_stabilizing_scalar_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarVectorT<DivVelT> & aDivPrevVel,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarVectorT<StabForceT> & aStabForce,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
 {
    for (Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for (Plato::OrdinalType tDimI = 0; tDimI < SpaceDim; tDimI++)
        {
            aResult(aCellOrdinal, tNode) += aStabForce(aCellOrdinal) * aCellVolume(aCellOrdinal)
                * ( aGradient(aCellOrdinal, tNode, tDimI) * aPrevVelGP(aCellOrdinal, tDimI) );
        }
        aResult(aCellOrdinal, tNode) += aStabForce(aCellOrdinal) * aCellVolume(aCellOrdinal)
            * ( aBasisFunctions(tNode) * aDivPrevVel(aCellOrdinal) );
    }
 }

template
<Plato::OrdinalType NumNodes,
 typename ResultT,
 typename ConfigT,
 typename CurStateT,
 typename PrevStateT>
DEVICE_TYPE inline void
calculate_inertial_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarVectorT<CurStateT> & aCurState,
 const Plato::ScalarVectorT<PrevStateT> & aPrevState,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for (Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        aResult(aCellOrdinal, tNode) +=
            aCellVolume(aCellOrdinal) * aBasisFunctions(tNode) * ( aCurState(aCellOrdinal) - aPrevState(aCellOrdinal) );
    }
}



template<typename PhysicsT, typename EvaluationT>
class TemperatureResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumTempDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell;   /*!< number of configuration degrees of freedom per cell */

    // set local ad type
    using ResultT   = typename EvaluationT::ResultScalarType;
    using ConfigT   = typename EvaluationT::ConfigScalarType;
    using ControlT  = typename EvaluationT::ControlScalarType;
    using CurTempT  = typename EvaluationT::CurrentEnergyScalarType;
    using PrevVelT  = typename EvaluationT::PreviousMomentumScalarType;
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType;
    using DivergenceT   = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>;

    Plato::DataMap& mDataMap;                   /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */

    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< integration rule */
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mHeatFlux; /*!< heat flux evaluator */

    Plato::Scalar mHeatSourceConstant         = 0.0;
    Plato::Scalar mCharacteristicLength       = 1.0;
    Plato::Scalar mReferenceTemperature       = 1.0;
    Plato::Scalar mEffectiveConductivity      = 1.0;
    Plato::Scalar mFluidThermalConductivity   = 1.0;

public:
    TemperatureResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setSourceTerm(aInputs);
        this->setThermalProperties(aInputs);
        this->setDimensionlessProperties(aInputs);
        this->setNaturalBoundaryConditions(aInputs);
    }

    virtual ~TemperatureResidual(){}

    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        // set constant heat source
        Plato::ScalarVectorT<ResultT> tHeatSource("heat source", tNumCells);
        Plato::blas1::fill(mHeatSourceConstant, tHeatSource);

        // set local data
        Plato::ScalarVectorT<ConfigT>     tCellVolume("cell weight", tNumCells);
        Plato::ScalarVectorT<ResultT>     tInternalForces("internal forces", tNumCells);
        Plato::ScalarVectorT<CurTempT>    tCurTempGP("current temperature at Gauss points", tNumCells);
        Plato::ScalarVectorT<PrevTempT>   tPrevTempGP("previous temperature at Gauss points", tNumCells);
        Plato::ScalarVectorT<DivergenceT> tDivPrevVel("divergence previous velocity", tNumCells);

        Plato::ScalarMultiVectorT<ResultT> tThermalFlux("thermal flux", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultT> tStabForces("stabilizing forces", tNumCells, mNumTempDofsPerCell);
        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity at Gauss points", tNumCells, mNumVelDofsPerNode);

        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode> tIntrplScalarField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // set input state worksets
        auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
        auto tControlWS  = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tConfigWS   = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS  = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tCurTempWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurTempT>>(aWorkSets.get("current temperature"));
        auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));

        // transfer member data to device
        auto tRefTemp          = mReferenceTemperature;
        auto tCharLength       = mCharacteristicLength;
        auto tEffConductivity  = mEffectiveConductivity;
        auto tFluidThermalCond = mFluidThermalConductivity;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. calculate internal forces
            Plato::Fluids::calculate_flux<mNumNodesPerCell,mNumSpatialDims>(aCellOrdinal, tGradient, tPrevTempWS, tThermalFlux);
            Plato::blas1::scale<mNumSpatialDims>(aCellOrdinal, tEffConductivity, tThermalFlux);
            Plato::Fluids::calculate_flux_divergence<mNumNodesPerCell,mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tThermalFlux, aResultWS);

            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            Plato::Fluids::calculate_convective_forces<mNumNodesPerCell,mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevVelGP, tPrevTempWS, tInternalForces);

            auto tHeatSrcConst = ( tCharLength * tCharLength ) / (tFluidThermalCond * tRefTemp);
            tInternalForces(aCellOrdinal) -= tHeatSrcConst * tHeatSource(aCellOrdinal);

            Plato::Fluids::integrate_scalar_field<mNumNodesPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tInternalForces, aResultWS);

            // 2. calculate stabilizing forces
            Plato::Fluids::divergence<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevVelWS, tDivPrevVel);
            Plato::Fluids::integrate_stabilizing_scalar_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tDivPrevVel, tGradient, tPrevVelGP, tInternalForces, tStabForces);
            Plato::Fluids::multiply_time_step<mNumNodesPerCell, mNumDofsPerNode>
                (aCellOrdinal, 0.5, tTimeStepWS, tStabForces);
            Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, 1.0, tStabForces, 1.0, aResultWS);

            // 3. apply time step to sum of internal and stabilizing forces, F = \Delta{t}\left( F_i^{int} + F_i^{stab} \right)
            Plato::Fluids::multiply_time_step<mNumNodesPerCell, mNumDofsPerNode>
                (aCellOrdinal, 1.0, tTimeStepWS, aResultWS);

            // 4. add inertial force contribution
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurTempWS, tCurTempGP);
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevTempWS, tPrevTempGP);
            Plato::Fluids::calculate_inertial_forces<mNumNodesPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tCurTempGP, tPrevTempGP, aResultWS);
        }, "conservation of energy internal forces");
    }

    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return; /* boundary integral equates zero */ }

    void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    {
        if( mHeatFlux != nullptr )
        {
            // set input state worksets
            auto tConfigWS   = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
            auto tControlWS  = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
            auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));

            // evaluate prescribed flux
            auto tNumCells = aResult.extent(0);
            Plato::ScalarMultiVectorT<ResultT> tResultWS("heat flux", tNumCells, mNumDofsPerCell);
            mHeatFlux->get( aSpatialModel, tPrevTempWS, tControlWS, tConfigWS, tResultWS );

            auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                Plato::Fluids::multiply_time_step<mNumNodesPerCell, mNumDofsPerNode>
                    (aCellOrdinal, -1.0, tTimeStepWS, tResultWS);
                Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, 1.0, tResultWS, 1.0, aResult);
            }, "heat flux contribution");
        }
    }

private:
    void setSourceTerm
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Heat Source"))
        {
            mHeatSourceConstant = aInputs.sublist("Heat Source").get<Plato::Scalar>("Constant", 0.0);
        }
    }

    void setThermalProperties
    (Teuchos::ParameterList & aInputs)
    {
        auto tMaterialName = mSpatialDomain.getMaterialName();
        Plato::is_material_defined(tMaterialName, aInputs);
        auto tMaterial = aInputs.sublist("Material Models").sublist(tMaterialName);
        auto tThermalPropBlock = std::string("Thermal Properties");
        mReferenceTemperature = Plato::parse_parameter<Plato::Scalar>("Reference Temperature", tThermalPropBlock, tMaterial);
        mFluidThermalConductivity = Plato::parse_parameter<Plato::Scalar>("Fluid Thermal Conductivity", tThermalPropBlock, tMaterial);
    }

    void setCharacteristicLength
    (Teuchos::ParameterList & aInputs)
    {
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        mCharacteristicLength = Plato::parse_parameter<Plato::Scalar>("Characteristic Length", "Dimensionless Properties", tHyperbolic);
    }

    void setEffectiveConductivity
    (Teuchos::ParameterList & aInputs)
    {
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
        auto tHeatTransfer = Plato::tolower(tTag);

        if(tHeatTransfer == "forced" || tHeatTransfer == "none")
        {
            auto tPrNum = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
            auto tReNum = Plato::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolic);
            mEffectiveConductivity = static_cast<Plato::Scalar>(1) / (tReNum*tPrNum);
        }
        else if(tHeatTransfer == "natural")
        {
            mEffectiveConductivity = 1.0;
        }
        else
        {
            THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
        }
    }

    void setDimensionlessProperties
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        this->setCharacteristicLength(aInputs);
        this->setEffectiveConductivity(aInputs);
    }

    void setNaturalBoundaryConditions
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Energy Natural Boundary Conditions"))
        {
            auto tSublist = aInputs.sublist("Energy Natural Boundary Conditions");
            mHeatFlux = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>>(tSublist);
        }
    }
};
// class TemperatureResidual


namespace SIMP
{

template<typename PhysicsT, typename EvaluationT>
class TemperatureResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumTempDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell;   /*!< number of configuration degrees of freedom per cell */

    // set local ad type
    using ResultT   = typename EvaluationT::ResultScalarType;
    using ConfigT   = typename EvaluationT::ConfigScalarType;
    using ControlT  = typename EvaluationT::ControlScalarType;
    using CurTempT  = typename EvaluationT::CurrentEnergyScalarType;
    using PrevVelT  = typename EvaluationT::PreviousMomentumScalarType;
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType;
    using DivergenceT   = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>;

    Plato::DataMap& mDataMap;                   /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */

    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< integration rule */
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mHeatFlux; /*!< heat flux evaluator */

    Plato::Scalar mHeatSourceConstant         = 0.0;
    Plato::Scalar mCharacteristicLength       = 1.0;
    Plato::Scalar mReferenceTemperature       = 1.0;
    Plato::Scalar mEffectiveConductivity      = 1.0;
    Plato::Scalar mSolidThermalDiffusivity    = 1.0;
    Plato::Scalar mFluidThermalDiffusivity    = 1.0;
    Plato::Scalar mFluidThermalConductivity   = 1.0;
    Plato::Scalar mHeatSourcePenaltyExponent  = 3.0;
    Plato::Scalar mThermalDiffPenaltyExponent = 3.0;

public:
    TemperatureResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setSourceTerm(aInputs);
        this->setThermalProperties(aInputs);
        this->setPenaltyModelParameters(aInputs);
        this->setDimensionlessProperties(aInputs);
        this->setNaturalBoundaryConditions(aInputs);
    }

    virtual ~TemperatureResidual(){}

    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        // set local data
        Plato::ScalarVectorT<ConfigT>     tCellVolume("cell weight", tNumCells);
        Plato::ScalarVectorT<ResultT>     tInternalForces("internal forces", tNumCells);
        Plato::ScalarVectorT<CurTempT>    tCurTempGP("current temperature at Gauss points", tNumCells);
        Plato::ScalarVectorT<PrevTempT>   tPrevTempGP("previous temperature at Gauss points", tNumCells);
        Plato::ScalarVectorT<ResultT>     tHeatSource("heat source", tNumCells);
        Plato::blas1::fill(mHeatSourceConstant, tHeatSource);
        Plato::ScalarVectorT<DivergenceT> tDivPrevVel("divergence previous velocity", tNumCells);

        Plato::ScalarMultiVectorT<ResultT> tThermalFlux("thermal flux", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultT> tStabForces("stabilizing forces", tNumCells, mNumTempDofsPerCell);
        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity at Gauss points", tNumCells, mNumVelDofsPerNode);
        
        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode> tIntrplScalarField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // set input state worksets
        auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
        auto tControlWS  = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tConfigWS   = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS  = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tCurTempWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurTempT>>(aWorkSets.get("current temperature"));
        auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));

        // transfer member data to device
        auto tRefTemp               = mReferenceTemperature;
        auto tCharLength            = mCharacteristicLength;
        auto tEffConductivity       = mEffectiveConductivity;
        auto tFluidThermalDiff      = mFluidThermalDiffusivity;
        auto tSolidThermalDiff      = mSolidThermalDiffusivity;
        auto tFluidThermalCond      = mFluidThermalConductivity;
        auto tHeatSrcPenaltyExp     = mHeatSourcePenaltyExponent;
        auto tThermalDiffPenaltyExp = mThermalDiffPenaltyExponent;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. calculate internal forces
            ControlT tPenalizedDiffusivity = Plato::Fluids::penalize_thermal_diffusivity<mNumNodesPerCell>
                (aCellOrdinal, tFluidThermalDiff, tSolidThermalDiff, tThermalDiffPenaltyExp, tControlWS);
            Plato::Fluids::calculate_flux<mNumNodesPerCell,mNumSpatialDims>(aCellOrdinal, tGradient, tPrevTempWS, tThermalFlux);
            tPenalizedDiffusivity = tEffConductivity * tPenalizedDiffusivity;
            Plato::blas1::scale<mNumSpatialDims>(aCellOrdinal, tPenalizedDiffusivity, tThermalFlux);
            Plato::Fluids::calculate_flux_divergence<mNumNodesPerCell,mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tThermalFlux, aResultWS);

            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            Plato::Fluids::calculate_convective_forces<mNumNodesPerCell,mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevVelGP, tPrevTempWS, tInternalForces);

            auto tHeatSrcConst = ( tCharLength * tCharLength ) / (tFluidThermalCond * tRefTemp);
            ControlT tPenalizedHeatSrcConst = Plato::Fluids::penalize_heat_source_constant<mNumNodesPerCell>
                (aCellOrdinal, tHeatSrcConst, tHeatSrcPenaltyExp, tControlWS);
            tInternalForces(aCellOrdinal) -= tPenalizedHeatSrcConst * tHeatSource(aCellOrdinal);

            Plato::Fluids::integrate_scalar_field<mNumNodesPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tInternalForces, aResultWS);

            // 2. calculate stabilizing forces
            Plato::Fluids::divergence<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevVelWS, tDivPrevVel);
            Plato::Fluids::integrate_stabilizing_scalar_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tDivPrevVel, tGradient, tPrevVelGP, tInternalForces, tStabForces);
            Plato::Fluids::multiply_time_step<mNumNodesPerCell, mNumDofsPerNode>
                (aCellOrdinal, 0.5, tTimeStepWS, tStabForces);
            Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, 1.0, tStabForces, 1.0, aResultWS);

            // 3. apply time step to sum of internal and stabilizing forces, F = \Delta{t}\left( F_i^{int} + F_i^{stab} \right)
            Plato::Fluids::multiply_time_step<mNumNodesPerCell, mNumDofsPerNode>
                (aCellOrdinal, 1.0, tTimeStepWS, aResultWS);

            // 4. add inertial force contribution
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurTempWS, tCurTempGP);
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevTempWS, tPrevTempGP);
            Plato::Fluids::calculate_inertial_forces<mNumNodesPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tCurTempGP, tPrevTempGP, aResultWS);
        }, "conservation of energy internal forces");
    }

    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return; /* boundary integral equates zero */ }

    void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    {
        if( mHeatFlux != nullptr )
        {
            // set input state worksets
            auto tConfigWS   = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
            auto tControlWS  = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
            auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));

            // evaluate prescribed flux
            auto tNumCells = aResult.extent(0);
            Plato::ScalarMultiVectorT<ResultT> tResultWS("heat flux", tNumCells, mNumDofsPerCell);
            mHeatFlux->get( aSpatialModel, tPrevTempWS, tControlWS, tConfigWS, tResultWS );

            auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                Plato::Fluids::multiply_time_step<mNumNodesPerCell, mNumDofsPerNode>
                    (aCellOrdinal, -1.0, tTimeStepWS, tResultWS);
                Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, 1.0, tResultWS, 1.0, aResult);
            }, "heat flux contribution");
        }
    }

private:
    void setSourceTerm
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Heat Source"))
        {
            mHeatSourceConstant = aInputs.sublist("Heat Source").get<Plato::Scalar>("Constant", 0.0);
        }
    }

    void setThermalProperties
    (Teuchos::ParameterList & aInputs)
    {
        auto tMaterialName = mSpatialDomain.getMaterialName();
        Plato::is_material_defined(tMaterialName, aInputs);
        auto tMaterial = aInputs.sublist("Material Models").sublist(tMaterialName);
        auto tThermalPropBlock = std::string("Thermal Properties");
        mReferenceTemperature = Plato::parse_parameter<Plato::Scalar>("Reference Temperature", tThermalPropBlock, tMaterial);
        mSolidThermalDiffusivity = Plato::parse_parameter<Plato::Scalar>("Solid Thermal Diffusivity", tThermalPropBlock, tMaterial);
        mFluidThermalConductivity = Plato::parse_parameter<Plato::Scalar>("Fluid Thermal Conductivity", tThermalPropBlock, tMaterial);
    }

    void setCharacteristicLength
    (Teuchos::ParameterList & aInputs)
    {
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        mCharacteristicLength = Plato::parse_parameter<Plato::Scalar>("Characteristic Length", "Dimensionless Properties", tHyperbolic);
    }

    void setEffectiveConductivity
    (Teuchos::ParameterList & aInputs)
    {
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
        auto tHeatTransfer = Plato::tolower(tTag);

        if(tHeatTransfer == "forced" || tHeatTransfer == "none")
        {
            auto tPrNum = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
            auto tReNum = Plato::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolic);
            mEffectiveConductivity = static_cast<Plato::Scalar>(1) / (tReNum*tPrNum);
        }
        else if(tHeatTransfer == "natural")
        {
            mEffectiveConductivity = 1.0;
        }
        else
        {
            THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
        }
    }

    void setDimensionlessProperties
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        this->setCharacteristicLength(aInputs);
        this->setEffectiveConductivity(aInputs);
    }

    void setPenaltyModelParameters
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        auto tHyperbolicParamList = aInputs.sublist("Hyperbolic");
        if(tHyperbolicParamList.isSublist("Energy Conservation"))
        {
            auto tEnergyParamList = tHyperbolicParamList.sublist("Energy Conservation");
            if (tEnergyParamList.isSublist("Penalty Function"))
            {
                auto tPenaltyFuncList = tEnergyParamList.sublist("Penalty Function");
                mHeatSourcePenaltyExponent = tPenaltyFuncList.get<Plato::Scalar>("Heat Source Penalty Exponent", 3.0);
                mThermalDiffPenaltyExponent = tPenaltyFuncList.get<Plato::Scalar>("Thermal Diffusion Penalty Exponent", 3.0);
            }
        }
    }

    void setNaturalBoundaryConditions
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Energy Natural Boundary Conditions"))
        {
            auto tSublist = aInputs.sublist("Energy Natural Boundary Conditions");
            mHeatFlux = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>>(tSublist);
        }
    }
};
// class TemperatureResidual

}
// namespace SIMP


template<typename PhysicsT, typename EvaluationT>
class MomentumSurfaceForces
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;        /*!< number of nodes per face */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace;   /*!< number of spatial dimensions on face */

    // forward automatic differentiation types
    using ResultT  = typename EvaluationT::ResultScalarType;
    using ConfigT  = typename EvaluationT::ConfigScalarType;
    using PrevVelT = typename EvaluationT::PreviousMomentumScalarType;

    const std::string mEntitySetName; /*!< side set name */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mVolumeCubatureRule;  /*!< volume integration rule */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace> mSurfaceCubatureRule; /*!< surface integration rule */

public:
    MomentumSurfaceForces
    (const Plato::SpatialDomain & aSpatialDomain,
     const std::string & aEntitySetName) :
         mEntitySetName(aEntitySetName),
         mSpatialDomain(aSpatialDomain)
    {
    }

    void operator()
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult,
     Plato::Scalar aMultiplier = 1.0) const
    {
        // get mesh vertices
        auto tFace2Verts = mSpatialDomain.Mesh.ask_verts_of(mNumSpatialDimsOnFace);
        auto tCell2Verts = mSpatialDomain.Mesh.ask_elem_verts();

        // get face to element graph
        auto tFace2eElems = mSpatialDomain.Mesh.ask_up(mNumSpatialDimsOnFace, mNumSpatialDims);
        auto tFace2Elems_map   = tFace2eElems.a2ab;
        auto tFace2Elems_elems = tFace2eElems.ab2b;

        // get element to face map
        auto tElem2Faces = mSpatialDomain.Mesh.ask_down(mNumSpatialDims, mNumSpatialDimsOnFace).ab2b;

        // set local functors
        Plato::CalculateSurfaceArea<mNumSpatialDims> tCalculateSurfaceArea;
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(&(mSpatialDomain.Mesh));
        Plato::CalculateSurfaceJacobians<mNumSpatialDims> tCalculateSurfaceJacobians;
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode,   0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // get sideset faces
        auto tFaceLocalOrdinals = Plato::side_set_face_ordinals(mSpatialDomain.MeshSets, mEntitySetName);
        auto tNumFaces = tFaceLocalOrdinals.size();
        Plato::ScalarArray3DT<ConfigT> tJacobians("jacobian", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);
        auto tNumCells = mSpatialDomain.Mesh.nelems();
        Plato::ScalarMultiVector tVelBCsGP("velocity boundary conditions", tNumCells, mNumVelDofsPerNode);
        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity", tNumCells, mNumVelDofsPerNode);

        // set input state worksets
        auto tVelBCsWS  = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("prescribed velocity"));
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));

        // evaluate integral
        auto tVolumeBasisFunctions = mVolumeCubatureRule.getBasisFunctions();
        auto tSurfaceCubatureWeight = mSurfaceCubatureRule.getCubWeight();
        auto tSurfaceBasisFunctions = mSurfaceCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
        {

          auto tFaceOrdinal = tFaceLocalOrdinals[aFaceI];
          // for each element that the face is connected to: (either 1 or 2 elements)
          for( Plato::OrdinalType tElem = tFace2Elems_map[tFaceOrdinal]; tElem < tFace2Elems_map[tFaceOrdinal + 1]; tElem++ )
          {
              // create a map from face local node index to elem local node index
              Plato::OrdinalType tLocalNodeOrd[mNumSpatialDims];
              auto tCellOrdinal = tFace2Elems_elems[tElem];
              tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrd);

              // calculate surface jacobians
              ConfigT tSurfaceAreaTimesCubWeight(0.0);
              tCalculateSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrd, tConfigWS, tJacobians);
              tCalculateSurfaceArea(aFaceI, tSurfaceCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

              // compute unit normal vector
              auto tElemFaceOrdinal = Plato::get_face_ordinal<mNumSpatialDims>(tCellOrdinal, tFaceOrdinal, tElem2Faces);
              auto tUnitNormalVec = Plato::unit_normal_vector(tCellOrdinal, tElemFaceOrdinal, tCoords);

              // project into aResult workset
              tIntrplVectorField(tCellOrdinal, tVolumeBasisFunctions, tVelBCsWS, tVelBCsGP);
              tIntrplVectorField(tCellOrdinal, tVolumeBasisFunctions, tPrevVelWS, tPrevVelGP);
              for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++ )
              {
                  auto tLocalCellNode = tLocalNodeOrd[tNode];
                  for( Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++ )
                  {
                      aResult(tCellOrdinal, tLocalCellNode) += tUnitNormalVec(tDim) *
                          ( tVelBCsGP(tCellOrdinal, tDim) - tPrevVelGP(tCellOrdinal, tDim) );
                  }
                  aResult(tCellOrdinal, tLocalCellNode) *=
                      aMultiplier * tSurfaceBasisFunctions(tNode) * tSurfaceAreaTimesCubWeight;
              }
          }
        }, "calculate surface momentum integral");
    }
};
// class MomentumSurfaceForces



// todo unit test all these inline functions for the pressure increment calculation
// integrate previous advective force, which is defined as
// \int_{\Omega_e}\frac{\partial N_p^a}{partial x_i}u_i^{n-1} d\Omega_e
template<Plato::OrdinalType NumNodesPerCell,
         Plato::OrdinalType NumSpatialDims,
         typename ConfigT,
         typename PrevVelT,
         typename ResultT>
DEVICE_TYPE inline void
integrate_advected_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVel,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < NumSpatialDims; tDim++)
        {
            aResult(aCellOrdinal, tNode) +=
                aCellVolume(aCellOrdinal) * aGradient(aCellOrdinal, tNode, tDim) * aPrevVel(aCellOrdinal, tDim);
        }
    }
}

// integrate current predicted advective force, which is defined as
// \int_{\Omega_e}\frac{\partial N_p^a}{partial x_i} \left( \theta_1\left( u^{\ast}_i - u^{n-1}_i \right) \right) d\Omega_e
template<Plato::OrdinalType NumNodesPerCell,
         Plato::OrdinalType NumSpatialDims,
         typename ConfigT,
         typename PrevVelT,
         typename PredVelT,
         typename ResultT>
DEVICE_TYPE inline void
integrate_delta_advected_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar & aMultiplier,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<PredVelT> & aPredVel,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVel,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < NumSpatialDims; tDim++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * aCellVolume(aCellOrdinal) *
                aGradient(aCellOrdinal, tNode, tDim) * ( aPredVel(aCellOrdinal, tDim) - aPrevVel(aCellOrdinal, tDim) );
        }
    }
}

// integrate fist term enforcing continuity, which is defined as
// -\theta_1\int_{\Omega_e}\frac{\partial N_p^a}{partial x_i}\Delta{t}\frac{\partial p^{n-1}}{\partial x_i} d\Omega_e,
template<Plato::OrdinalType NumNodesPerCell,
         Plato::OrdinalType NumSpatialDims,
         typename ConfigT,
         typename PrevPressT,
         typename ResultT>
DEVICE_TYPE inline void
previous_pressure_gradient_divergence
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar & aMultiplier,
 const Plato::ScalarMultiVector & aTimeStep,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<PrevPressT> & aPrevPressGrad,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < NumSpatialDims; tDim++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * aTimeStep(aCellOrdinal, tNode) *
                aGradient(aCellOrdinal, tNode, tDim) * aPrevPressGrad(aCellOrdinal, tDim) * aCellVolume(aCellOrdinal);
        }
    }
}

// integrate second term enforcing continuity, which is defined as
// -\theta_1\int_{\Omega_e}\frac{\partial N_p^a}{partial x_i}\Delta{t}\theta_2\frac{\partial \Delta{p}}{\partial x_i} d\Omega_e,
// where \Delta{p}=p^{n}-p^{n-1}
template<Plato::OrdinalType NumNodesPerCell,
         Plato::OrdinalType NumSpatialDims,
         typename ConfigT,
         typename CurPressT,
         typename PrevPressT,
         typename ResultT>
DEVICE_TYPE inline void
delta_pressure_gradient_divergence
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar & aMultiplier,
 const Plato::ScalarMultiVector & aTimeStep,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<CurPressT> & aCurPressGrad,
 const Plato::ScalarMultiVectorT<PrevPressT> & aPrevPressGrad,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < NumSpatialDims; tDim++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * aTimeStep(aCellOrdinal, tNode) * aCellVolume(aCellOrdinal) *
                aGradient(aCellOrdinal, tNode, tDim) * ( aCurPressGrad(aCellOrdinal, tDim) - aPrevPressGrad(aCellOrdinal, tDim) );
        }
    }
}

// integrate inertial forces, which are defined as
// \int_{\Omega_e} N_p^a\left(\frac{1}{\beta^2}\right)(p^n - p^{n-1}) d\Omega_e
template<Plato::OrdinalType NumNodesPerCell,
         typename ConfigT,
         typename PrevPressT,
         typename CurPressT,
         typename ResultT>
DEVICE_TYPE inline void
integrate_inertial_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarVectorT<CurPressT> & aCurPress,
 const Plato::ScalarVectorT<PrevPressT> & aPrevPress,
 const Plato::ScalarMultiVector & aArtificialCompress,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        aResult(aCellOrdinal, tNode) += aCellVolume(aCellOrdinal) * aBasisFunctions(tNode) *
            ( static_cast<Plato::Scalar>(1.0) / aArtificialCompress(aCellOrdinal, tNode) ) *
            ( aCurPress(aCellOrdinal) - aPrevPress(aCellOrdinal) );
    }
}




template<typename PhysicsT, typename EvaluationT>
class PressureResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumPressDofsPerCell  = PhysicsT::SimplexT::mNumMassDofsPerCell;     /*!< number of mass dofs per cell */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell;   /*!< number of configuration degrees of freedom per cell */

    using ResultT    = typename EvaluationT::ResultScalarType;
    using ConfigT    = typename EvaluationT::ConfigScalarType;
    using PredVelT   = typename EvaluationT::MomentumPredictorScalarType;
    using ControlT   = typename EvaluationT::ControlScalarType;
    using CurPressT  = typename EvaluationT::CurrentMassScalarType;
    using PrevVelT   = typename EvaluationT::PreviousMomentumScalarType;
    using PrevPressT = typename EvaluationT::PreviousMassScalarType;

    using CurPressGradT  = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurPressT, ConfigT>;
    using PrevPressGradT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevPressT, ConfigT>;

    Plato::DataMap& mDataMap;                   /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< integration rule */

    Plato::Scalar mThetaOne = 0.5;
    Plato::Scalar mThetaTwo = 0.0;

    using MomentumForces = Plato::Fluids::MomentumSurfaceForces<PhysicsT, EvaluationT>;
    std::unordered_map<std::string, std::shared_ptr<MomentumForces>> mMomentumBCs;

public:
    PressureResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setParameters(aInputs);
        this->setNaturalBoundaryConditions(aInputs);
    }

    PressureResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
    }

    virtual ~PressureResidual(){}

    void setThetaOne(const Plato::Scalar & aThetaOne)
    {
        mThetaOne = aThetaOne;
    }

    void setThetaTwo(const Plato::Scalar & aThetaTwo)
    {
        mThetaTwo = aThetaTwo;
    }

    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResult.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResult.extent(0)) + "' elements.")
        }

        // set local data
        Plato::ScalarVectorT<ConfigT>    tCellVolume("cell weight", tNumCells);
        Plato::ScalarVectorT<CurPressT>  tCurPressGP("current pressure", tNumCells);
        Plato::ScalarVectorT<PrevPressT> tPrevPressGP("previous pressure", tNumCells);
        Plato::ScalarArray3DT<ConfigT>   tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        Plato::ScalarMultiVectorT<ResultT>  tInternalForces("internal forces", tNumCells, mNumPressDofsPerCell);
        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity at Gauss point", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PredVelT> tPredVelGP("predicted velocity at Gauss point", tNumCells, mNumSpatialDims);

        Plato::ScalarMultiVectorT<CurPressGradT>  tCurPressGrad("current pressure gradient", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PrevPressGradT> tPrevPressGrad("previous pressure gradient", tNumCells, mNumSpatialDims);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode,   0/*offset*/, mNumSpatialDims> tIntrplVectorField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumPressDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplScalarField;

        // set input state worksets
        auto tTimeStepWS  = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
        auto tACompressWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("artificial compressibility"));
        auto tConfigWS    = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tCurPressWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurPressT>>(aWorkSets.get("current pressure"));
        auto tPrevPressWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevPressT>>(aWorkSets.get("previous pressure"));
        auto tPredictorWS = Plato::metadata<Plato::ScalarMultiVectorT<PredVelT>>(aWorkSets.get("current predictor"));

        // transfer member data to device
        auto tThetaOne = mThetaOne;
        auto tThetaTwo = mThetaTwo;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. integrate internal forces
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            Plato::Fluids::integrate_advected_forces<mNumNodesPerCell,mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tPrevVelGP, tInternalForces);

            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPredictorWS, tPredVelGP);
            Plato::Fluids::integrate_delta_advected_forces<mNumNodesPerCell,mNumSpatialDims>
                (aCellOrdinal, tThetaOne, tGradient, tCellVolume, tPredVelGP, tPrevVelGP, tInternalForces);

            Plato::calculate_scalar_field_gradient<mNumNodesPerCell,mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevPressWS, tPrevPressGrad);
            Plato::calculate_scalar_field_gradient<mNumNodesPerCell,mNumSpatialDims>
                (aCellOrdinal, tGradient, tCurPressWS, tCurPressGrad);
            Plato::Fluids::previous_pressure_gradient_divergence<mNumNodesPerCell,mNumSpatialDims>
               (aCellOrdinal, -tThetaOne, tTimeStepWS, tGradient, tCellVolume, tPrevPressGrad, tInternalForces);
            auto tMultiplier = -tThetaOne * tThetaTwo;
            Plato::Fluids::delta_pressure_gradient_divergence<mNumNodesPerCell,mNumSpatialDims>
                (aCellOrdinal, tMultiplier, tTimeStepWS, tGradient, tCellVolume, tCurPressGrad, tPrevPressGrad, tInternalForces);

            Plato::Fluids::multiply_time_step<mNumNodesPerCell,mNumPressDofsPerNode>
                (aCellOrdinal, 1.0, tTimeStepWS, tInternalForces);

            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tCurPressWS, tCurPressGP);
            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tPrevPressWS, tPrevPressGP);
            Plato::Fluids::integrate_inertial_forces<mNumNodesPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tCurPressGP, tPrevPressGP, tACompressWS, aResult);

            // 2. add internal forces to inertial forces
            Plato::blas1::update<mNumPressDofsPerCell>(aCellOrdinal, -1.0, tInternalForces, 1.0, aResult);
        }, "conservation of mass internal forces");
    }

    // todo: verify implementation with formulation - checked!
    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    {
        // calculate previous momentum forces, which are defined as
        // -\theta_1\Delta{t} \int_{\Gamma_u} N_u^a n_i( \hat{u}_i^n - u_i^{n-1} ) d\Gamma_u,
        // where \hat{u}_i^n denote the prescribed velocities.
        auto tNumCells = aResult.extent(0);
        Plato::ScalarMultiVectorT<ResultT> tResultWS("delta momentum surface forces", tNumCells, mNumPressDofsPerCell);
        for(auto& tPair : mMomentumBCs)
        {
            tPair.second->operator()(aWorkSets, tResultWS);
        }

        // multiply force vector by the corresponding nodal time steps
        auto tThetaOne = mThetaOne;
        auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            Plato::Fluids::multiply_time_step<mNumNodesPerCell,mNumPressDofsPerNode>
                (aCellOrdinal, tThetaOne, tTimeStepWS, tResultWS);
            Plato::blas1::update<mNumPressDofsPerCell>(aCellOrdinal, 1.0, tResultWS, 1.0, aResult);
        }, "delta momentum surface forces");
    }

    void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return; }

private:
    void setParameters(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mThetaTwo = tTimeIntegration.get<Plato::Scalar>("Artificial Damping One", 0.5);
            mThetaTwo = tTimeIntegration.get<Plato::Scalar>("Artificial Damping Two", 0.0);
        }
    }

    void setNaturalBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        // the natural bcs are applied on the side sets where velocity essential
        // bcs are enforced. therefore, these side sets should be read by this function.
        std::unordered_map<std::string, std::vector<std::pair<Plato::OrdinalType, Plato::Scalar>>> tMap;
        if(aInputs.isSublist("Momentum Essential Boundary Conditions") == false)
        {
            THROWERR("'Momentum Essential Boundary Conditions' block must be defined for fluid flow problems.")
        }
        auto tSublist = aInputs.sublist("Momentum Essential Boundary Conditions");

        for (Teuchos::ParameterList::ConstIterator tItr = tSublist.begin(); tItr != tSublist.end(); ++tItr)
        {
            const Teuchos::ParameterEntry &tEntry = tSublist.entry(tItr);
            if (!tEntry.isList())
            {
                THROWERR(std::string("Error reading 'Momentum Essential Boundary Conditions' block: Expects a parameter ")
                    + "list input with information pertaining to the velocity boundary conditions .")
            }

            const std::string& tParamListName = tSublist.name(tItr);
            Teuchos::ParameterList & tParamList = tSublist.sublist(tParamListName);
            if (tParamList.isParameter("Sides") == false)
            {
                THROWERR(std::string("Keyword 'Sides' is not define in Parameter List '") + tParamListName + "'.")
            }
            const auto tEntitySetName = tParamList.get<std::string>("Sides");
            auto tMapItr = mMomentumBCs.find(tEntitySetName);
            if(tMapItr == mMomentumBCs.end())
            {
                mMomentumBCs[tEntitySetName] = std::make_shared<MomentumForces>(mSpatialDomain, tEntitySetName);
            }
        }
    }
};
// class PressureResidual




// todo: vector function
/******************************************************************************/
/*! vector function class

   This class takes as a template argument a vector function in the form:

   \f$ F = F(\phi, U^k, P^k, T^k, X) \f$

   and manages the evaluation of the function and derivatives with respect to
   control, \f$\phi\f$, momentum state, \f$ U^k \f$, mass state, \f$ P^k \f$,
   energy state, \f$ T^k \f$, and configuration, \f$ X \f$.

*/
/******************************************************************************/
template<typename PhysicsT>
class VectorFunction
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    static constexpr auto mNumSpatialDims        = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell       = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumPressDofsPerCell   = PhysicsT::SimplexT::mNumMassDofsPerCell;     /*!< number of mass dofs per cell */
    static constexpr auto mNumTempDofsPerCell    = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumVelDofsPerCell     = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumControlDofsPerCell = PhysicsT::SimplexT::mNumControlDofsPerCell;  /*!< number of design variable per cell */
    static constexpr auto mNumPressDofsPerNode   = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumTempDofsPerNode    = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumVelDofsPerNode     = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumControlDofsPerNode = PhysicsT::SimplexT::mNumControlDofsPerNode;  /*!< number of design variable per node */

    static constexpr auto mNumConfigDofsPerNode = PhysicsT::SimplexT::mNumConfigDofsPerNode; /*!< number of configuration degrees of freedom per cell */
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell; /*!< number of configuration degrees of freedom per cell */

    static constexpr auto mNumTimeStepsDofsPerNode = 1; /*!< number of time step dofs per node */
    static constexpr auto mNumACompressDofsPerNode = 1; /*!< number of artificial compressibility dofs per node */

    // forward automatic differentiation evaluation types
    using ResidualEvalT      = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradConfigEvalT    = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradConfig;
    using GradControlEvalT   = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradControl;
    using GradCurVelEvalT    = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurMomentum;
    using GradPrevVelEvalT   = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradPrevMomentum;
    using GradCurTempEvalT   = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurEnergy;
    using GradPrevTempEvalT  = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradPrevEnergy;
    using GradCurPressEvalT  = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurMass;
    using GradPrevPressEvalT = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradPrevMass;
    using GradPredictorEvalT = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradPredictor;

    // element residual vector function types
    using ResidualFuncT      = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, ResidualEvalT>>;
    using GradConfigFuncT    = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradConfigEvalT>>;
    using GradControlFuncT   = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradControlEvalT>>;
    using GradCurVelFuncT    = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradCurVelEvalT>>;
    using GradPrevVelFuncT   = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradPrevVelEvalT>>;
    using GradCurTempFuncT   = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradCurTempEvalT>>;
    using GradPrevTempFuncT  = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradPrevTempEvalT>>;
    using GradCurPressFuncT  = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradCurPressEvalT>>;
    using GradPrevPressFuncT = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradPrevPressEvalT>>;
    using GradPredictorFuncT = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradPredictorEvalT>>;

    // element vector functions per element block, i.e. domain
    std::unordered_map<std::string, ResidualFuncT>      mResidualFuncs;
    std::unordered_map<std::string, GradConfigFuncT>    mGradConfigFuncs;
    std::unordered_map<std::string, GradControlFuncT>   mGradControlFuncs;
    std::unordered_map<std::string, GradCurVelFuncT>    mGradCurVelFuncs;
    std::unordered_map<std::string, GradPrevVelFuncT>   mGradPrevVelFuncs;
    std::unordered_map<std::string, GradCurTempFuncT>   mGradCurTempFuncs;
    std::unordered_map<std::string, GradPrevTempFuncT>  mGradPrevTempFuncs;
    std::unordered_map<std::string, GradCurPressFuncT>  mGradCurPressFuncs;
    std::unordered_map<std::string, GradPrevPressFuncT> mGradPrevPressFuncs;
    std::unordered_map<std::string, GradPredictorFuncT> mGradPredictorFuncs;

    Plato::DataMap& mDataMap;
    const Plato::SpatialModel& mSpatialModel;
    Plato::LocalOrdinalMaps<PhysicsT> mLocalOrdinalMaps;
    Plato::VectorEntryOrdinal<mNumSpatialDims,mNumDofsPerNode> mStateOrdinalsMap;

public:
    /**************************************************************************//**
    * \brief Constructor
    * \param [in] aSpatialModel struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap      problem-specific data map
    * \param [in] aInputs       Teuchos parameter list with input data
    * \param [in] aProblemType  problem type
    ******************************************************************************/
    VectorFunction
    (const std::string            & aName,
     const Plato::SpatialModel    & aSpatialModel,
           Plato::DataMap         & aDataMap,
           Teuchos::ParameterList & aInputs) :
        mSpatialModel(aSpatialModel),
        mDataMap(aDataMap),
        mLocalOrdinalMaps(aSpatialModel.Mesh),
        mStateOrdinalsMap(&aSpatialModel.Mesh)
    {
        this->initialize(aName, aDataMap, aInputs);
    }

    decltype(mNumSpatialDims) getNumSpatialDims() const
    {
        return mNumSpatialDims;
    }

    decltype(mNumDofsPerCell) getNumDofsPerCell() const
    {
        return mNumDofsPerCell;
    }

    decltype(mNumDofsPerNode) getNumDofsPerNode() const
    {
        return mNumDofsPerNode;
    }

    Plato::ScalarVector value
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename ResidualEvalT::ResultScalarType;

        auto tNumNodes = mSpatialModel.Mesh.nverts();
        auto tLength = tNumNodes * mNumDofsPerNode;
        Plato::ScalarVector tReturnValue("Assembled Residual", tLength);

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<ResidualEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mResidualFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::assemble_residual<mNumNodesPerCell,mNumDofsPerNode>(tNumCells, mStateOrdinalsMap, tResultWS, tReturnValue);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<ResidualEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mResidualFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_residual<mNumNodesPerCell,mNumDofsPerNode>(tNumCells, mStateOrdinalsMap, tResultWS, tReturnValue);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mResidualFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_residual<mNumNodesPerCell,mNumDofsPerNode>(tNumCells, mStateOrdinalsMap, tResultWS, tReturnValue);
        }

        return tReturnValue;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientConfig
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradConfigEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumConfigDofsPerNode, mNumDofsPerNode>(&tMesh);

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradConfigEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradConfigFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumConfigDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumConfigDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradConfigEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradConfigFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumConfigDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumConfigDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradConfigFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumConfigDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientControl
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradControlEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControlDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradControlEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradControlFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControlDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumControlDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradControlEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradControlFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControlDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumControlDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradControlFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumControlDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPredictor
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradPredictorEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumVelDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradPredictorEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPredictorFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradPredictorEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPredictorFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPredictorFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradPrevVelEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumVelDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradPrevVelEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevVelFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradPrevVelEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevVelFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPrevVelFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradPrevPressEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumPressDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradPrevPressEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevPressFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradPrevPressEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevPressFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPrevPressFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradPrevTempEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumTempDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradPrevTempEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevTempFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradPrevTempEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevTempFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPrevTempFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradCurVelEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumVelDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradCurVelEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurVelFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradCurVelEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurVelFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradCurVelFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradCurPressEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumPressDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradCurPressEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurPressFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntires = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntires);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradCurPressEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurPressFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradCurPressFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradCurTempEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumTempDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradCurTempEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurTempFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate prescribed forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradCurTempEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate boundary forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurTempFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradCurTempFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

private:
    void initialize
    (const std::string      & aName,
     Plato::DataMap         & aDataMap,
     Teuchos::ParameterList & aInputs)
    {
        typename PhysicsT::FunctionFactory tVecFuncFactory;
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mResidualFuncs[tName]  = tVecFuncFactory.template createVectorFunction<PhysicsT, ResidualEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradControlFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradControlEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradConfigFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradConfigEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradCurPressFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradCurPressEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradPrevPressFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradPrevPressEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradCurTempFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradCurTempEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradPrevTempFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradPrevTempEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradCurVelFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradCurVelEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradPrevVelFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradPrevVelEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradPredictorFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradPredictorEvalT>
                (aName, tDomain, aDataMap, aInputs);
        }
    }
};
// class VectorFunction


template <typename PhysicsT, typename EvaluationT>
inline std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>>
temperature_residual
(const Plato::SpatialDomain & aDomain,
 Plato::DataMap & aDataMap,
 Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }
    auto tHyperbolic = aInputs.sublist("Hyperbolic");

    auto tScenario = tHyperbolic.get<std::string>("Scenario","Analysis");
    auto tLowerScenario = Plato::tolower(tScenario);
    if( tLowerScenario == "density to" )
    {
        return ( std::make_shared<Plato::Fluids::SIMP::TemperatureResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
    }
    else if( tLowerScenario == "analysis" || tLowerScenario == "levelset to" )
    {
        return ( std::make_shared<Plato::Fluids::TemperatureResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
    }
    else
    {
        THROWERR(std::string("Scenario with tag '") + tScenario + "' is not supported. Options are 1) Analysis, 2) Density TO or 3) Levelset TO.")
    }
}

template <typename PhysicsT, typename EvaluationT>
inline std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>>
velocity_predictor_residual
(const Plato::SpatialDomain & aDomain,
 Plato::DataMap & aDataMap,
 Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }
    auto tHyperbolic = aInputs.sublist("Hyperbolic");

    auto tScenario = tHyperbolic.get<std::string>("Scenario","Analysis");
    auto tLowerScenario = Plato::tolower(tScenario);
    if( tLowerScenario == "density to")
    {
        return ( std::make_shared<Plato::Fluids::Brinkman::VelocityPredictorResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
    }
    else if( tLowerScenario == "analysis" || tLowerScenario == "levelset to" )
    {
        return ( std::make_shared<Plato::Fluids::VelocityPredictorResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
    }
    else
    {
        THROWERR(std::string("Scenario with tag '") + tScenario + "' is not supported. Options are 1) Analysis, 2) Density TO or 3) Levelset TO.")
    }
}

struct FunctionFactory
{
public:
    template <typename PhysicsT, typename EvaluationT>
    std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>>
    createVectorFunction
    (const std::string          & aTag,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs)
    {
        // TODO: explore function interface for constructor, similar to how it is done in the xml generator
        auto tLowerTag = Plato::tolower(aTag);
        if( tLowerTag == "pressure" )
        {
            return ( std::make_shared<Plato::Fluids::PressureResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
        }
        else if( tLowerTag == "velocity" )
        {
            return ( std::make_shared<Plato::Fluids::VelocityCorrectorResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
        }
        else if( tLowerTag == "temperature" )
        {
            return ( Plato::Fluids::temperature_residual<PhysicsT, EvaluationT>(aDomain, aDataMap, aInputs) );
        }
        else if( tLowerTag == "velocity predictor" )
        {
            return ( Plato::Fluids::velocity_predictor_residual<PhysicsT, EvaluationT>(aDomain, aDataMap, aInputs) );
        }
        else
        {
            THROWERR(std::string("Vector function with tag '") + aTag + "' is not supported.")
        }
    }

    template <typename PhysicsT, typename EvaluationT>
    std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>>
    createScalarFunction
    (const std::string          & aName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs)
    {
        if( !aInputs.isSublist("Criteria") )
        {
            THROWERR("'Criteria' block is not defined.")
        }
        auto tCriteriaList = aInputs.sublist("Criteria");
        if( !tCriteriaList.isSublist(aName) )
        {
            THROWERR(std::string("Criteria Block with name '") + aName + "' is not defined.")
        }
        auto tCriterion = tCriteriaList.sublist(aName);

        if(!tCriterion.isParameter("Scalar Function Type"))
        {
            THROWERR(std::string("'Scalar Function Type' keyword is not defined in Criterion with name '") + aName + "'.")
        }

        auto tFlowTag = tCriterion.get<std::string>("Flow", "Not Defined");
        auto tFlowLowerTag = Plato::tolower(tFlowTag);
        auto tCriterionTag = tCriterion.get<std::string>("Scalar Function Type", "Not Defined");
        auto tCriterionLowerTag = Plato::tolower(tCriterionTag);

        if( tCriterionLowerTag == "average surface pressure" )
        {
            return ( std::make_shared<Plato::Fluids::AverageSurfacePressure<PhysicsT, EvaluationT>>
                (aName, aDomain, aDataMap, aInputs) );
        }
        else if( tCriterionLowerTag == "average surface temperature" )
        {
            return ( std::make_shared<Plato::Fluids::AverageSurfaceTemperature<PhysicsT, EvaluationT>>
                (aName, aDomain, aDataMap, aInputs) );
        }
        else if( tCriterionLowerTag == "internal dissipation energy" && tFlowLowerTag == "incompressible")
        {
            return ( std::make_shared<Plato::Fluids::InternalDissipationEnergyIncompressible<PhysicsT, EvaluationT>>
                (aName, aDomain, aDataMap, aInputs) );
        }
        else
        {
            THROWERR(std::string("'Scalar Function Type' with tag '") + tCriterionTag
                + "' in Criterion Block '" + aName + "' is not supported.")
        }
    }
};
// struct FunctionFactory


// todo: criterion factory
template<typename PhysicsT>
class CriterionFactory
{
private:
    using ScalarFunctionType = std::shared_ptr<Plato::Fluids::CriterionBase>;

public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    CriterionFactory () {}

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    ~CriterionFactory() {}

    /******************************************************************************//**
     * \brief Creates criterion interface, which allows evaluations.
     * \param [in] aSpatialModel  C++ structure with volume and surface mesh databases
     * \param [in] aDataMap       Plato Analyze data map
     * \param [in] aInputs        input parameters from Analyze's input file
     * \param [in] aName          scalar function name
     **********************************************************************************/
    ScalarFunctionType
    createCriterion
    (Plato::SpatialModel & aSpatialModel,
     Plato::DataMap & aDataMap,
     Teuchos::ParameterList & aInputs,
     std::string & aName)
     {
        auto tFunctionTag = aInputs.sublist("Criteria").sublist(aName);
        auto tType = tFunctionTag.get<std::string>("Type", "Not Defined");
        auto tLowerType = Plato::tolower(tType);

        if(tLowerType == "scalar function")
        {
            auto tCriterion =
                std::make_shared<Plato::Fluids::PhysicsScalarFunction<PhysicsT>>
                    (aSpatialModel, aDataMap, aInputs, aName);
            return tCriterion;
        }
        /*else if(tLowerType == "weighted sum")
        {
            auto tCriterion =
                std::make_shared<Plato::Fluids::WeightedScalarFunction<PhysicsT>>
                    (aSpatialModel, aDataMap, aInputs, aName);
            return tCriterion;
        }
        else if(tLowerType == "least squares")
        {
            auto tCriterion =
                std::make_shared<Plato::Fluids::LeastSquaresScalarFunction<PhysicsT>>
                    (aSpatialModel, aDataMap, aInputs, aName);
            return tCriterion;
        }*/
        else
        {
            THROWERR(std::string("Scalar function in block '") + aName + "' with type '" + tType + "' is not supported.")
        }
     }
};


// todo: weighted scalar function
template<typename PhysicsT>
class WeightedScalarFunction : public Plato::Fluids::CriterionBase
{
private:
    // static metadata
    static constexpr auto mNumSpatialDims        = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumPressDofsPerNode   = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumTempDofsPerNode    = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumVelDofsPerNode     = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumControlDofsPerNode = PhysicsT::SimplexT::mNumControlDofsPerNode;  /*!< number of design variables per node */

    // set local typenames
    using Criterion    = std::shared_ptr<Plato::Fluids::CriterionBase>;

    bool mDiagnostics; /*!< write diagnostics to terminal */
    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialModel& mSpatialModel; /*!< mesh database */

    std::string mFuncName; /*!< weighted scalar function name */

    std::vector<Criterion>     mCriteria;         /*!< list of scalar function criteria */
    std::vector<std::string>   mCriterionNames;   /*!< list of criterion names */
    std::vector<Plato::Scalar> mCriterionWeights; /*!< list of criterion weights */

public:
    WeightedScalarFunction
    (const Plato::SpatialModel & aSpatialModel,
     Plato::DataMap & aDataMap,
     Teuchos::ParameterList & aInputs,
     std::string & aName) :
         mDiagnostics(false),
         mDataMap     (aDataMap),
         mSpatialModel(aSpatialModel),
         mFuncName    (aName)
    {
        this->initialize(aInputs);
    }

    virtual ~WeightedScalarFunction(){}

    void append
    (const Criterion     & aFunc,
     const std::string   & aName,
           Plato::Scalar   aWeight = 1.0)
    {
        mCriteria.push_back(aFunc);
        mCriterionNames.push_back(aName);
        mCriterionWeights.push_back(aWeight);
    }

    std::string name() const override
    {
        return mFuncName;
    }

    Plato::Scalar
    value
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        Plato::Scalar tResult = 0.0;
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tValue = tCriterion->value(aControls, aVariables);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            const auto tFuncValue = tFuncWeight * tValue;

            const auto tFuncName = mCriterionNames[tIndex];
            mDataMap.mScalarValues[tFuncName] = tFuncValue;
            tResult += tFuncValue;

            if(mDiagnostics)
            {
                printf("Scalar Function Name = %s \t Value = %f\n", tFuncName.c_str(), tFuncValue);
            }
        }

        if(mDiagnostics)
        {
            printf("Weighted Sum Name = %s \t Value = %f\n", mFuncName.c_str(), tResult);
        }
        return tResult;
    }

    Plato::ScalarVector gradientConfig
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tTotalDerivative("total derivative", mNumSpatialDims * tNumNodes);
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            auto tGradient = tCriterion->gradientConfig(aControls, aVariables);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            Plato::blas1::update(tFuncWeight, tGradient, static_cast<Plato::Scalar>(1.0), tTotalDerivative);
        }
        return tTotalDerivative;
    }

    Plato::ScalarVector gradientControl
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tTotalDerivative("total derivative", mNumControlDofsPerNode * tNumNodes);
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            auto tGradient = tCriterion->gradientControl(aControls, aVariables);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            Plato::blas1::update(tFuncWeight, tGradient, static_cast<Plato::Scalar>(1.0), tTotalDerivative);
        }
        return tTotalDerivative;
    }

    Plato::ScalarVector gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tTotalDerivative("total derivative", mNumPressDofsPerNode * tNumNodes);
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            auto tGradient = tCriterion->gradientCurrentPress(aControls, aVariables);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            Plato::blas1::update(tFuncWeight, tGradient, static_cast<Plato::Scalar>(1.0), tTotalDerivative);
        }
        return tTotalDerivative;
    }

    Plato::ScalarVector gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tTotalDerivative("total derivative", mNumTempDofsPerNode * tNumNodes);
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            auto tGradient = tCriterion->gradientCurrentTemp(aControls, aVariables);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            Plato::blas1::update(tFuncWeight, tGradient, static_cast<Plato::Scalar>(1.0), tTotalDerivative);
        }
        return tTotalDerivative;
    }

    Plato::ScalarVector gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tTotalDerivative("total derivative", mNumVelDofsPerNode * tNumNodes);
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            auto tGradient = tCriterion->gradientCurrentVel(aControls, aVariables);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            Plato::blas1::update(tFuncWeight, tGradient, static_cast<Plato::Scalar>(1.0), tTotalDerivative);
        }
        return tTotalDerivative;
    }

private:
    void checkInputs()
    {
        if (mCriterionNames.size() != mCriterionWeights.size())
        {
            THROWERR(std::string("Dimensions mismatch.  Number of 'Functions' and 'Weights' do not match. ") +
                     "Check scalar function with name '" + mFuncName + "'.")
        }
    }

    void initialize(Teuchos::ParameterList & aInputs)
    {
        if(aInputs.sublist("Criteria").isSublist(mFuncName) == false)
        {
            THROWERR(std::string("Scalar function with tag '") + mFuncName + "' is not defined in the input file.")
        }

        auto tCriteriaInputs = aInputs.sublist("Criteria").sublist(mFuncName);
        this->parseFunctions(tCriteriaInputs);
        this->parseWeights(tCriteriaInputs);
        this->checkInputs();

        Plato::Fluids::CriterionFactory<PhysicsT> tFactory;
        for(auto& tName : mCriterionNames)
        {
            auto tScalarFunction = tFactory.createCriterion(mSpatialModel, mDataMap, aInputs, tName);
            mCriteria.push_back(tScalarFunction);
        }
    }

    void parseFunctions(Teuchos::ParameterList & aInputs)
    {
        mCriterionNames = Plato::parse_array<std::string>("Functions", aInputs);
        if(mCriterionNames.empty())
        {
            THROWERR(std::string("'Functions' keyword was not defined in function block with name '") + mFuncName
                + "'. User must define 'functions' used in a 'Weighted Sum' criterion.")
        }
    }

    void parseWeights(Teuchos::ParameterList & aInputs)
    {
        mCriterionWeights = Plato::parse_array<Plato::Scalar>("Weights", aInputs);
        if(mCriterionNames.empty())
        {
            THROWERR(std::string("'Weights' keyword was not defined in function block with name '") + mFuncName
                + "'. User must define 'weights' used in a 'Weighted Sum' criterion.")
        }
    }
};
// class WeightedScalarFunction





// todo: least squares scalar function
template<typename PhysicsT>
class LeastSquaresScalarFunction : public Plato::Fluids::CriterionBase
{
private:
    // static metadata
    static constexpr auto mNumSpatialDims        = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumPressDofsPerNode   = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumTempDofsPerNode    = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumVelDofsPerNode     = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumControlDofsPerNode = PhysicsT::SimplexT::mNumControlDofsPerNode;  /*!< number of control dofs per node */
    static constexpr auto mNumConfigDofsPerNode  = PhysicsT::SimplexT::mNumConfigDofsPerNode;   /*!< number of configuration dofs per node */

    // set local typenames
    using Criterion = std::shared_ptr<Plato::Fluids::CriterionBase>;

    bool mDiagnostics; /*!< write diagnostics to terminal */
    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialModel& mSpatialModel; /*!< mesh database */

    std::string mFuncName; /*!< weighted scalar function name */

    std::vector<Criterion>     mCriteria;                /*!< list of scalar function criteria */
    std::vector<std::string>   mCriterionNames;          /*!< list of criterion names */
    std::vector<Plato::Scalar> mCriterionWeights;        /*!< list of criterion weights */
    std::vector<Plato::Scalar> mCriterionNormalizations; /*!< list of criterion normalization */

public:
    LeastSquaresScalarFunction
    (const Plato::SpatialModel & aSpatialModel,
     Plato::DataMap & aDataMap,
     Teuchos::ParameterList & aInputs,
     std::string & aName) :
         mDiagnostics(false),
         mDataMap     (aDataMap),
         mSpatialModel(aSpatialModel),
         mFuncName    (aName)
    {
        this->initialize(aInputs);
    }

    std::string name() const override
    {
        return mFuncName;
    }

    Plato::Scalar value
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        Plato::Scalar tResult = 0.0;
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aVariables);

            auto tValue = tCriterionValue / tNormalization;
            tResult += tWeight * (tValue * tValue);
        }
        return tResult;
    }

    Plato::ScalarVector gradientConfig
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumDofs = mNumConfigDofsPerNode * mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradConfig("gradient configuration", tNumDofs);
        Plato::blas1::fill(0.0, tGradConfig);
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aVariables);
            auto tCriterionGrad  = tCriterion->gradientConfig(aControls, aVariables);

            auto tMultiplier = ( static_cast<Plato::Scalar>(2.0) * tWeight * tCriterionValue )
                               / ( tNormalization * tNormalization );
            Plato::blas1::update(tMultiplier, tCriterionGrad, 1.0, tGradConfig);
        }
        return tGradConfig;
    }

    Plato::ScalarVector gradientControl
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumDofs = mNumControlDofsPerNode * mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradControl("gradient control", tNumDofs);
        Plato::blas1::fill(0.0, tGradControl);
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aVariables);
            auto tCriterionGrad  = tCriterion->gradientControl(aControls, aVariables);

            auto tMultiplier = ( static_cast<Plato::Scalar>(2.0) * tWeight * tCriterionValue )
                               / ( tNormalization * tNormalization );
            Plato::blas1::update(tMultiplier, tCriterionGrad, 1.0, tGradControl);
        }
        return tGradControl;
    }

    Plato::ScalarVector gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumDofs = mNumPressDofsPerNode * mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradCurPress("gradient current pressure", tNumDofs);
        Plato::blas1::fill(0.0, tGradCurPress);
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aVariables);
            auto tCriterionGrad  = tCriterion->gradientCurrentPress(aControls, aVariables);

            auto tMultiplier = ( static_cast<Plato::Scalar>(2.0) * tWeight * tCriterionValue )
                               / ( tNormalization * tNormalization );
            Plato::blas1::update(tMultiplier, tCriterionGrad, 1.0, tGradCurPress);
        }
        return tGradCurPress;
    }

    Plato::ScalarVector gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumDofs = mNumTempDofsPerNode * mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradCurTemp("gradient current temperature", tNumDofs);
        Plato::blas1::fill(0.0, tGradCurTemp);
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aVariables);
            auto tCriterionGrad  = tCriterion->gradientCurrentTemp(aControls, aVariables);

            auto tMultiplier = ( static_cast<Plato::Scalar>(2.0) * tWeight * tCriterionValue )
                               / ( tNormalization * tNormalization );
            Plato::blas1::update(tMultiplier, tCriterionGrad, 1.0, tGradCurTemp);
        }
        return tGradCurTemp;
    }

    Plato::ScalarVector gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumDofs = mNumVelDofsPerNode * mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradCurVel("gradient current velocity", tNumDofs);
        Plato::blas1::fill(0.0, tGradCurVel);
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aVariables);
            auto tCriterionGrad  = tCriterion->gradientCurrentVel(aControls, aVariables);

            auto tMultiplier = ( static_cast<Plato::Scalar>(2.0) * tWeight * tCriterionValue )
                               / ( tNormalization * tNormalization );
            Plato::blas1::update(tMultiplier, tCriterionGrad, 1.0, tGradCurVel);
        }
        return tGradCurVel;
    }

private:
    void checkInputs()
    {
        if (mCriterionNames.size() != mCriterionWeights.size())
        {
            THROWERR(std::string("Dimensions mismatch.  Number of 'Functions' and 'Weights' do not match. ") +
                     "Check scalar function with name '" + mFuncName + "'.")
        }

        if(mCriterionNames.size() != mCriterionNormalizations.size())
        {
            THROWERR(std::string("Dimensions mismatch.  Number of 'Functions' and 'Normalizations' do not match. ") +
                     "Check scalar function with name '" + mFuncName + "'.")
        }
    }

    void initialize(Teuchos::ParameterList & aInputs)
    {
        if(aInputs.sublist("Criteria").isSublist(mFuncName) == false)
        {
            THROWERR(std::string("Scalar function with tag '") + mFuncName + "' is not defined in the input file.")
        }

        auto tCriteriaInputs = aInputs.sublist("Criteria").sublist(mFuncName);
        this->parseFunctions(tCriteriaInputs);
        this->parseWeights(tCriteriaInputs);
        this->parseNormalization(tCriteriaInputs);
        this->checkInputs();

        Plato::Fluids::CriterionFactory<PhysicsT> tFactory;
        for(auto& tName : mCriterionNames)
        {
            auto tScalarFunction = tFactory.createCriterion(mSpatialModel, mDataMap, aInputs, tName);
            mCriteria.push_back(tScalarFunction);
        }
    }

    void parseFunctions(Teuchos::ParameterList & aInputs)
    {
        mCriterionNames = Plato::parse_array<std::string>("Functions", aInputs);
        if(mCriterionNames.empty())
        {
            THROWERR(std::string("'Functions' keyword was not defined in function block with name '") + mFuncName
                + "'. User must define 'functions' used in a 'Weighted Sum' criterion.")
        }
    }

    void parseWeights(Teuchos::ParameterList & aInputs)
    {
        mCriterionWeights = Plato::parse_array<Plato::Scalar>("Weights", aInputs);
        if(mCriterionNames.empty())
        {
            THROWERR(std::string("'Weights' keyword was not defined in function block with name '") + mFuncName
                + "'. User must define 'weights' used in a 'Weighted Sum' criterion.")
        }
    }

    void parseNormalization(Teuchos::ParameterList & aInputs)
    {
        mCriterionNormalizations = Plato::parse_array<Plato::Scalar>("Normalizations", aInputs);
        if(mCriterionNormalizations.empty())
        {
            mCriterionNormalizations.resize(mCriterionNames.size());
            std::fill(mCriterionNormalizations.begin(), mCriterionNormalizations.end(), 1.0);
        }
    }
};
// class LeastSquares

}
// namespace Fluids


// todo: physics types
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class MomentumConservation : public Plato::SimplexFluids<SpaceDim, NumControls>
{
public:
    typedef Plato::Fluids::FunctionFactory FunctionFactory;
    using SimplexT = Plato::SimplexFluids<SpaceDim, NumControls>;

    static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexT::mNumMomentumDofsPerNode;
    static constexpr Plato::OrdinalType mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class MassConservation : public Plato::SimplexFluids<SpaceDim, NumControls>
{
public:
    typedef Plato::Fluids::FunctionFactory FunctionFactory;
    using SimplexT = Plato::SimplexFluids<SpaceDim, NumControls>;

    static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexT::mNumMassDofsPerNode;
    static constexpr Plato::OrdinalType mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class EnergyConservation : public Plato::SimplexFluids<SpaceDim, NumControls>
{
public:
    typedef Plato::Fluids::FunctionFactory FunctionFactory;
    using SimplexT = Plato::SimplexFluids<SpaceDim, NumControls>;

    static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexT::mNumEnergyDofsPerNode;
    static constexpr Plato::OrdinalType mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class IncompressibleFluids : public Plato::SimplexFluids<SpaceDim, NumControls>
{
public:
    static constexpr auto mNumSpatialDims = SpaceDim;

    typedef Plato::Fluids::FunctionFactory FunctionFactory;
    using SimplexT = typename Plato::SimplexFluids<SpaceDim, NumControls>;

    using MassPhysicsT     = typename Plato::MassConservation<SpaceDim, NumControls>;
    using EnergyPhysicsT   = typename Plato::EnergyConservation<SpaceDim, NumControls>;
    using MomentumPhysicsT = typename Plato::MomentumConservation<SpaceDim, NumControls>;
};




// todo: unit test inline functions
namespace cbs
{

template<Plato::OrdinalType NumSpatialDims,
         Plato::OrdinalType NumNodesPerCell>
inline Plato::ScalarVector
calculate_element_characteristic_sizes
(const Plato::SpatialModel & aSpatialModel)
{
    auto tCoords = aSpatialModel.Mesh.coords();
    auto tCells2Nodes = aSpatialModel.Mesh.ask_elem_verts();

    Plato::OrdinalType tNumCells = aSpatialModel.Mesh.nelems();
    Plato::OrdinalType tNumNodes = aSpatialModel.Mesh.nverts();
    Plato::ScalarVector tElemCharSize("element characteristic size", tNumNodes);
    Plato::blas1::fill(std::numeric_limits<Plato::Scalar>::max(), tElemCharSize);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tElemSize = Plato::omega_h::calculate_element_size<NumSpatialDims,NumNodesPerCell>(aCellOrdinal, tCells2Nodes, tCoords);
        for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
        {
            auto tVertexIndex = tCells2Nodes[aCellOrdinal*NumNodesPerCell + tNode];
            tElemCharSize(tVertexIndex) = tElemSize <= tElemCharSize(tVertexIndex) ? tElemSize : tElemCharSize(tVertexIndex);
        }
    },"calculate characteristic element size");

    return tElemCharSize;
}

template<Plato::OrdinalType NodesPerCell>
Plato::ScalarVector
calculate_convective_velocity_magnitude
(const Plato::SpatialModel & aSpatialModel,
 const Plato::ScalarVector & aVelocityField)
{
    auto tCell2Node = aSpatialModel.Mesh.ask_elem_verts();
    Plato::OrdinalType tSpaceDim = aSpatialModel.Mesh.dim();
    Plato::OrdinalType tNumCells = aSpatialModel.Mesh.nelems();
    Plato::OrdinalType tNumNodes = aSpatialModel.Mesh.nverts();

    Plato::ScalarVector tConvectiveVelocity("convective velocity", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCell)
    {
        for(Plato::OrdinalType tNode = 0; tNode < NodesPerCell; tNode++)
        {
            Plato::Scalar tSum = 0.0;
            Plato::OrdinalType tVertexIndex = tCell2Node[aCell*NodesPerCell + tNode];
            for(Plato::OrdinalType tDim = 0; tDim < tSpaceDim; tDim++)
            {
                auto tDofIndex = tVertexIndex * tSpaceDim + tDim;
                tSum += aVelocityField(tDofIndex) * aVelocityField(tDofIndex);
            }
            auto tMyValue = sqrt(tSum);
            tConvectiveVelocity(tVertexIndex) =
                tMyValue >= tConvectiveVelocity(tVertexIndex) ? tMyValue : tConvectiveVelocity(tVertexIndex);
        }
    }, "calculate_convective_velocity_magnitude");

    return tConvectiveVelocity;
}

template<Plato::OrdinalType NodesPerCell>
Plato::ScalarVector
calculate_diffusive_velocity_magnitude
(const Plato::SpatialModel & aSpatialModel,
 const Plato::Scalar & aReynoldsNum,
 const Plato::ScalarVector & aCharElemSize)
{
    auto tCell2Node = aSpatialModel.Mesh.ask_elem_verts();
    Plato::OrdinalType tNumCells = aSpatialModel.Mesh.nelems();
    Plato::OrdinalType tNumNodes = aSpatialModel.Mesh.nverts();

    Plato::ScalarVector tDiffusiveVelocity("diffusive velocity", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(decltype(tNumNodes) tNode = 0; tNode < NodesPerCell; tNode++)
        {
            Plato::OrdinalType tVertexIndex = tCell2Node[aCellOrdinal*NodesPerCell + tNode];
            auto tMyValue = static_cast<Plato::Scalar>(1.0) / (aCharElemSize(tVertexIndex) * aReynoldsNum);
            tDiffusiveVelocity(tVertexIndex) = 
                tMyValue >= tDiffusiveVelocity(tVertexIndex) ? tMyValue : tDiffusiveVelocity(tVertexIndex);
        }
    }, "calculate_diffusive_velocity_magnitude");

    return tDiffusiveVelocity;
}

template<Plato::OrdinalType NodesPerCell>
Plato::ScalarVector
calculate_thermal_velocity_magnitude
(const Plato::SpatialModel & aSpatialModel,
 const Plato::Scalar & aPrandtlNum,
 const Plato::Scalar & aReynoldsNum,
 const Plato::ScalarVector & aCharElemSize)
{
    auto tCell2Node = aSpatialModel.Mesh.ask_elem_verts();
    Plato::OrdinalType tNumCells = aSpatialModel.Mesh.nelems();
    Plato::OrdinalType tNumNodes = aSpatialModel.Mesh.nverts();

    Plato::ScalarVector tThermalVelocity("thermal velocity", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(decltype(tNumNodes) tNode = 0; tNode < NodesPerCell; tNode++)
        {
            Plato::OrdinalType tVertexIndex = tCell2Node[aCellOrdinal*NodesPerCell + tNode];
            auto tMyValue = static_cast<Plato::Scalar>(1.0) / (aCharElemSize(tVertexIndex) * aReynoldsNum * aPrandtlNum);
            tThermalVelocity(tVertexIndex) =
                tMyValue >= tThermalVelocity(tVertexIndex) ? tMyValue : tThermalVelocity(tVertexIndex);
        }
    }, "calculate_thermal_velocity_magnitude");

    return tThermalVelocity;
}

/***************************************************************************//**
 *  \brief Calculate artificial compressibility for incompressible flow problems.
 *  The artificial compressibility is computed as follows:
 *  \f$ \beta=\max(\epsilon,u_{convective},u_{diffusive},u_{thermal}) \f$,
 *  where
 *  \f$ u_{convective} = \sqrt(u_iu_i),\quad\in\{1,\dots,\mbox{dim}\} \f$
 *  \f$ u_{diffusive}  = \frac{1.0}{h_e\mbox{Re}} \f$
 *  \f$ u_{thermal}    = \frac{1.0}{h_e\mbox{Re}\mbox{Pr}} \f$
 *  Here, $h_e$ is the $e$-th element characteristic length, Re is the Reynolds number,
 *  Pr is the Prandtl number
 *
 *  \param aStates                [in] metadata structure with current set of primal states
 *  \param aCritialCompresibility [in] artificial compressibility lower bound
 *
 ******************************************************************************/
inline Plato::ScalarVector
calculate_artificial_compressibility
(const Plato::ScalarVector & aConvectiveVelocity,
 const Plato::ScalarVector & aDiffusiveVelocity,
 const Plato::ScalarVector & aThermalVelocity,
 Plato::Scalar aCritialCompresibility = 0.5)
{
    Plato::OrdinalType tNumNodes = aThermalVelocity.size();
    Plato::ScalarVector tArtificialCompressibility("artificial compressibility", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNode)
    {
        auto tMyArtificialCompressibility = (aConvectiveVelocity(aNode) >= aDiffusiveVelocity(aNode))
            && (aConvectiveVelocity(aNode) >= aThermalVelocity(aNode))
            && (aConvectiveVelocity(aNode) >= aCritialCompresibility) ?
                aConvectiveVelocity(aNode) : aCritialCompresibility;

        tMyArtificialCompressibility = (aDiffusiveVelocity(aNode) >= aConvectiveVelocity(aNode) )
            && (aDiffusiveVelocity(aNode) >= aThermalVelocity(aNode))
            && (aDiffusiveVelocity(aNode) >= aCritialCompresibility) ?
                aDiffusiveVelocity(aNode) : tMyArtificialCompressibility;

        tMyArtificialCompressibility = (aThermalVelocity(aNode) >= aConvectiveVelocity(aNode) )
            && (aThermalVelocity(aNode) >= aDiffusiveVelocity(aNode))
            && (aThermalVelocity(aNode) >= aCritialCompresibility) ?
                aThermalVelocity(aNode) : tMyArtificialCompressibility;

        tArtificialCompressibility(aNode) = tMyArtificialCompressibility;
    }, "calculate_artificial_compressibility");

    return tArtificialCompressibility;
}

inline Plato::ScalarVector
calculate_stable_time_step
(const Plato::SpatialModel & aSpatialModel,
 const Plato::ScalarVector & aElemCharSize,
 const Plato::ScalarVector & aVelocityField,
 const Plato::ScalarVector & aArtificialCompress,
 Plato::Scalar aSafetyFactor = 0.5)
{
    auto tNumNodes = aSpatialModel.Mesh.nverts();
    Plato::ScalarVector tTimeStep("time step", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tTimeStep(aNodeOrdinal) = aSafetyFactor * ( aElemCharSize(aNodeOrdinal) /
            ( aVelocityField(aNodeOrdinal) + aArtificialCompress(aNodeOrdinal) ) );
    }, "calculate stable time step");
    return tTimeStep;
}

inline void
enforce_boundary_condition
(const Plato::LocalOrdinalVector & aBcDofs,
 const Plato::ScalarVector       & aBcValues,
 Plato::ScalarVector             & aState)
{
    auto tLength = aBcValues.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        auto tDOF = aBcDofs(aOrdinal);
        aState(tDOF) = aBcValues(aOrdinal);
    }, "enforce boundary condition");
}

inline Plato::ScalarVector
calculate_pressure_residual
(const Plato::ScalarVector& aTimeStep,
 const Plato::ScalarVector& aCurrentPressure,
 const Plato::ScalarVector& aPreviousPressure,
 const Plato::ScalarVector& aArtificialCompress)
{
    // calculate stopping criterion, which is defined as
    // \frac{1}{\beta^2} \left( \frac{p^{n} - p^{n-1}}{\Delta{t}}\right ),
    // where \beta denotes the artificial compressibility
    auto tNumNodes = aCurrentPressure.size();
    Plato::ScalarVector tResidual("pressure residual", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        auto tDeltaPressOverTimeStep = ( aCurrentPressure(aNodeOrdinal) - aPreviousPressure(aNodeOrdinal) ) / aTimeStep(aNodeOrdinal);
        auto tOneOverBetaSquared = static_cast<Plato::Scalar>(1) / ( aArtificialCompress(aNodeOrdinal) * aArtificialCompress(aNodeOrdinal) );
        tResidual(aNodeOrdinal) = tOneOverBetaSquared * tDeltaPressOverTimeStep;
    }, "calculate stopping criterion");

    return tResidual;
}

inline Plato::Scalar
calculate_residual_euclidean_norm
(const Plato::Variables & aStates)
{
    auto tTimeStep = aStates.vector("time steps");
    auto tCurrentPressure = aStates.vector("current pressure");
    auto tPreviousPressure = aStates.vector("previous pressure");
    auto tArtificialCompress = aStates.vector("artificial compressibility");
    auto tResidual = Plato::cbs::calculate_pressure_residual(tTimeStep, tCurrentPressure, tPreviousPressure, tArtificialCompress);
    auto tValue = Plato::blas1::norm(tResidual);
    return tValue;
}

inline Plato::Scalar
calculate_residual_inf_norm
(const Plato::Variables & aStates)
{
    std::vector<Plato::Scalar> tErrors;

    // pressure error
    auto tTimeStep = aStates.vector("time steps");
    auto tCurrentState = aStates.vector("current pressure");
    auto tPreviousState = aStates.vector("previous pressure");
    auto tArtificialCompress = aStates.vector("artificial compressibility");
    auto tMyResidual = Plato::cbs::calculate_pressure_residual(tTimeStep, tCurrentState, tPreviousState, tArtificialCompress);
    Plato::Scalar tInfinityNorm = 0.0;
    Plato::blas1::max(tMyResidual, tInfinityNorm);
    return tInfinityNorm;
}

}
// namespace cbs

namespace Fluids
{

class AbstractProblem
{
public:
    virtual ~AbstractProblem() {}

    virtual void output(std::string aFilePath) = 0;
    virtual const Plato::DataMap& getDataMap() const = 0;
    virtual Plato::Solutions solution(const Plato::ScalarVector& aControl) = 0;
    virtual Plato::Scalar criterionValue(const Plato::ScalarVector & aControl, const std::string& aName) = 0;
    virtual Plato::ScalarVector criterionGradient(const Plato::ScalarVector &aControl, const std::string &aName) = 0;
    virtual Plato::ScalarVector criterionGradientX(const Plato::ScalarVector &aControl, const std::string &aName) = 0;
};

template<typename PhysicsT>
class SteadyState : public Plato::Fluids::AbstractProblem
{
private:
    static constexpr auto mNumSpatialDims     = PhysicsT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell    = PhysicsT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerNode  = PhysicsT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumPresDofsPerNode = PhysicsT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumTempDofsPerNode = PhysicsT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */

    Plato::DataMap      mDataMap;
    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    bool mIsExplicitSolve = true;
    bool mCalculateHeatTransfer = false;
    Plato::Scalar mPrandtlNumber = 1.0;
    Plato::Scalar mReynoldsNumber = 1.0;
    Plato::Scalar mCBSsolverTolerance = 1e-5;
    Plato::Scalar mTimeStepSafetyFactor = 0.5; /*!< safety factor applied to stable time step */
    Plato::OrdinalType mMaxNumIterations = 1e3; /*!< maximum number of steady state iterations */

    Plato::ScalarMultiVector mPressure;
    Plato::ScalarMultiVector mVelocity;
    Plato::ScalarMultiVector mPredictor;
    Plato::ScalarMultiVector mTemperature;

    Plato::Fluids::VectorFunction<typename PhysicsT::MassPhysicsT>     mPressureResidual;
    Plato::Fluids::VectorFunction<typename PhysicsT::MomentumPhysicsT> mPredictorResidual;
    Plato::Fluids::VectorFunction<typename PhysicsT::MomentumPhysicsT> mVelocityResidual;
    // Using pointer since default VectorFunction constructor allocations are not permitted.
    // Temperature VectorFunction allocation is optional since heat transfer calculations are optional
    std::shared_ptr<Plato::Fluids::VectorFunction<typename PhysicsT::EnergyPhysicsT>> mTemperatureResidual;

    using Criterion = std::shared_ptr<Plato::Fluids::CriterionBase>;
    using Criteria  = std::unordered_map<std::string, Criterion>;
    Criteria mCriteria;

    using MassConservationT     = typename Plato::MassConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>;
    using EnergyConservationT   = typename Plato::EnergyConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>;
    using MomentumConservationT = typename Plato::MomentumConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>;
    Plato::EssentialBCs<MassConservationT>     mPressureEssentialBCs;
    Plato::EssentialBCs<MomentumConservationT> mVelocityEssentialBCs;
    Plato::EssentialBCs<EnergyConservationT>   mTemperatureEssentialBCs;

    std::shared_ptr<Plato::AbstractSolver> mVectorFieldSolver;
    std::shared_ptr<Plato::AbstractSolver> mScalarFieldSolver;

public:
    SteadyState
    (Omega_h::Mesh          & aMesh,
     Omega_h::MeshSets      & aMeshSets,
     Teuchos::ParameterList & aInputs,
     Comm::Machine          & aMachine) :
        mSpatialModel(aMesh, aMeshSets, aInputs),
        mPressureResidual("Pressure", mSpatialModel, mDataMap, aInputs),
        mVelocityResidual("Velocity", mSpatialModel, mDataMap, aInputs),
        mPredictorResidual("Velocity Predictor", mSpatialModel, mDataMap, aInputs),
        mPressureEssentialBCs(aInputs.sublist("Pressure Essential Boundary Conditions",false),aMeshSets),
        mVelocityEssentialBCs(aInputs.sublist("Momentum Essential Boundary Conditions",false),aMeshSets),
        mTemperatureEssentialBCs(aInputs.sublist("Temperature Essential Boundary Conditions",false),aMeshSets)
    {
        this->initialize(aInputs, aMachine);
    }

    const decltype(mDataMap)& getDataMap() const
    {
        return mDataMap;
    }

    void output(std::string aFilePath = "output")
    {
        auto tMesh = mSpatialModel.Mesh;
        auto tWriter = Omega_h::vtk::Writer(aFilePath.c_str(), &tMesh, mNumSpatialDims);

        constexpr auto tStride = 0;
        constexpr auto tCurrentTimeStep = 1;
        const auto tNumNodes = tMesh.nverts();

        auto tPressSubView = Kokkos::subview(mPressure, tCurrentTimeStep, Kokkos::ALL());
        Omega_h::Write<Omega_h::Real> tPressure(tPressSubView.size(), "Pressure");
        Plato::copy<mNumPresDofsPerNode, mNumPresDofsPerNode>(tStride, tNumNodes, tPressSubView, tPressure);
        tMesh.add_tag(Omega_h::VERT, "Pressure", mNumPresDofsPerNode, Omega_h::Reals(tPressure));

        auto tVelSubView = Kokkos::subview(mVelocity, tCurrentTimeStep, Kokkos::ALL());
        Omega_h::Write<Omega_h::Real> tVelocity(tVelSubView.size(), "Velocity");
        Plato::copy<mNumVelDofsPerNode, mNumVelDofsPerNode>(tStride, tNumNodes, tVelSubView, tVelocity);
        tMesh.add_tag(Omega_h::VERT, "Velocity", mNumVelDofsPerNode, Omega_h::Reals(tVelocity));

        if(mCalculateHeatTransfer)
        {
            auto tTempSubView = Kokkos::subview(mTemperature, tCurrentTimeStep, Kokkos::ALL());
            Omega_h::Write<Omega_h::Real> tTemperature(tTempSubView.size(), "Temperature");
            Plato::copy<mNumTempDofsPerNode, mNumTempDofsPerNode>(tStride, tNumNodes, tTempSubView, tTemperature);
            tMesh.add_tag(Omega_h::VERT, "Temperature", mNumTempDofsPerNode, Omega_h::Reals(tTemperature));
        }

        auto tTags = Omega_h::vtk::get_all_vtk_tags(&tMesh, mNumSpatialDims);
        auto tTime = static_cast<Plato::Scalar>(tCurrentTimeStep);
        tWriter.write(tCurrentTimeStep, tTime, tTags);
    }

    Plato::Solutions solution
    (const Plato::ScalarVector& aControl)
    {
        this->checkEssentialBoundaryConditions();

        Plato::Primal tPrimalVars;
        this->calculateElemCharacteristicSize(tPrimalVars);
        for(Plato::OrdinalType tIteration = 0; tIteration < mMaxNumIterations; tIteration++)
        {
            tPrimalVars.scalar("iteration", tIteration);
            this->setPrimalVariables(tPrimalVars);
            this->calculateStableTimeSteps(tPrimalVars);

            this->updatePredictor(aControl, tPrimalVars);
            this->updatePressure(aControl, tPrimalVars);
            this->updateVelocity(aControl, tPrimalVars);
            this->enforceVelocityBoundaryConditions(tPrimalVars);
            this->enforcePressureBoundaryConditions(tPrimalVars);

            // todo: verify BC enforcement
            if(mCalculateHeatTransfer)
            {
                this->updateTemperature(aControl, tPrimalVars);
                this->enforceTemperatureBoundaryConditions(tPrimalVars);
            }

            if(this->checkStoppingCriteria(tPrimalVars))
            {
                break;
            }
            tIteration++;
        }

        Plato::Solutions tSolution;
        tSolution.set("mass state", mPressure);
        tSolution.set("energy state", mTemperature);
        tSolution.set("momentum state", mVelocity);
        return tSolution;
    }

    Plato::Scalar criterionValue
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            Plato::Primal tPrimalVars;
            this->calculateElemCharacteristicSize(tPrimalVars);

            Plato::Scalar tOutput(0);
            auto tNumTimeSteps = mVelocity.extent(0);
            for (Plato::OrdinalType tStep = 0; tStep < tNumTimeSteps; tStep++)
            {
                tPrimalVars.scalar("step", tStep);
                auto tPressure = Kokkos::subview(mPressure, tStep, Kokkos::ALL());
                auto tVelocity = Kokkos::subview(mVelocity, tStep, Kokkos::ALL());
                auto tTemperature = Kokkos::subview(mTemperature, tStep, Kokkos::ALL());
                tPrimalVars.vector("current pressure", tPressure);
                tPrimalVars.vector("current velocity", tVelocity);
                tPrimalVars.vector("current temperature", tTemperature);
                tOutput += tItr->second->value(aControl, tPrimalVars);
            }
            return tOutput;
        }
        else
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not supported");
        }
    }

    Plato::ScalarVector criterionGradient
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            Plato::Dual tDualVars;
            Plato::Primal tCurPrimalVars, tPrevPrimalVars;
            this->calculateElemCharacteristicSize(tCurPrimalVars);

            auto tLastStepIndex = mVelocity.extent(0) - 1;
            Plato::ScalarVector tTotalDerivative("Total Derivative", aControl.size());
            for(Plato::OrdinalType tStep = tLastStepIndex; tStep >= 0; tStep--)
            {
                tDualVars.scalar("step", tStep);
                tCurPrimalVars.scalar("step", tStep);
                tPrevPrimalVars.scalar("step", tStep + 1);

                this->setDualVariables(tDualVars);
                this->setPrimalVariables(tCurPrimalVars);
                this->setPrimalVariables(tPrevPrimalVars);

                this->calculateStableTimeSteps(tCurPrimalVars);
                this->calculateStableTimeSteps(tPrevPrimalVars);

                this->calculateVelocityAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculateTemperatureAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculatePressureAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculatePredictorAdjoint(aControl, tCurPrimalVars, tDualVars);

                this->calculateGradientControl(aName, aControl, tCurPrimalVars, tDualVars, tTotalDerivative);
            }

            return tTotalDerivative;
        }
        else
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not supported");
        }
    }

    Plato::ScalarVector criterionGradientX
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            Plato::Dual tDualVars;
            Plato::Primal tCurPrimalVars, tPrevPrimalVars;
            this->calculateElemCharacteristicSize(tCurPrimalVars);

            auto tLastStepIndex = mVelocity.extent(0) - 1;
            Plato::ScalarVector tTotalDerivative("Total Derivative", aControl.size());
            for(Plato::OrdinalType tStep = tLastStepIndex; tStep >= 0; tStep--)
            {
                tDualVars.scalar("step", tStep);
                tCurPrimalVars.scalar("step", tStep);
                tPrevPrimalVars.scalar("step", tStep + 1);

                this->setDualVariables(tDualVars);
                this->setPrimalVariables(tCurPrimalVars);
                this->setPrimalVariables(tPrevPrimalVars);

                this->calculateStableTimeSteps(tCurPrimalVars);
                this->calculateStableTimeSteps(tPrevPrimalVars);

                this->calculateVelocityAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculateTemperatureAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculatePressureAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculatePredictorAdjoint(aControl, tCurPrimalVars, tDualVars);

                this->calculateGradientConfig(aName, aControl, tCurPrimalVars, tDualVars, tTotalDerivative);
            }

            return tTotalDerivative;
        }
        else
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not supported");
        }
    }

private:
    void initialize
    (Teuchos::ParameterList & aInputs,
     Comm::Machine          & aMachine)
    {
        this->allocateLinearSolvers(aInputs, aMachine);

        this->allocateCriteriaList(aInputs);
        this->allocateMemberStates(aInputs);
        this->calculateHeatTransfer(aInputs);
        this->readTimeIntegrationInputs(aInputs);
        this->readDimensionlessProperties(aInputs);
    }

    void calculateHeatTransfer
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
        auto tHeatTransfer = Plato::tolower(tTag);
        mCalculateHeatTransfer = tHeatTransfer == "none" ? false : true;

        if(mCalculateHeatTransfer)
        {
            mTemperatureResidual =
                std::make_shared<Plato::Fluids::VectorFunction<typename PhysicsT::EnergyPhysicsT>>("Temperature", mSpatialModel, mDataMap, aInputs);
        }
    }

    void allocateLinearSolvers
    (Teuchos::ParameterList & aInputs,
     Comm::Machine          & aMachine)
    {
        Plato::SolverFactory tSolverFactory(aInputs.sublist("Linear Solver"));
        mVectorFieldSolver = tSolverFactory.create(mSpatialModel.Mesh, aMachine, mNumVelDofsPerNode);
        mScalarFieldSolver = tSolverFactory.create(mSpatialModel.Mesh, aMachine, mNumPresDofsPerNode);
    }

    void readTimeIntegrationInputs
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            auto tArtificialDampingTwo = tTimeIntegration.get<Plato::Scalar>("Artificial Damping Two", 0.0);
            mIsExplicitSolve = tArtificialDampingTwo > static_cast<Plato::Scalar>(0.0) ? false : true;
            mTimeStepSafetyFactor = tTimeIntegration.get<Plato::Scalar>("Safety Factor", 0.5);
        }
    }

    void checkEssentialBoundaryConditions()
    {
        if(mPressureEssentialBCs.empty())
        {
            THROWERR("Pressure essential boundary conditions are not defined.")
        }
        if(mVelocityEssentialBCs.empty())
        {
            THROWERR("Velocity essential boundary conditions are not defined.")
        }
        if(mCalculateHeatTransfer)
        {
            if(mTemperatureEssentialBCs.empty())
            {
                THROWERR("Temperature essential boundary conditions are not defined.")
            }
            if(mTemperatureResidual.use_count() == 0)
            {
                THROWERR("Heat transfer calculation requested but temperature 'Vector Function' is not allocated.")
            }
        }
    }

    void readDimensionlessProperties
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        auto tHyperbolicParamList = aInputs.sublist("Hyperbolic");
        mPrandtlNumber = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolicParamList);
        mReynoldsNumber = Plato::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolicParamList);
    }

    void allocateMemberStates(Teuchos::ParameterList & aInputs)
    {
        constexpr auto tTimeSnapshotsStored = 2;
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        mPressure    = Plato::ScalarMultiVector("Pressure Snapshots", tTimeSnapshotsStored, tNumNodes);
        mTemperature = Plato::ScalarMultiVector("Temperature Snapshots", tTimeSnapshotsStored, tNumNodes);
        mVelocity    = Plato::ScalarMultiVector("Velocity Snapshots", tTimeSnapshotsStored, tNumNodes * mNumVelDofsPerNode);
        mPredictor   = Plato::ScalarMultiVector("Predictor Snapshots", tTimeSnapshotsStored, tNumNodes * mNumVelDofsPerNode);
    }

    void allocateCriteriaList(Teuchos::ParameterList &aInputs)
    {
        if(aInputs.isSublist("Criteria"))
        {
            Plato::Fluids::CriterionFactory<PhysicsT> tScalarFuncFactory;

            auto tCriteriaParams = aInputs.sublist("Criteria");
            for(Teuchos::ParameterList::ConstIterator tIndex = tCriteriaParams.begin(); tIndex != tCriteriaParams.end(); ++tIndex)
            {
                const Teuchos::ParameterEntry& tEntry = tCriteriaParams.entry(tIndex);
                if(tEntry.isList())
                {
                    THROWERR("Parameter in Criteria block is not supported.  Expect lists only.")
                }
                auto tName = tCriteriaParams.name(tIndex);
                auto tCriterion = tScalarFuncFactory.createCriterion(mSpatialModel, mDataMap, aInputs, tName);
                if( tCriterion != nullptr )
                {
                    mCriteria[tName] = tCriterion;
                }
            }
        }
    }

    Plato::Scalar calculateResidualNorm(const Plato::Primal & aVariables)
    {
        Plato::Scalar tCriterionValue(0.0);
        if (mIsExplicitSolve)
        {
            tCriterionValue = Plato::cbs::calculate_residual_euclidean_norm(aVariables);
        }
        else
        {
            tCriterionValue = Plato::cbs::calculate_residual_inf_norm(aVariables);
        }
        return tCriterionValue;
    }

    bool checkStoppingCriteria(const Plato::Primal & aVariables)
    {
        bool tStop = false;
        auto tIteration = aVariables.scalar("iteration");
        auto tCriterionValue = this->calculateResidualNorm(aVariables);
        if (tCriterionValue < mCBSsolverTolerance)
        {
            tStop = true;
        }
        else if (tIteration >= mMaxNumIterations)
        {
            tStop = true;
        }
        return tStop;
    }

    void calculateElemCharacteristicSize(Plato::Primal & aVariables)
    {
        auto tElemCharSizes =
            Plato::cbs::calculate_element_characteristic_sizes<mNumSpatialDims,mNumNodesPerCell>(mSpatialModel);
        aVariables.vector("element characteristic size", tElemCharSizes);
    }

    void calculateStableTimeSteps(Plato::Primal & aVariables)
    {
        auto tCurrentVelocity = aVariables.vector("current velocity");
        auto tElemCharSize = aVariables.vector("element characteristic size");

        auto tConvectiveVel =
            Plato::cbs::calculate_convective_velocity_magnitude<mNumNodesPerCell>(mSpatialModel, tCurrentVelocity);
        auto tDiffusiveVel =
            Plato::cbs::calculate_diffusive_velocity_magnitude<mNumNodesPerCell>(mSpatialModel, mReynoldsNumber, tElemCharSize);
        auto tThermalVel =
            Plato::cbs::calculate_thermal_velocity_magnitude<mNumNodesPerCell>(mSpatialModel, mPrandtlNumber, mReynoldsNumber, tElemCharSize);
        auto tArtificialCompress =
            Plato::cbs::calculate_artificial_compressibility(tConvectiveVel, tDiffusiveVel, tThermalVel);
        aVariables.vector("artificial compressibility", tArtificialCompress);

        auto tTimeStep =
            Plato::cbs::calculate_stable_time_step(mSpatialModel, tElemCharSize, tConvectiveVel, tArtificialCompress, mTimeStepSafetyFactor);
        aVariables.vector("time steps", tTimeStep);
    }

    void enforceVelocityBoundaryConditions(Plato::Primal & aVariables)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mVelocityEssentialBCs.get(tBcDofs, tBcValues);

        auto tCurrentVelocity = aVariables.vector("current velocity");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentVelocity);

        auto tLength = tCurrentVelocity.size();
        Plato::ScalarVector tPrescribedVelocities("prescribed velocity", tLength);
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tPrescribedVelocities);
        aVariables.vector("prescribed velocity", tPrescribedVelocities);
    }

    void enforcePressureBoundaryConditions(Plato::Primal& aVariables)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mPressureEssentialBCs.get(tBcDofs, tBcValues);
        auto tCurrentPressure = aVariables.vector("current pressure");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentPressure);
    }

    void enforceTemperatureBoundaryConditions(Plato::Primal & aVariables)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mTemperatureEssentialBCs.get(tBcDofs, tBcValues);
        auto tCurrentTemperature = aVariables.vector("current temperature");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentTemperature);
    }

    void setDualVariables(Plato::Dual & aVariables)
    {
        if(aVariables.isVectorMapEmpty())
        {
            // FIRST BACKWARD TIME INTEGRATION STEP
            auto tTotalNumNodes = mSpatialModel.Mesh.nverts();
            std::vector<std::string> tNames =
                {"current pressure adjoint" , "current temperature adjoint",
                "previous pressure adjoint", "previous temperature adjoint"};
            for(auto& tName : tNames)
            {
                Plato::ScalarVector tView(tName, tTotalNumNodes);
                aVariables.vector(tName, tView);
            }

            auto tTotalNumDofs = mNumVelDofsPerNode * tTotalNumNodes;
            tNames = {"current velocity adjoint" , "current predictor adjoint" ,
                      "previous velocity adjoint", "previous predictor adjoint"};
            for(auto& tName : tNames)
            {
                Plato::ScalarVector tView(tName, tTotalNumDofs);
                aVariables.vector(tName, tView);
            }
        }
        else
        {
            // N-TH BACKWARD TIME INTEGRATION STEP
            std::vector<std::string> tNames =
                {"pressure adjoint", "temperature adjoint", "velocity adjoint", "predictor adjoint" };
            for(auto& tName : tNames)
            {
                auto tVector = aVariables.vector(std::string("current ") + tName);
                aVariables.vector(std::string("previous ") + tName, tVector);
            }
        }
    }

    void setPrimalVariables(Plato::Primal & aVariables)
    {
        constexpr Plato::OrdinalType tCurrentStep = 1;
        auto tCurrentVel   = Kokkos::subview(mVelocity, tCurrentStep, Kokkos::ALL());
        auto tCurrentPred  = Kokkos::subview(mPredictor, tCurrentStep, Kokkos::ALL());
        auto tCurrentTemp  = Kokkos::subview(mTemperature, tCurrentStep, Kokkos::ALL());
        auto tCurrentPress = Kokkos::subview(mPressure, tCurrentStep, Kokkos::ALL());
        aVariables.vector("current velocity", tCurrentVel);
        aVariables.vector("current pressure", tCurrentPress);
        aVariables.vector("current temperature", tCurrentTemp);
        aVariables.vector("current predictor", tCurrentPred);

        constexpr auto tPrevStep = tCurrentStep - 1;
        if (tPrevStep >= static_cast<Plato::OrdinalType>(0))
        {
            auto tPreviouVel    = Kokkos::subview(mVelocity, tPrevStep, Kokkos::ALL());
            auto tPreviousPred  = Kokkos::subview(mPredictor, tPrevStep, Kokkos::ALL());
            auto tPreviousTemp  = Kokkos::subview(mTemperature, tPrevStep, Kokkos::ALL());
            auto tPreviousPress = Kokkos::subview(mPressure, tPrevStep, Kokkos::ALL());
            aVariables.vector("previous velocity", tPreviouVel);
            aVariables.vector("previous predictor", tPreviousPred);
            aVariables.vector("previous pressure", tPreviousPress);
            aVariables.vector("previous temperature", tPreviousTemp);
        }
        else
        {
            auto tLength = mPressure.extent(1);
            Plato::ScalarVector tPreviousPress("previous pressure", tLength);
            aVariables.vector("previous pressure", tPreviousPress);
            tLength = mTemperature.extent(1);
            Plato::ScalarVector tPreviousTemp("previous temperature", tLength);
            aVariables.vector("previous temperature", tPreviousTemp);
            tLength = mVelocity.extent(1);
            Plato::ScalarVector tPreviousVel("previous velocity", tLength);
            aVariables.vector("previous velocity", tPreviousVel);
            tLength = mPredictor.extent(1);
            Plato::ScalarVector tPreviousPred("previous predictor", tLength);
            aVariables.vector("previous previous predictor", tPreviousPred);
        }
    }

    void updateVelocity
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aVariables)
    {
        // calculate current residual and jacobian matrix
        auto tResidualVelocity = mVelocityResidual.value(aControl, aVariables);
        auto tJacobianVelocity = mVelocityResidual.gradientCurrentVel(aControl, aVariables);

        // solve velocity equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaVelocity("increment", tResidualVelocity.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaVelocity);
        mVectorFieldSolver->solve(*tJacobianVelocity, tDeltaVelocity, tResidualVelocity);

        // update velocity
        auto tCurrentVelocity  = aVariables.vector("current velocity");
        auto tPreviousVelocity = aVariables.vector("previous velocity");
        Plato::blas1::copy(tPreviousVelocity, tCurrentVelocity);
        Plato::blas1::axpy(1.0, tDeltaVelocity, tCurrentVelocity);
    }

    void updatePredictor
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aVariables)
    {
        // calculate current residual and jacobian matrix
        auto tResidualPredictor = mPredictorResidual.value(aControl, aVariables);
        auto tJacobianPredictor = mPredictorResidual.gradientPredictor(aControl, aVariables);

        // solve predictor equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaPredictor("increment", tResidualPredictor.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaPredictor);
        mVectorFieldSolver->solve(*tJacobianPredictor, tDeltaPredictor, tResidualPredictor);

        // update current predictor
        auto tCurrentPredictor  = aVariables.vector("current predictor");
        auto tPreviousPredictor = aVariables.vector("previous predictor");
        Plato::blas1::copy(tPreviousPredictor, tCurrentPredictor);
        Plato::blas1::axpy(1.0, tDeltaPredictor, tCurrentPredictor);
    }

    void updatePressure
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aVariables)
    {
        // calculate current residual and jacobian matrix
        auto tResidualPressure = mPressureResidual.value(aControl, aVariables);
        auto tJacobianPressure = mPressureResidual.gradientCurrentPress(aControl, aVariables);

        // solve mass equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaPressure("increment", tResidualPressure.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaPressure);
        mScalarFieldSolver->solve(*tJacobianPressure, tDeltaPressure, tResidualPressure);

        // update pressure
        auto tCurrentPressure = aVariables.vector("current pressure");
        auto tPreviousPressure = aVariables.vector("previous pressure");
        Plato::blas1::copy(tPreviousPressure, tCurrentPressure);
        Plato::blas1::axpy(1.0, tDeltaPressure, tCurrentPressure);
    }

    void updateTemperature
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aVariables)
    {
        // calculate current residual and jacobian matrix
        auto tResidualTemperature = mTemperatureResidual->value(aControl, aVariables);
        auto tJacobianTemperature = mTemperatureResidual->gradientCurrentTemp(aControl, aVariables);

        // solve energy equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaTemperature("increment", tResidualTemperature.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaTemperature);
        mScalarFieldSolver->solve(*tJacobianTemperature, tDeltaTemperature, tResidualTemperature);

        // update temperature
        auto tCurrentTemperature  = aVariables.vector("current temperature");
        auto tPreviousTemperature = aVariables.vector("previous temperature");
        Plato::blas1::copy(tPreviousTemperature, tCurrentTemperature);
        Plato::blas1::axpy(1.0, tDeltaTemperature, tCurrentTemperature);
    }

    void calculatePredictorAdjoint
    (const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
           Plato::Dual         & aDualVars)
    {

        auto tCurrentVelocityAdjoint = aDualVars.vector("current velocity adjoint");
        auto tGradResVelWrtPredictor = mVelocityResidual.gradientPredictor(aControl, aCurPrimalVars);
        Plato::ScalarVector tRHS("right hand side", tCurrentVelocityAdjoint.size());
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtPredictor, tCurrentVelocityAdjoint, tRHS);

        auto tCurrentPressureAdjoint = aDualVars.vector("current pressure adjoint");
        auto tGradResPressWrtPredictor = mPressureResidual.gradientPredictor(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPredictor, tCurrentPressureAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentPredictorAdjoint = aDualVars.vector("current predictor adjoint");
        Plato::blas1::fill(0.0, tCurrentPredictorAdjoint);
        auto tJacobianPredictor = mPredictorResidual.gradientPredictor(aControl, aCurPrimalVars);
        mVectorFieldSolver->solve(*tJacobianPredictor, tCurrentPredictorAdjoint, tRHS);
    }

    void calculatePressureAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
     const Plato::Primal       & aPrevPrimalVars,
           Plato::Dual         & aDualVars)
    {
        auto tRHS = mCriteria[aName]->gradientCurrentPress(aControl, aCurPrimalVars);

        auto tGradResVelWrtCurPress = mVelocityResidual.gradientCurrentPress(aControl, aCurPrimalVars);
        auto tCurrentVelocityAdjoint = aDualVars.vector("current velocity adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtCurPress, tCurrentVelocityAdjoint, tRHS);

        auto tGradResPressWrtPrevPress = mPressureResidual.gradientPreviousPress(aControl, aPrevPrimalVars);
        auto tPrevPressureAdjoint = aDualVars.vector("previous pressure adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPrevPress, tPrevPressureAdjoint, tRHS);

        auto tGradResVelWrtPrevPress = mVelocityResidual.gradientPreviousPress(aControl, aPrevPrimalVars);
        auto tPrevVelocityAdjoint = aDualVars.vector("previous velocity adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtPrevPress, tPrevVelocityAdjoint, tRHS);

        auto tGradResPredWrtPrevPress = mPredictorResidual.gradientPreviousPress(aControl, aPrevPrimalVars);
        auto tPrevPredictorAdjoint = aDualVars.vector("previous predictor adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPrevPress, tPrevPredictorAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentPressAdjoint = aDualVars.vector("current pressure adjoint");
        Plato::blas1::fill(0.0, tCurrentPressAdjoint);
        auto tJacobianPressure = mPressureResidual.gradientCurrentPress(aControl, aCurPrimalVars);
        mScalarFieldSolver->solve(*tJacobianPressure, tCurrentPressAdjoint, tRHS);
    }

    void calculateTemperatureAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
     const Plato::Primal       & aPrevPrimalVars,
           Plato::Dual         & aDualVars)
    {
        auto tRHS = mCriteria[aName]->gradientCurrentTemp(aControl, aCurPrimalVars);

        auto tGradResPredWrtPrevTemp = mPredictorResidual.gradientPreviousTemp(aControl, aPrevPrimalVars);
        auto tPrevPredictorAdjoint = aDualVars.vector("previous predictor adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPrevTemp, tPrevPredictorAdjoint, tRHS);

        auto tGradResTempWrtPrevTemp = mTemperatureResidual->gradientPreviousTemp(aControl, aPrevPrimalVars);
        auto tPrevTempAdjoint = aDualVars.vector("previous temperature adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtPrevTemp, tPrevTempAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentTempAdjoint = aDualVars.vector("current temperature adjoint");
        Plato::blas1::fill(0.0, tCurrentTempAdjoint);
        auto tJacobianTemperature = mTemperatureResidual->gradientCurrentTemp(aControl, aCurPrimalVars);
        mScalarFieldSolver->solve(*tJacobianTemperature, tCurrentTempAdjoint, tRHS);
    }

    void calculateVelocityAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
     const Plato::Primal       & aPrevPrimalVars,
           Plato::Dual         & aDualVars)
    {
        auto tRHS = mCriteria[aName]->gradientCurrentVel(aControl, aCurPrimalVars);

        auto tGradResPredWrtPrevVel = mPredictorResidual.gradientPreviousVel(aControl, aPrevPrimalVars);
        auto tPrevPredictorAdjoint = aDualVars.vector("previous predictor adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPrevVel, tPrevPredictorAdjoint, tRHS);

        auto tGradResVelWrtPrevVel = mVelocityResidual.gradientPreviousVel(aControl, aPrevPrimalVars);
        auto tPrevVelocityAdjoint = aDualVars.vector("previous velocity adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtPrevVel, tPrevVelocityAdjoint, tRHS);

        auto tGradResPressWrtPrevVel = mPressureResidual.gradientPreviousVel(aControl, aPrevPrimalVars);
        auto tPrevPressureAdjoint = aDualVars.vector("previous pressure adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPrevVel, tPrevPressureAdjoint, tRHS);

        auto tGradResTempWrtPrevVel = mTemperatureResidual->gradientPreviousVel(aControl, aPrevPrimalVars);
        auto tPrevTemperatureAdjoint = aDualVars.vector("previous temperature adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtPrevVel, tPrevTemperatureAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentVelocityAdjoint = aDualVars.vector("current velocity adjoint");
        Plato::blas1::fill(0.0, tCurrentVelocityAdjoint);
        auto tJacobianVelocity = mVelocityResidual.gradientCurrentVel(aControl, aCurPrimalVars);
        mVectorFieldSolver->solve(*tJacobianVelocity, tCurrentVelocityAdjoint, tRHS);
    }

    void calculateGradientControl
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
     const Plato::Dual         & aDualVars,
           Plato::ScalarVector & aTotalDerivative)
    {
        auto tGradCriterionWrtControl = mCriteria[aName]->gradientControl(aControl, aCurPrimalVars);

        auto tCurrentPredictorAdjoint = aDualVars.vector("current predictor adjoint");
        auto tGradResPredWrtControl = mPredictorResidual.gradientControl(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtControl, tCurrentPredictorAdjoint, tGradCriterionWrtControl);

        auto tCurrentPressureAdjoint = aDualVars.vector("current pressure adjoint");
        auto tGradResPressWrtControl = mPressureResidual.gradientControl(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtControl, tCurrentPressureAdjoint, tGradCriterionWrtControl);

        auto tCurrentTemperatureAdjoint = aDualVars.vector("current temperature adjoint");
        auto tGradResTempWrtControl = mTemperatureResidual->gradientControl(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtControl, tCurrentTemperatureAdjoint, tGradCriterionWrtControl);

        auto tCurrentVelocityAdjoint = aDualVars.vector("current velocity adjoint");
        auto tGradResVelWrtControl = mVelocityResidual.gradientControl(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtControl, tCurrentVelocityAdjoint, tGradCriterionWrtControl);

        Plato::blas1::axpy(1.0, tGradCriterionWrtControl, aTotalDerivative);
    }

    void calculateGradientConfig
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
     const Plato::Dual         & aDualVars,
           Plato::ScalarVector & aTotalDerivative)
    {
        auto tGradCriterionWrtConfig = mCriteria[aName]->gradientConfig(aControl, aCurPrimalVars);

        auto tCurrentPredictorAdjoint = aDualVars.vector("current predictor adjoint");
        auto tGradResPredWrtConfig = mPredictorResidual.gradientConfig(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtConfig, tCurrentPredictorAdjoint, tGradCriterionWrtConfig);

        auto tCurrentPressureAdjoint = aDualVars.vector("current pressure adjoint");
        auto tGradResPressWrtConfig = mPressureResidual.gradientConfig(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtConfig, tCurrentPressureAdjoint, tGradCriterionWrtConfig);

        auto tCurrentTemperatureAdjoint = aDualVars.vector("current temperature adjoint");
        auto tGradResTempWrtConfig = mTemperatureResidual->gradientConfig(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtConfig, tCurrentTemperatureAdjoint, tGradCriterionWrtConfig);

        auto tCurrentVelocityAdjoint = aDualVars.vector("current velocity adjoint");
        auto tGradResVelWrtConfig = mVelocityResidual.gradientConfig(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtConfig, tCurrentVelocityAdjoint, tGradCriterionWrtConfig);

        Plato::blas1::axpy(1.0, tGradCriterionWrtConfig, aTotalDerivative);
    }
};

template<typename PhysicsT>
class Transient : public Plato::Fluids::AbstractProblem
{
private:
    static constexpr auto mNumSpatialDims     = PhysicsT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell    = PhysicsT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerNode  = PhysicsT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumPresDofsPerNode = PhysicsT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumTempDofsPerNode = PhysicsT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */

    Plato::DataMap      mDataMap;
    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    bool mIsExplicitSolve = true;
    bool mCalculateHeatTransfer = false;
    Plato::Scalar mPrandtlNumber = 1.0;
    Plato::Scalar mReynoldsNumber = 1.0;
    Plato::Scalar mCBSsolverTolerance = 1e-5;
    Plato::Scalar mTimeStepSafetyFactor = 0.5; /*!< safety factor applied to stable time step */
    Plato::OrdinalType mMaxNumTimeSteps = 100;

    Plato::ScalarMultiVector mPressure;
    Plato::ScalarMultiVector mVelocity;
    Plato::ScalarMultiVector mPredictor;
    Plato::ScalarMultiVector mTemperature;

    Plato::Fluids::VectorFunction<typename PhysicsT::MassPhysicsT>     mPressureResidual;
    Plato::Fluids::VectorFunction<typename PhysicsT::MomentumPhysicsT> mPredictorResidual;
    Plato::Fluids::VectorFunction<typename PhysicsT::MomentumPhysicsT> mVelocityResidual;
    // Using pointer since default VectorFunction constructor allocations are not permitted.
    // Temperature VectorFunction allocation is optional since heat transfer calculations are optional
    std::shared_ptr<Plato::Fluids::VectorFunction<typename PhysicsT::EnergyPhysicsT>> mTemperatureResidual;

    using Criterion = std::shared_ptr<Plato::Fluids::CriterionBase>;
    using Criteria  = std::unordered_map<std::string, Criterion>;
    Criteria mCriteria;

    using MassConservationT     = typename Plato::MassConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>;
    using EnergyConservationT   = typename Plato::EnergyConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>;
    using MomentumConservationT = typename Plato::MomentumConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>;
    Plato::EssentialBCs<MassConservationT>     mPressureEssentialBCs;
    Plato::EssentialBCs<MomentumConservationT> mVelocityEssentialBCs;
    Plato::EssentialBCs<EnergyConservationT>   mTemperatureEssentialBCs;

    std::shared_ptr<Plato::AbstractSolver> mVectorFieldSolver;
    std::shared_ptr<Plato::AbstractSolver> mScalarFieldSolver;

public:
    Transient
    (Omega_h::Mesh          & aMesh,
     Omega_h::MeshSets      & aMeshSets,
     Teuchos::ParameterList & aInputs,
     Comm::Machine          & aMachine) :
        mSpatialModel(aMesh, aMeshSets, aInputs),
        mPressureResidual("Pressure", mSpatialModel, mDataMap, aInputs),
        mVelocityResidual("Velocity", mSpatialModel, mDataMap, aInputs),
        mPredictorResidual("Velocity Predictor", mSpatialModel, mDataMap, aInputs),
        mPressureEssentialBCs(aInputs.sublist("Pressure Essential Boundary Conditions",false),aMeshSets),
        mVelocityEssentialBCs(aInputs.sublist("Momentum Essential Boundary Conditions",false),aMeshSets),
        mTemperatureEssentialBCs(aInputs.sublist("Temperature Essential Boundary Conditions",false),aMeshSets)
    {
        this->initialize(aInputs, aMachine);
    }

    const decltype(mDataMap)& getDataMap() const
    {
        return mDataMap;
    }

    void output(std::string aFilePath = "output")
    {
        auto tMesh = mSpatialModel.Mesh;
        const auto tTimeSteps = mVelocity.extent(0);
        auto tWriter = Omega_h::vtk::Writer(aFilePath.c_str(), &tMesh, mNumSpatialDims);

        constexpr auto tStride = 0;
        const auto tNumNodes = tMesh.nverts();
        for(Plato::OrdinalType tStep = 0; tStep < tTimeSteps; tStep++)
        {
            auto tPressSubView = Kokkos::subview(mPressure, tStep, Kokkos::ALL());
            Omega_h::Write<Omega_h::Real> tPressure(tPressSubView.size(), "Pressure");
            Plato::copy<mNumPresDofsPerNode, mNumPresDofsPerNode>(tStride, tNumNodes, tPressSubView, tPressure);
            tMesh.add_tag(Omega_h::VERT, "Pressure", mNumPresDofsPerNode, Omega_h::Reals(tPressure));

            auto tVelSubView = Kokkos::subview(mVelocity, tStep, Kokkos::ALL());
            Omega_h::Write<Omega_h::Real> tVelocity(tVelSubView.size(), "Velocity");
            Plato::copy<mNumVelDofsPerNode, mNumVelDofsPerNode>(tStride, tNumNodes, tVelSubView, tVelocity);
            tMesh.add_tag(Omega_h::VERT, "Velocity", mNumVelDofsPerNode, Omega_h::Reals(tVelocity));

            if(mCalculateHeatTransfer)
            {
                auto tTempSubView = Kokkos::subview(mTemperature, tStep, Kokkos::ALL());
                Omega_h::Write<Omega_h::Real> tTemperature(tTempSubView.size(), "Temperature");
                Plato::copy<mNumTempDofsPerNode, mNumTempDofsPerNode>(tStride, tNumNodes, tTempSubView, tTemperature);
                tMesh.add_tag(Omega_h::VERT, "Temperature", mNumTempDofsPerNode, Omega_h::Reals(tTemperature));

            }

            auto tTags = Omega_h::vtk::get_all_vtk_tags(&tMesh, mNumSpatialDims);
            auto tTime = static_cast<Plato::Scalar>(1.0 / tTimeSteps) * static_cast<Plato::Scalar>(tStep + 1);
            tWriter.write(tStep, tTime, tTags);
        }
    }

    Plato::Solutions solution
    (const Plato::ScalarVector& aControl)
    {
        this->checkEssentialBoundaryConditions();

        Plato::Primal tPrimalVars;
        this->calculateElemCharacteristicSize(tPrimalVars);
        for (Plato::OrdinalType tStep = 1; tStep < mMaxNumTimeSteps; tStep++)
        {
            tPrimalVars.scalar("step", tStep);
            this->setPrimalVariables(tPrimalVars);
            this->calculateStableTimeSteps(tPrimalVars);

            this->updatePredictor(aControl, tPrimalVars);
            this->updatePressure(aControl, tPrimalVars);
            this->updateVelocity(aControl, tPrimalVars);
            this->enforceVelocityBoundaryConditions(tPrimalVars);
            this->enforcePressureBoundaryConditions(tPrimalVars);

            // todo: verify BC enforcement
            if(mCalculateHeatTransfer)
            {
                this->updateTemperature(aControl, tPrimalVars);
                this->enforceTemperatureBoundaryConditions(tPrimalVars);
            }

            if(this->checkStoppingCriteria(tPrimalVars))
            {
                break;
            }
        }

        Plato::Solutions tSolution;
        tSolution.set("mass state", mPressure);
        tSolution.set("energy state", mTemperature);
        tSolution.set("momentum state", mVelocity);
        return tSolution;
    }

    Plato::Scalar criterionValue
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            Plato::Primal tPrimalVars;
            this->calculateElemCharacteristicSize(tPrimalVars);

            Plato::Scalar tOutput(0);
            auto tNumTimeSteps = mVelocity.extent(0);
            for (Plato::OrdinalType tStep = 0; tStep < tNumTimeSteps; tStep++)
            {
                tPrimalVars.scalar("step", tStep);
                auto tPressure = Kokkos::subview(mPressure, tStep, Kokkos::ALL());
                auto tVelocity = Kokkos::subview(mVelocity, tStep, Kokkos::ALL());
                auto tTemperature = Kokkos::subview(mTemperature, tStep, Kokkos::ALL());
                tPrimalVars.vector("current pressure", tPressure);
                tPrimalVars.vector("current velocity", tVelocity);
                tPrimalVars.vector("current temperature", tTemperature);
                tOutput += tItr->second->value(aControl, tPrimalVars);
            }
            return tOutput;
        }
        else
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not supported");
        }
    }

    Plato::ScalarVector criterionGradient
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            Plato::Dual tDualVars;
            Plato::Primal tCurPrimalVars, tPrevPrimalVars;
            this->calculateElemCharacteristicSize(tCurPrimalVars);

            auto tLastStepIndex = mVelocity.extent(0) - 1;
            Plato::ScalarVector tTotalDerivative("Total Derivative", aControl.size());
            for(Plato::OrdinalType tStep = tLastStepIndex; tStep >= 0; tStep--)
            {
                tDualVars.scalar("step", tStep);
                tCurPrimalVars.scalar("step", tStep);
                tPrevPrimalVars.scalar("step", tStep + 1);

                this->setDualVariables(tDualVars);
                this->setPrimalVariables(tCurPrimalVars);
                this->setPrimalVariables(tPrevPrimalVars);

                this->calculateStableTimeSteps(tCurPrimalVars);
                this->calculateStableTimeSteps(tPrevPrimalVars);

                this->calculateVelocityAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculateTemperatureAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculatePressureAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculatePredictorAdjoint(aControl, tCurPrimalVars, tDualVars);

                this->calculateGradientControl(aName, aControl, tCurPrimalVars, tDualVars, tTotalDerivative);
            }

            return tTotalDerivative;
        }
        else
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not supported");
        }
    }

    Plato::ScalarVector criterionGradientX
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            Plato::Dual tDualVars;
            Plato::Primal tCurPrimalVars, tPrevPrimalVars;
            this->calculateElemCharacteristicSize(tCurPrimalVars);

            auto tLastStepIndex = mVelocity.extent(0) - 1;
            Plato::ScalarVector tTotalDerivative("Total Derivative", aControl.size());
            for(Plato::OrdinalType tStep = tLastStepIndex; tStep >= 0; tStep--)
            {
                tDualVars.scalar("step", tStep);
                tCurPrimalVars.scalar("step", tStep);
                tPrevPrimalVars.scalar("step", tStep + 1);

                this->setDualVariables(tDualVars);
                this->setPrimalVariables(tCurPrimalVars);
                this->setPrimalVariables(tPrevPrimalVars);

                this->calculateStableTimeSteps(tCurPrimalVars);
                this->calculateStableTimeSteps(tPrevPrimalVars);

                this->calculateVelocityAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculateTemperatureAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculatePressureAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculatePredictorAdjoint(aControl, tCurPrimalVars, tDualVars);

                this->calculateGradientConfig(aName, aControl, tCurPrimalVars, tDualVars, tTotalDerivative);
            }

            return tTotalDerivative;
        }
        else
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not supported");
        }
    }

private:
    void checkEssentialBoundaryConditions()
    {
        if(mPressureEssentialBCs.empty())
        {
            THROWERR("Pressure essential boundary conditions are not defined.")
        }
        if(mVelocityEssentialBCs.empty())
        {
            THROWERR("Velocity essential boundary conditions are not defined.")
        }
        if(mCalculateHeatTransfer)
        {
            if(mTemperatureEssentialBCs.empty())
            {
                THROWERR("Temperature essential boundary conditions are not defined.")
            }
        }
    }

    void initialize
    (Teuchos::ParameterList & aInputs,
     Comm::Machine          & aMachine)
    {
        this->allocateLinearSolvers(aInputs,aMachine);

        this->allocateCriteriaList(aInputs);
        this->allocateMemberStates(aInputs);
        this->calculateHeatTransfer(aInputs);
        this->readTimeIntegrationInputs(aInputs);
        this->readDimensionlessProperties(aInputs);
    }

    void calculateHeatTransfer
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }

        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
        auto tHeatTransfer = Plato::tolower(tTag);
        mCalculateHeatTransfer = tHeatTransfer == "none" ? false : true;

        if(mCalculateHeatTransfer)
        {
            mTemperatureResidual =
                std::make_shared<Plato::Fluids::VectorFunction<typename PhysicsT::EnergyPhysicsT>>("Temperature", mSpatialModel, mDataMap, aInputs);
        }
    }

    void allocateLinearSolvers
    (Teuchos::ParameterList & aInputs,
     Comm::Machine & aMachine)
    {
        Plato::SolverFactory tSolverFactory(aInputs.sublist("Linear Solver"));
        mVectorFieldSolver = tSolverFactory.create(mSpatialModel.Mesh, aMachine, mNumVelDofsPerNode);
        mScalarFieldSolver = tSolverFactory.create(mSpatialModel.Mesh, aMachine, mNumPresDofsPerNode);
    }

    void readTimeIntegrationInputs
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            auto tArtificialDampingTwo = tTimeIntegration.get<Plato::Scalar>("Artificial Damping Two", 0.0);
            mIsExplicitSolve = tArtificialDampingTwo > static_cast<Plato::Scalar>(0.0) ? false : true;
            mTimeStepSafetyFactor = tTimeIntegration.get<Plato::Scalar>("Safety Factor", 0.5);
        }
    }

    void readDimensionlessProperties
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        mPrandtlNumber = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
        mReynoldsNumber = Plato::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolic);
    }

    void allocateMemberStates(Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            mMaxNumTimeSteps = aInputs.sublist("Time Integration").get<int>("Number Time Steps", 100);
        }
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        mPressure    = Plato::ScalarMultiVector("Pressure Snapshots", mMaxNumTimeSteps, tNumNodes);
        mVelocity    = Plato::ScalarMultiVector("Velocity Snapshots", mMaxNumTimeSteps, tNumNodes * mNumVelDofsPerNode);
        mPredictor   = Plato::ScalarMultiVector("Predictor Snapshots", mMaxNumTimeSteps, tNumNodes * mNumVelDofsPerNode);
        mTemperature = Plato::ScalarMultiVector("Temperature Snapshots", mMaxNumTimeSteps, tNumNodes);
    }

    void allocateCriteriaList(Teuchos::ParameterList &aInputs)
    {
        if(aInputs.isSublist("Criteria"))
        {
            Plato::Fluids::CriterionFactory<PhysicsT> tScalarFuncFactory;

            auto tCriteriaParams = aInputs.sublist("Criteria");
            for(Teuchos::ParameterList::ConstIterator tIndex = tCriteriaParams.begin(); tIndex != tCriteriaParams.end(); ++tIndex)
            {
                const Teuchos::ParameterEntry& tEntry = tCriteriaParams.entry(tIndex);
                if(tEntry.isList())
                {
                    THROWERR("Parameter in Criteria block is not supported.  Expect lists only.")
                }
                auto tName = tCriteriaParams.name(tIndex);
                auto tCriterion = tScalarFuncFactory.createCriterion(mSpatialModel, mDataMap, aInputs, tName);
                if( tCriterion != nullptr )
                {
                    mCriteria[tName] = tCriterion;
                }
            }
        }
    }

    Plato::Scalar calculateResidualNorm(const Plato::Primal & aVariables)
    {
        Plato::Scalar tCriterionValue(0.0);
        if (mIsExplicitSolve)
        {
            tCriterionValue = Plato::cbs::calculate_residual_euclidean_norm(aVariables);
        }
        else
        {
            tCriterionValue = Plato::cbs::calculate_residual_inf_norm(aVariables);
        }
        return tCriterionValue;
    }

    bool checkStoppingCriteria(const Plato::Primal & aVariables)
    {
        bool tStop = false;
        auto tTimeStepIndex = aVariables.scalar("step");
        auto tCriterionValue = this->calculateResidualNorm(aVariables);
        if (tCriterionValue < mCBSsolverTolerance)
        {
            tStop = true;
        }
        else if (tTimeStepIndex >= mMaxNumTimeSteps)
        {
            tStop = true;
        }
        return tStop;
    }

    void calculateElemCharacteristicSize(Plato::Primal & aVariables)
    {
        auto tElemCharSizes =
            Plato::cbs::calculate_element_characteristic_sizes<mNumSpatialDims,mNumNodesPerCell>(mSpatialModel);
        aVariables.vector("element characteristic size", tElemCharSizes);
    }

    void calculateStableTimeSteps(Plato::Primal & aVariables)
    {
        auto tCurrentVelocity = aVariables.vector("current velocity");
        auto tElemCharSize = aVariables.vector("element characteristic size");

        auto tConvectiveVel =
            Plato::cbs::calculate_convective_velocity_magnitude<mNumNodesPerCell>(mSpatialModel, tCurrentVelocity);
        auto tDiffusiveVel =
            Plato::cbs::calculate_diffusive_velocity_magnitude<mNumNodesPerCell>(mSpatialModel, mReynoldsNumber, tElemCharSize);
        auto tThermalVel =
            Plato::cbs::calculate_thermal_velocity_magnitude<mNumNodesPerCell>(mSpatialModel, mPrandtlNumber, mReynoldsNumber, tElemCharSize);
        auto tArtificialCompress =
            Plato::cbs::calculate_artificial_compressibility(tConvectiveVel, tDiffusiveVel, tThermalVel);
        aVariables.vector("artificial compressibility", tArtificialCompress);

        auto tTimeStep =
            Plato::cbs::calculate_stable_time_step(mSpatialModel, tElemCharSize, tConvectiveVel, tArtificialCompress, mTimeStepSafetyFactor);
        aVariables.vector("time steps", tTimeStep);
    }

    void enforceVelocityBoundaryConditions(Plato::Primal & aVariables)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mVelocityEssentialBCs.get(tBcDofs, tBcValues);

        auto tCurrentVelocity = aVariables.vector("current velocity");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentVelocity);

        auto tLength = tCurrentVelocity.size();
        Plato::ScalarVector tCurrentPrescribedVelocities("prescribed velocity", tLength);
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentPrescribedVelocities);
        aVariables.vector("prescribed velocity", tCurrentPrescribedVelocities);
    }

    void enforcePressureBoundaryConditions(Plato::Primal& aVariables)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mPressureEssentialBCs.get(tBcDofs, tBcValues);
        auto tCurrentPressure = aVariables.vector("current pressure");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentPressure);
    }

    void enforceTemperatureBoundaryConditions(Plato::Primal & aVariables)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mTemperatureEssentialBCs.get(tBcDofs, tBcValues);
        auto tCurrentTemperature = aVariables.vector("current temperature");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentTemperature);
    }

    void setDualVariables(Plato::Dual & aVariables)
    {
        if(aVariables.isVectorMapEmpty())
        {
            // FIRST BACKWARD TIME INTEGRATION STEP
            auto tTotalNumNodes = mSpatialModel.Mesh.nverts();
            std::vector<std::string> tNames =
                {"current pressure adjoint" , "current temperature adjoint",
                "previous pressure adjoint", "previous temperature adjoint"};
            for(auto& tName : tNames)
            {
                Plato::ScalarVector tView(tName, tTotalNumNodes);
                aVariables.vector(tName, tView);
            }

            auto tTotalNumDofs = mNumVelDofsPerNode * tTotalNumNodes;
            tNames = {"current velocity adjoint" , "current predictor adjoint" ,
                      "previous velocity adjoint", "previous predictor adjoint"};
            for(auto& tName : tNames)
            {
                Plato::ScalarVector tView(tName, tTotalNumDofs);
                aVariables.vector(tName, tView);
            }
        }
        else
        {
            // N-TH BACKWARD TIME INTEGRATION STEP
            std::vector<std::string> tNames =
                {"pressure adjoint", "temperature adjoint", "velocity adjoint", "predictor adjoint" };
            for(auto& tName : tNames)
            {
                auto tVector = aVariables.vector(std::string("current ") + tName);
                aVariables.vector(std::string("previous ") + tName, tVector);
            }
        }
    }



    void setPrimalVariables(Plato::Primal & aVariables)
    {
        Plato::OrdinalType tStep = aVariables.scalar("step");
        auto tCurrentVel   = Kokkos::subview(mVelocity, tStep, Kokkos::ALL());
        auto tCurrentPred  = Kokkos::subview(mPredictor, tStep, Kokkos::ALL());
        auto tCurrentTemp  = Kokkos::subview(mTemperature, tStep, Kokkos::ALL());
        auto tCurrentPress = Kokkos::subview(mPressure, tStep, Kokkos::ALL());
        aVariables.vector("current velocity", tCurrentVel);
        aVariables.vector("current pressure", tCurrentPress);
        aVariables.vector("current temperature", tCurrentTemp);
        aVariables.vector("current predictor", tCurrentPred);

        auto tPrevStep = tStep - 1;
        if (tPrevStep >= static_cast<Plato::OrdinalType>(0))
        {
            auto tPreviouVel    = Kokkos::subview(mVelocity, tPrevStep, Kokkos::ALL());
            auto tPreviousPred  = Kokkos::subview(mPredictor, tPrevStep, Kokkos::ALL());
            auto tPreviousTemp  = Kokkos::subview(mTemperature, tPrevStep, Kokkos::ALL());
            auto tPreviousPress = Kokkos::subview(mPressure, tPrevStep, Kokkos::ALL());
            aVariables.vector("previous velocity", tPreviouVel);
            aVariables.vector("previous predictor", tPreviousPred);
            aVariables.vector("previous pressure", tPreviousPress);
            aVariables.vector("previous temperature", tPreviousTemp);
        }
        else
        {
            auto tLength = mPressure.extent(1);
            Plato::ScalarVector tPreviousPress("previous pressure", tLength);
            aVariables.vector("previous pressure", tPreviousPress);
            tLength = mTemperature.extent(1);
            Plato::ScalarVector tPreviousTemp("previous temperature", tLength);
            aVariables.vector("previous temperature", tPreviousTemp);
            tLength = mVelocity.extent(1);
            Plato::ScalarVector tPreviousVel("previous velocity", tLength);
            aVariables.vector("previous velocity", tPreviousVel);
            tLength = mPredictor.extent(1);
            Plato::ScalarVector tPreviousPred("previous predictor", tLength);
            aVariables.vector("previous previous predictor", tPreviousPred);
        }
    }

    void updateVelocity
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aVariables)
    {
        // calculate current residual and jacobian matrix
        auto tResidualVelocity = mVelocityResidual.value(aVariables);
        auto tJacobianVelocity = mVelocityResidual.gradientCurrentVel(aVariables);

        // solve velocity equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaVelocity("increment", tResidualVelocity.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaVelocity);
        mVectorFieldSolver->solve(*tJacobianVelocity, tDeltaVelocity, tResidualVelocity);

        // update velocity
        auto tCurrentVelocity  = aVariables.vector("current velocity");
        auto tPreviousVelocity = aVariables.vector("previous velocity");
        Plato::blas1::copy(tPreviousVelocity, tCurrentVelocity);
        Plato::blas1::axpy(1.0, tDeltaVelocity, tCurrentVelocity);
    }

    void updatePredictor
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aVariables)
    {
        // calculate current residual and jacobian matrix
        auto tResidualPredictor = mPredictorResidual.value(aVariables);
        auto tJacobianPredictor = mPredictorResidual.gradientPredictor(aVariables);

        // solve predictor equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaPredictor("increment", tResidualPredictor.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaPredictor);
        mVectorFieldSolver->solve(*tJacobianPredictor, tDeltaPredictor, tResidualPredictor);

        // update current predictor
        auto tCurrentPredictor  = aVariables.vector("current predictor");
        auto tPreviousPredictor = aVariables.vector("previous predictor");
        Plato::blas1::copy(tPreviousPredictor, tCurrentPredictor);
        Plato::blas1::axpy(1.0, tDeltaPredictor, tCurrentPredictor);
    }

    void updatePressure
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aVariables)
    {
        // calculate current residual and jacobian matrix
        auto tResidualPressure = mPressureResidual.value(aVariables);
        auto tJacobianPressure = mPressureResidual.gradientCurrentPress(aVariables);

        // solve mass equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaPressure("increment", tResidualPressure.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaPressure);
        mScalarFieldSolver->solve(*tJacobianPressure, tDeltaPressure, tResidualPressure);

        // update pressure
        auto tCurrentPressure = aVariables.vector("current pressure");
        auto tPreviousPressure = aVariables.vector("previous pressure");
        Plato::blas1::copy(tPreviousPressure, tCurrentPressure);
        Plato::blas1::axpy(1.0, tDeltaPressure, tCurrentPressure);
    }

    void updateTemperature
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aVariables)
    {
        // calculate current residual and jacobian matrix
        auto tResidualTemperature = mTemperatureResidual->value(aVariables);
        auto tJacobianTemperature = mTemperatureResidual->gradientCurrentTemp(aVariables);

        // solve energy equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaTemperature("increment", tResidualTemperature.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaTemperature);
        mScalarFieldSolver->solve(*tJacobianTemperature, tDeltaTemperature, tResidualTemperature);

        // update temperature
        auto tCurrentTemperature  = aVariables.vector("current temperature");
        auto tPreviousTemperature = aVariables.vector("previous temperature");
        Plato::blas1::copy(tPreviousTemperature, tCurrentTemperature);
        Plato::blas1::axpy(1.0, tDeltaTemperature, tCurrentTemperature);
    }

    void calculatePredictorAdjoint
    (const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
           Plato::Dual         & aDualVars)
    {

        auto tCurrentVelocityAdjoint = aDualVars.vector("current velocity adjoint");
        auto tGradResVelWrtPredictor = mVelocityResidual.gradientPredictor(aControl, aCurPrimalVars);
        Plato::ScalarVector tRHS("right hand side", tCurrentVelocityAdjoint.size());
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtPredictor, tCurrentVelocityAdjoint, tRHS);

        auto tCurrentPressureAdjoint = aDualVars.vector("current pressure adjoint");
        auto tGradResPressWrtPredictor = mPressureResidual.gradientPredictor(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPredictor, tCurrentPressureAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentPredictorAdjoint = aDualVars.vector("current predictor adjoint");
        Plato::blas1::fill(0.0, tCurrentPredictorAdjoint);
        auto tJacobianPredictor = mPredictorResidual.gradientPredictor(aControl, aCurPrimalVars);
        mVectorFieldSolver->solve(*tJacobianPredictor, tCurrentPredictorAdjoint, tRHS);
    }

    void calculatePressureAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
     const Plato::Primal       & aPrevPrimalVars,
           Plato::Dual         & aDualVars)
    {
        auto tRHS = mCriteria[aName]->gradientCurrentPress(aControl, aCurPrimalVars);

        auto tGradResVelWrtCurPress = mVelocityResidual.gradientCurrentPress(aControl, aCurPrimalVars);
        auto tCurrentVelocityAdjoint = aDualVars.vector("current velocity adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtCurPress, tCurrentVelocityAdjoint, tRHS);

        auto tGradResPressWrtPrevPress = mPressureResidual.gradientPreviousPress(aControl, aPrevPrimalVars);
        auto tPrevPressureAdjoint = aDualVars.vector("previous pressure adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPrevPress, tPrevPressureAdjoint, tRHS);

        auto tGradResVelWrtPrevPress = mVelocityResidual.gradientPreviousPress(aControl, aPrevPrimalVars);
        auto tPrevVelocityAdjoint = aDualVars.vector("previous velocity adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtPrevPress, tPrevVelocityAdjoint, tRHS);

        auto tGradResPredWrtPrevPress = mPredictorResidual.gradientPreviousPress(aControl, aPrevPrimalVars);
        auto tPrevPredictorAdjoint = aDualVars.vector("previous predictor adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPrevPress, tPrevPredictorAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentPressAdjoint = aDualVars.vector("current pressure adjoint");
        Plato::blas1::fill(0.0, tCurrentPressAdjoint);
        auto tJacobianPressure = mPressureResidual.gradientCurrentPress(aControl, aCurPrimalVars);
        mScalarFieldSolver->solve(*tJacobianPressure, tCurrentPressAdjoint, tRHS);
    }

    void calculateTemperatureAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
     const Plato::Primal       & aPrevPrimalVars,
           Plato::Dual         & aDualVars)
    {
        auto tRHS = mCriteria[aName]->gradientCurrentTemp(aControl, aCurPrimalVars);

        auto tGradResPredWrtPrevTemp = mPredictorResidual.gradientPreviousTemp(aControl, aPrevPrimalVars);
        auto tPrevPredictorAdjoint = aDualVars.vector("previous predictor adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPrevTemp, tPrevPredictorAdjoint, tRHS);

        auto tGradResTempWrtPrevTemp = mTemperatureResidual->gradientPreviousTemp(aControl, aPrevPrimalVars);
        auto tPrevTempAdjoint = aDualVars.vector("previous temperature adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtPrevTemp, tPrevTempAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentTempAdjoint = aDualVars.vector("current temperature adjoint");
        Plato::blas1::fill(0.0, tCurrentTempAdjoint);
        auto tJacobianTemperature = mTemperatureResidual->gradientCurrentTemp(aControl, aCurPrimalVars);
        mScalarFieldSolver->solve(*tJacobianTemperature, tCurrentTempAdjoint, tRHS);
    }

    void calculateVelocityAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
     const Plato::Primal       & aPrevPrimalVars,
           Plato::Dual         & aDualVars)
    {
        auto tRHS = mCriteria[aName]->gradientCurrentVel(aControl, aCurPrimalVars);

        auto tGradResPredWrtPrevVel = mPredictorResidual.gradientPreviousVel(aControl, aPrevPrimalVars);
        auto tPrevPredictorAdjoint = aDualVars.vector("previous predictor adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPrevVel, tPrevPredictorAdjoint, tRHS);

        auto tGradResVelWrtPrevVel = mVelocityResidual.gradientPreviousVel(aControl, aPrevPrimalVars);
        auto tPrevVelocityAdjoint = aDualVars.vector("previous velocity adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtPrevVel, tPrevVelocityAdjoint, tRHS);

        auto tGradResPressWrtPrevVel = mPressureResidual.gradientPreviousVel(aControl, aPrevPrimalVars);
        auto tPrevPressureAdjoint = aDualVars.vector("previous pressure adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPrevVel, tPrevPressureAdjoint, tRHS);

        auto tGradResTempWrtPrevVel = mTemperatureResidual->gradientPreviousVel(aControl, aPrevPrimalVars);
        auto tPrevTemperatureAdjoint = aDualVars.vector("previous temperature adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtPrevVel, tPrevTemperatureAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentVelocityAdjoint = aDualVars.vector("current velocity adjoint");
        Plato::blas1::fill(0.0, tCurrentVelocityAdjoint);
        auto tJacobianVelocity = mVelocityResidual.gradientCurrentVel(aControl, aCurPrimalVars);
        mVectorFieldSolver->solve(*tJacobianVelocity, tCurrentVelocityAdjoint, tRHS);
    }

    void calculateGradientControl
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
     const Plato::Dual         & aDualVars,
           Plato::ScalarVector & aTotalDerivative)
    {
        auto tGradCriterionWrtControl = mCriteria[aName]->gradientControl(aControl, aCurPrimalVars);

        auto tCurrentPredictorAdjoint = aDualVars.vector("current predictor adjoint");
        auto tGradResPredWrtControl = mPredictorResidual.gradientControl(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtControl, tCurrentPredictorAdjoint, tGradCriterionWrtControl);

        auto tCurrentPressureAdjoint = aDualVars.vector("current pressure adjoint");
        auto tGradResPressWrtControl = mPressureResidual.gradientControl(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtControl, tCurrentPressureAdjoint, tGradCriterionWrtControl);

        auto tCurrentTemperatureAdjoint = aDualVars.vector("current temperature adjoint");
        auto tGradResTempWrtControl = mTemperatureResidual->gradientControl(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtControl, tCurrentTemperatureAdjoint, tGradCriterionWrtControl);

        auto tCurrentVelocityAdjoint = aDualVars.vector("current velocity adjoint");
        auto tGradResVelWrtControl = mVelocityResidual.gradientControl(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtControl, tCurrentVelocityAdjoint, tGradCriterionWrtControl);

        Plato::blas1::axpy(1.0, tGradCriterionWrtControl, aTotalDerivative);
    }

    void calculateGradientConfig
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
     const Plato::Dual         & aDualVars,
           Plato::ScalarVector & aTotalDerivative)
    {
        auto tGradCriterionWrtConfig = mCriteria[aName]->gradientConfig(aControl, aCurPrimalVars);

        auto tCurrentPredictorAdjoint = aDualVars.vector("current predictor adjoint");
        auto tGradResPredWrtConfig = mPredictorResidual.gradientConfig(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtConfig, tCurrentPredictorAdjoint, tGradCriterionWrtConfig);

        auto tCurrentPressureAdjoint = aDualVars.vector("current pressure adjoint");
        auto tGradResPressWrtConfig = mPressureResidual.gradientConfig(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtConfig, tCurrentPressureAdjoint, tGradCriterionWrtConfig);

        auto tCurrentTemperatureAdjoint = aDualVars.vector("current temperature adjoint");
        auto tGradResTempWrtConfig = mTemperatureResidual->gradientConfig(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtConfig, tCurrentTemperatureAdjoint, tGradCriterionWrtConfig);

        auto tCurrentVelocityAdjoint = aDualVars.vector("current velocity adjoint");
        auto tGradResVelWrtConfig = mVelocityResidual.gradientConfig(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtConfig, tCurrentVelocityAdjoint, tGradCriterionWrtConfig);

        Plato::blas1::axpy(1.0, tGradCriterionWrtConfig, aTotalDerivative);
    }
};

}
// namespace Hyperbolic

}
//namespace Plato

namespace ComputationalFluidDynamicsTests
{

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, FindNodeIdsOnFaceSet)
{
    auto tSideSetName = "x-";
    constexpr auto tSpaceDim = 2;
    constexpr auto tNumPointsA = 1;
    Plato::ScalarMultiVector tPointsA("points",tNumPointsA,tSpaceDim);
    auto tHostPoints = Kokkos::create_mirror(tPointsA);
    tHostPoints(0,0) = 0.0; tHostPoints(0,1) = 0.0;
    Kokkos::deep_copy(tPointsA, tHostPoints);
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());

    // test one
    auto tNodeIds = Plato::find_node_ids_on_face_set<tNumPointsA,tSpaceDim>(*tMesh, tMeshSets, tSideSetName, tPointsA);
    TEST_EQUALITY(1,tNodeIds.size());
    auto tHostNodeIds = Kokkos::create_mirror(tNodeIds);
    Kokkos::deep_copy(tHostNodeIds, tNodeIds);
    std::vector<Plato::OrdinalType> tGold = {0};
    for(auto& tGoldId : tGold)
    {
        auto tIndex = &tGoldId - &tGold[0];
        TEST_EQUALITY(tGoldId, tHostNodeIds(tIndex));
    }

    // test two
    constexpr auto tNumPointsB = 2;
    Plato::ScalarMultiVector tPointsB("points",tNumPointsB,tSpaceDim);
    tHostPoints = Kokkos::create_mirror(tPointsB);
    tHostPoints(0,0) = 0.0; tHostPoints(0,1) = 0.0;
    tHostPoints(1,0) = 0.0; tHostPoints(1,1) = 1.0;
    Kokkos::deep_copy(tPointsB, tHostPoints);
    tNodeIds = Plato::find_node_ids_on_face_set<tNumPointsB,tSpaceDim>(*tMesh, tMeshSets, tSideSetName, tPointsB);
    TEST_EQUALITY(2,tNodeIds.size());
    tHostNodeIds = Kokkos::create_mirror(tNodeIds);
    Kokkos::deep_copy(tHostNodeIds, tNodeIds);
    tGold = {0,1};
    for(auto& tGoldId : tGold)
    {
        auto tIndex = &tGoldId - &tGold[0];
        TEST_EQUALITY(tGoldId, tHostNodeIds(tIndex));
    }

    // test three 
    constexpr auto tNumPointsC = 2;
    Plato::ScalarMultiVector tPointsC("points",tNumPointsC,tSpaceDim);
    tHostPoints = Kokkos::create_mirror(tPointsC);
    tHostPoints(0,0) = 0.0; tHostPoints(0,1) = 0.0;
    tHostPoints(1,0) = 0.0; tHostPoints(1,1) = 2.0;
    Kokkos::deep_copy(tPointsC, tHostPoints);
    tNodeIds = Plato::find_node_ids_on_face_set<tNumPointsC,tSpaceDim>(*tMesh, tMeshSets, tSideSetName, tPointsC);
    TEST_EQUALITY(1,tNodeIds.size());
    tHostNodeIds = Kokkos::create_mirror(tNodeIds);
    Kokkos::deep_copy(tHostNodeIds, tNodeIds);
    tGold = {0};
    for(auto& tGoldId : tGold)
    {
        auto tIndex = &tGoldId - &tGold[0];
        TEST_EQUALITY(tGoldId, tHostNodeIds(tIndex));
    } 

    // test four
    tNodeIds = Plato::find_node_ids_on_face_set<tNumPointsC,tSpaceDim>(*tMesh, tMeshSets, "dog", tPointsC);
    TEST_EQUALITY(0,tNodeIds.size());
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoProblem_SteadyState)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Prandtl Number'   type='double'  value='1.7'/>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='400'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Steel'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Material Models'>"
            "    <ParameterList name='Steel'>"
            "      <ParameterList name='Thermal Properties'>"
            "        <Parameter  name='Fluid Thermal Conductivity'  type='double'  value='1'/>"
            "        <Parameter  name='Reference Temperature'       type='double'  value='10.0'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Momentum Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Pressure Essential Boundary Conditions'>"
            "    <ParameterList  name='Zero Pressure'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='pressure'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // add pressure essential boundary condition to node set list
    Plato::ScalarMultiVector tPoints("points",1,2);
    auto tHostPoints = Kokkos::create_mirror(tPoints);
    tHostPoints(0,0) = 0.0; tHostPoints(0,1) = 0.0;
    auto tNodeIds = Plato::find_node_ids_on_face_set<1,2>(*tMesh, tMeshSets, "x-", tPoints);
    auto tLocalNodeIds = Plato::omega_h::copy(tNodeIds);
    tMeshSets[Omega_h::NODE_SET].insert( std::pair<std::string,Omega_h::LOs>("pressure",tLocalNodeIds) );

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and run incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::SteadyState<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    const auto tNumVerts = tMesh->nverts();
    auto tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);
    tProblem.output("cfd_test_problem");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ApplyWeight_Brinkman)
{
    constexpr auto tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MomentumPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    Plato::Fluids::ApplyWeightData tInputs;
    tInputs.set("convexity", 0.5);
    Plato::Fluids::ApplyWeight<Plato::Fluids::WeightBrinkman<PhysicsT,EvaluationT>,EvaluationT> tApplyWeight(tInputs);

    auto tNumCells = 2;
    auto tPhysicalParam = 20;
    auto tNumNodesPerCell = 3;
    Plato::ScalarVector tOutput("output", tNumCells);
    Plato::ScalarMultiVector tControlWS("controls", tNumCells, tNumNodesPerCell);
    Plato::blas2::fill(0.5, tControlWS);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tOutput(aCellOrdinal) = tApplyWeight(aCellOrdinal, tPhysicalParam, tControlWS);
    }, "unit test apply weight function - brinkman");

    // test results
    auto tTol = 1e-4;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    std::vector<Plato::Scalar> tGold = {8.0,8.0};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        TEST_FLOATING_EQUALITY(tGold[tCell], tHostOutput(tCell), tTol);
    }
    //Plato::print(tOutput, "output");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ApplyWeight_MSIMP)
{
    constexpr auto tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MomentumPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    Plato::Fluids::ApplyWeightData tInputs;
    tInputs.set("penalty", 3.0);
    tInputs.set("minimum ersatz", 1e-9);
    Plato::Fluids::ApplyWeight<Plato::Fluids::WeightMSIMP<PhysicsT,EvaluationT>,EvaluationT> tApplyWeight(tInputs);

    auto tNumCells = 2;
    auto tPhysicalParam = 20;
    auto tNumNodesPerCell = 3;
    Plato::ScalarVector tOutput("output", tNumCells);
    Plato::ScalarMultiVector tControlWS("controls", tNumCells, tNumNodesPerCell);
    Plato::blas2::fill(0.5, tControlWS);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tOutput(aCellOrdinal) = tApplyWeight(aCellOrdinal, tPhysicalParam, tControlWS);
    }, "unit test apply weight function - msimp");

    // test results
    auto tTol = 1e-4;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    std::vector<Plato::Scalar> tGold = {2.5,2.5};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        TEST_FLOATING_EQUALITY(tGold[tCell], tHostOutput(tCell), tTol);
    }
    //Plato::print(tOutput, "output");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ApplyWeight_SIMP)
{
    constexpr auto tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MomentumPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    Plato::Fluids::ApplyWeightData tInputs;
    tInputs.set("penalty", 3.0);
    Plato::Fluids::ApplyWeight<Plato::Fluids::WeightSIMP<PhysicsT,EvaluationT>,EvaluationT> tApplyWeight(tInputs);

    auto tNumCells = 2;
    auto tPhysicalParam = 20;
    auto tNumNodesPerCell = 3;
    Plato::ScalarVector tOutput("output", tNumCells);
    Plato::ScalarMultiVector tControlWS("controls", tNumCells, tNumNodesPerCell);
    Plato::blas2::fill(0.5, tControlWS);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tOutput(aCellOrdinal) = tApplyWeight(aCellOrdinal, tPhysicalParam, tControlWS);
    }, "unit test apply weight function - simp");

    // test results
    auto tTol = 1e-4;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    std::vector<Plato::Scalar> tGold = {2.5,2.5};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        TEST_FLOATING_EQUALITY(tGold[tCell], tHostOutput(tCell), tTol);
    }
    //Plato::print(tOutput, "output");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ApplyWeight_RAMP)
{
    constexpr auto tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MomentumPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    Plato::Fluids::ApplyWeightData tInputs;
    tInputs.set("convexity", 0.5);
    Plato::Fluids::ApplyWeight<Plato::Fluids::WeightRAMP<PhysicsT,EvaluationT>,EvaluationT> tApplyWeight(tInputs);

    auto tNumCells = 2;
    auto tPhysicalParam = 20;
    auto tNumNodesPerCell = 3;
    Plato::ScalarVector tOutput("output", tNumCells);
    Plato::ScalarMultiVector tControlWS("controls", tNumCells, tNumNodesPerCell);
    Plato::blas2::fill(0.5, tControlWS);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tOutput(aCellOrdinal) = tApplyWeight(aCellOrdinal, tPhysicalParam, tControlWS);
    }, "unit test apply weight function - ramp");

    // test results
    auto tTol = 1e-4;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    std::vector<Plato::Scalar> tGold = {0.22,0.22};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        TEST_FLOATING_EQUALITY(tGold[tCell], tHostOutput(tCell), tTol);
    }
    //Plato::print(tOutput, "output");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ApplyWeight_NoWeight)
{
    constexpr auto tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MomentumPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    Plato::Fluids::ApplyWeightData tInputs;
    Plato::Fluids::ApplyWeight<Plato::Fluids::NoWeight<EvaluationT>,EvaluationT> tApplyWeight(tInputs);

    auto tReNum = 20;
    auto tNumCells = 2;
    auto tNumNodesPerCell = 3;
    Plato::ScalarVector tOutput("output", tNumCells);
    Plato::ScalarMultiVector tControlWS("controls", tNumCells, tNumNodesPerCell);
    Plato::blas2::fill(0.1, tControlWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tOutput(aCellOrdinal) = tApplyWeight(aCellOrdinal, tReNum, tControlWS);
    }, "unit test apply weight function - no weight");

    // test results
    auto tTol = 1e-4;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    std::vector<Plato::Scalar> tGold = {20.0,20.0};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        TEST_FLOATING_EQUALITY(tGold[tCell], tHostOutput(tCell), tTol);
    }
    //Plato::print(tOutput, "output");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateResidualEuclideanNorm)
{
    Plato::Variables tVariables;

    // set time step
    constexpr auto tNumNodes = 4;
    Plato::ScalarVector tTimeStep("time step", tNumNodes);
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep);
    tHostTimeStep(0) = 0.1;
    tHostTimeStep(1) = 0.2;
    tHostTimeStep(2) = 0.3;
    tHostTimeStep(3) = 0.4;
    Kokkos::deep_copy(tTimeStep, tHostTimeStep);
    tVariables.vector("time steps", tTimeStep);

    // set element characteristic size
    Plato::ScalarVector tCurPressure("current pressure", tNumNodes);
    auto tHostCurPressure = Kokkos::create_mirror(tCurPressure);
    tHostCurPressure(0) = 1;
    tHostCurPressure(1) = 2;
    tHostCurPressure(2) = 3;
    tHostCurPressure(3) = 4;
    Kokkos::deep_copy(tCurPressure, tHostCurPressure);
    tVariables.vector("current pressure", tCurPressure);

    // set convective velocity
    Plato::ScalarVector tPrevPressure("previous pressure", tNumNodes);
    auto tHostPrevPressure = Kokkos::create_mirror(tPrevPressure);
    tHostPrevPressure(0) = 0.5;
    tHostPrevPressure(1) = 0.6;
    tHostPrevPressure(2) = 0.7;
    tHostPrevPressure(3) = 0.8;
    Kokkos::deep_copy(tPrevPressure, tHostPrevPressure);
    tVariables.vector("previous pressure", tPrevPressure);

    // set artificial compressibility
    Plato::ScalarVector tArtificialCompress("artificial compressibility", tNumNodes);
    auto tHostArtificialCompress = Kokkos::create_mirror(tArtificialCompress);
    tHostArtificialCompress(0) = 0.1;
    tHostArtificialCompress(1) = 0.2;
    tHostArtificialCompress(2) = 0.3;
    tHostArtificialCompress(3) = 0.4;
    Kokkos::deep_copy(tArtificialCompress, tHostArtificialCompress);
    tVariables.vector("artificial compressibility", tArtificialCompress);

    // call funciton
    auto tValue = Plato::cbs::calculate_residual_euclidean_norm(tVariables);

    // test result
    auto tTol = 1e-4;
    TEST_FLOATING_EQUALITY(5.3887059e2, tValue, tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateResidualInfNorm)
{
    Plato::Variables tVariables;

    // set time step
    constexpr auto tNumNodes = 4;
    Plato::ScalarVector tTimeStep("time step", tNumNodes);
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep);
    tHostTimeStep(0) = 0.1;
    tHostTimeStep(1) = 0.2;
    tHostTimeStep(2) = 0.3;
    tHostTimeStep(3) = 0.4;
    Kokkos::deep_copy(tTimeStep, tHostTimeStep);
    tVariables.vector("time steps", tTimeStep);

    // set element characteristic size
    Plato::ScalarVector tCurPressure("current pressure", tNumNodes);
    auto tHostCurPressure = Kokkos::create_mirror(tCurPressure);
    tHostCurPressure(0) = 1;
    tHostCurPressure(1) = 2;
    tHostCurPressure(2) = 3;
    tHostCurPressure(3) = 4;
    Kokkos::deep_copy(tCurPressure, tHostCurPressure);
    tVariables.vector("current pressure", tCurPressure);

    // set convective velocity
    Plato::ScalarVector tPrevPressure("previous pressure", tNumNodes);
    auto tHostPrevPressure = Kokkos::create_mirror(tPrevPressure);
    tHostPrevPressure(0) = 0.5;
    tHostPrevPressure(1) = 0.6;
    tHostPrevPressure(2) = 0.7;
    tHostPrevPressure(3) = 0.8;
    Kokkos::deep_copy(tPrevPressure, tHostPrevPressure);
    tVariables.vector("previous pressure", tPrevPressure);

    // set artificial compressibility
    Plato::ScalarVector tArtificialCompress("artificial compressibility", tNumNodes);
    auto tHostArtificialCompress = Kokkos::create_mirror(tArtificialCompress);
    tHostArtificialCompress(0) = 0.1;
    tHostArtificialCompress(1) = 0.2;
    tHostArtificialCompress(2) = 0.3;
    tHostArtificialCompress(3) = 0.4;
    Kokkos::deep_copy(tArtificialCompress, tHostArtificialCompress);
    tVariables.vector("artificial compressibility", tArtificialCompress);

    // call funciton
    auto tValue = Plato::cbs::calculate_residual_inf_norm(tVariables);

    // test result
    auto tTol = 1e-4;
    TEST_FLOATING_EQUALITY(500.0, tValue, tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculatePressureResidual)
{
    // set time step
    constexpr auto tNumNodes = 4;
    Plato::ScalarVector tTimeStep("time step", tNumNodes);
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep);
    tHostTimeStep(0) = 0.1;
    tHostTimeStep(1) = 0.2;
    tHostTimeStep(2) = 0.3;
    tHostTimeStep(3) = 0.4;
    Kokkos::deep_copy(tTimeStep, tHostTimeStep);

    // set element characteristic size
    Plato::ScalarVector tCurPressure("current pressure", tNumNodes);
    auto tHostCurPressure = Kokkos::create_mirror(tCurPressure);
    tHostCurPressure(0) = 1;
    tHostCurPressure(1) = 2;
    tHostCurPressure(2) = 3;
    tHostCurPressure(3) = 4;
    Kokkos::deep_copy(tCurPressure, tHostCurPressure);

    // set convective velocity
    Plato::ScalarVector tPrevPressure("previous pressure", tNumNodes);
    auto tHostPrevPressure = Kokkos::create_mirror(tPrevPressure);
    tHostPrevPressure(0) = 0.5;
    tHostPrevPressure(1) = 0.6;
    tHostPrevPressure(2) = 0.7;
    tHostPrevPressure(3) = 0.8;
    Kokkos::deep_copy(tPrevPressure, tHostPrevPressure);

    // set artificial compressibility
    Plato::ScalarVector tArtificialCompress("artificial compressibility", tNumNodes);
    auto tHostArtificialCompress = Kokkos::create_mirror(tArtificialCompress);
    tHostArtificialCompress(0) = 0.1;
    tHostArtificialCompress(1) = 0.2;
    tHostArtificialCompress(2) = 0.3;
    tHostArtificialCompress(3) = 0.4;
    Kokkos::deep_copy(tArtificialCompress, tHostArtificialCompress);

    // call function
    auto tResidual = Plato::cbs::calculate_pressure_residual(tTimeStep, tCurPressure, tPrevPressure, tArtificialCompress);

    // test results
    auto tTol = 1e-4;
    auto tHostResidual = Kokkos::create_mirror(tResidual);
    Kokkos::deep_copy(tHostResidual, tResidual);
    std::vector<Plato::Scalar> tGold = {500,175,85.185185185,50};
    for (Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostResidual(tNode), tTol);
    }
    //Plato::print(tResidual, "residual");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateStableTimeStep)
{
    // build mesh and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);

    // set element characteristic size
    auto tNumNodes = tMesh->nverts();
    Plato::ScalarVector tElemCharSize("char elem size", tNumNodes);
    auto tHostElemCharSize = Kokkos::create_mirror(tElemCharSize);
    tHostElemCharSize(0) = 0.5;
    tHostElemCharSize(1) = 0.5;
    tHostElemCharSize(2) = 1.0;
    tHostElemCharSize(3) = 1.0;
    Kokkos::deep_copy(tElemCharSize, tHostElemCharSize);

    // set convective velocity
    Plato::ScalarVector tVelocity("convective velocity", tNumNodes);
    auto tHostVelocity = Kokkos::create_mirror(tVelocity);
    tHostVelocity(0) = 1;
    tHostVelocity(1) = 2;
    tHostVelocity(2) = 3;
    tHostVelocity(3) = 4;
    Kokkos::deep_copy(tVelocity, tHostVelocity);

    // set artificial compressibility
    Plato::ScalarVector tArtificialCompress("artificial compressibility", tNumNodes);
    auto tHostArtificialCompress = Kokkos::create_mirror(tArtificialCompress);
    tHostArtificialCompress(0) = 0.1;
    tHostArtificialCompress(1) = 0.2;
    tHostArtificialCompress(2) = 0.3;
    tHostArtificialCompress(3) = 0.4;
    Kokkos::deep_copy(tArtificialCompress, tHostArtificialCompress);

    // call function
    auto tTimeStep = Plato::cbs::calculate_stable_time_step(tSpatialModel, tElemCharSize, tVelocity, tArtificialCompress);

    // test results
    auto tTol = 1e-4;
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep);
    Kokkos::deep_copy(tHostTimeStep, tTimeStep);
    std::vector<Plato::Scalar> tGold = {2.272727e-1,1.136364e-1,1.515152e-1,1.136364e-1};
    for (decltype(tNumNodes) tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostTimeStep(tNode), tTol);
    }
    //Plato::print(tTimeStep, "time step");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateThermalVelocityMagnitude)
{
    // build mesh and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);

    // set element characteristic size
    auto tNumNodes = tMesh->nverts();
    Plato::ScalarVector tElemCharSize("char elem size", tNumNodes);
    auto tHostElemCharSize = Kokkos::create_mirror(tElemCharSize);
    tHostElemCharSize(0) = 0.5;
    tHostElemCharSize(1) = 0.5;
    tHostElemCharSize(2) = 1.0;
    tHostElemCharSize(3) = 1.0;
    Kokkos::deep_copy(tElemCharSize, tHostElemCharSize);

    // call function
    auto tPrandtlNum = 1;
    auto tReynoldsNum = 2;
    constexpr auto tNumNodesPerCell = 3;
    auto tThermalVelocity =
        Plato::cbs::calculate_thermal_velocity_magnitude<tNumNodesPerCell>(tSpatialModel, tPrandtlNum, tReynoldsNum, tElemCharSize);

    // test value
    auto tTol = 1e-4;
    auto tHostThermalVelocity = Kokkos::create_mirror(tThermalVelocity);
    Kokkos::deep_copy(tHostThermalVelocity, tThermalVelocity);
    std::vector<Plato::Scalar> tGold = {1.0,1.0,0.5,0.5};
    for (decltype(tNumNodes) tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostThermalVelocity(tNode), tTol);
    }
    //Plato::print(tThermalVelocity, "thermal velocity");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateDiffusiveVelocityMagnitude)
{
    // build mesh and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);

    // set element characteristic size
    auto tNumNodes = tMesh->nverts();
    Plato::ScalarVector tElemCharSize("char elem size", tNumNodes);
    auto tHostElemCharSize = Kokkos::create_mirror(tElemCharSize);
    tHostElemCharSize(0) = 0.5;
    tHostElemCharSize(1) = 0.5;
    tHostElemCharSize(2) = 1.0;
    tHostElemCharSize(3) = 1.0;
    Kokkos::deep_copy(tElemCharSize, tHostElemCharSize);

    // call function
    auto tReynoldsNum = 2;
    constexpr auto tNumNodesPerCell = 3;
    auto tDiffusiveVelocity =
        Plato::cbs::calculate_diffusive_velocity_magnitude<tNumNodesPerCell>(tSpatialModel, tReynoldsNum, tElemCharSize);

    // test value
    auto tTol = 1e-4;
    auto tHostDiffusiveVelocity = Kokkos::create_mirror(tDiffusiveVelocity);
    Kokkos::deep_copy(tHostDiffusiveVelocity, tDiffusiveVelocity);
    std::vector<Plato::Scalar> tGold = {1.0,1.0,0.5,0.5};
    for (decltype(tNumNodes) tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostDiffusiveVelocity(tNode), tTol);
    }
    //Plato::print(tDiffusiveVelocity, "diffusive velocity");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateConvectiveVelocityMagnitude)
{
    // build mesh and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);

    // set velocity field
    auto tNumNodes = tMesh->nverts();
    auto tNumSpaceDims = tMesh->dim();
    Plato::ScalarVector tVelocity("velocity", tNumNodes * tNumSpaceDims);
    auto tHostVelocity = Kokkos::create_mirror(tVelocity);
    tHostVelocity(0) = 1;
    tHostVelocity(1) = 2;
    tHostVelocity(2) = 3;
    tHostVelocity(3) = 4;
    tHostVelocity(4) = 5;
    tHostVelocity(5) = 6;
    tHostVelocity(6) = 7;
    tHostVelocity(7) = 8;
    Kokkos::deep_copy(tVelocity, tHostVelocity);

    // call function
    constexpr auto tNumNodesPerCell = 3;
    auto tConvectiveVelocity =
        Plato::cbs::calculate_convective_velocity_magnitude<tNumNodesPerCell>(tSpatialModel, tVelocity);

    // test value
    auto tTol = 1e-4;
    auto tHostConvectiveVelocity = Kokkos::create_mirror(tConvectiveVelocity);
    Kokkos::deep_copy(tHostConvectiveVelocity, tConvectiveVelocity);
    std::vector<Plato::Scalar> tGold = {2.23606797749978969640,5.0,7.81024967590665439412,10.63014581273464940799};
    for (decltype(tNumNodes) tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostConvectiveVelocity(tNode), tTol);
    }
    //Plato::print(tConvectiveVelocity, "convective velocity");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateArtificialCompressibility)
{
    constexpr auto tNumNodes = 4;
    Plato::ScalarVector tConvectiveVelocity("convective velocity", tNumNodes);
    auto tHostConvectiveVelocity = Kokkos::create_mirror(tConvectiveVelocity);
    tHostConvectiveVelocity(0) = 1.0;
    tHostConvectiveVelocity(1) = 2.0;
    tHostConvectiveVelocity(2) = 3.0;
    tHostConvectiveVelocity(3) = 0.3;
    Kokkos::deep_copy(tConvectiveVelocity, tHostConvectiveVelocity);
    Plato::ScalarVector tDiffusiveVelocity("diffusive velocity", tNumNodes);
    auto tHostDiffusiveVelocity = Kokkos::create_mirror(tDiffusiveVelocity);
    tHostDiffusiveVelocity(0) = 0.9;
    tHostDiffusiveVelocity(1) = 3.0;
    tHostDiffusiveVelocity(2) = 2.0;
    tHostDiffusiveVelocity(3) = 0.2;
    Kokkos::deep_copy(tDiffusiveVelocity, tHostDiffusiveVelocity);
    Plato::ScalarVector tThermalVelocity("thermal velocity", tNumNodes);
    auto tHostThermalVelocity = Kokkos::create_mirror(tThermalVelocity);
    tHostThermalVelocity(0) = 0.9;
    tHostThermalVelocity(1) = 3.0;
    tHostThermalVelocity(2) = 4.0;
    tHostThermalVelocity(3) = 0.1;
    Kokkos::deep_copy(tThermalVelocity, tHostThermalVelocity);
    auto tCriticalValue = 0.5;

    auto tOutput =
        Plato::cbs::calculate_artificial_compressibility(tConvectiveVelocity, tDiffusiveVelocity, tThermalVelocity, tCriticalValue);

    // test value
    auto tTol = 1e-4;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    std::vector<Plato::Scalar> tGold = {1.0,3.0,4.0,0.5};
    for (Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostOutput(tNode), tTol);
    }
    //Plato::print(tOutput, "artificial compressibility");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateElementCharacteristicSizes)
{
    // build mesh and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);

    constexpr auto tNumSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    auto tElemCharSize = 
        Plato::cbs::calculate_element_characteristic_sizes<tNumSpaceDims,tNumNodesPerCell>(tSpatialModel);

    // test value
    auto tTol = 1e-4;
    auto tHostElemCharSize = Kokkos::create_mirror(tElemCharSize);
    Kokkos::deep_copy(tHostElemCharSize, tElemCharSize);
    std::vector<Plato::Scalar> tGold = {5.857864e-01,5.857864e-01,5.857864e-01,5.857864e-01};

    auto tNumNodes = tSpatialModel.Mesh.nverts();
    for (Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {   
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostElemCharSize(tNode), tTol);
    }
    //Plato::print(tElemCharSize, "element characteristic size");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, TemperatureIncrementResidual_EvaluatePrescribed)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Characteristic Length' type='double' value='1.0'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Material Models'>"
            "    <ParameterList name='Madeuptinum'>"
            "      <ParameterList name='Thermal Properties'>"
            "        <Parameter  name='Fluid Thermal Conductivity'  type='double'  value='1'/>"
            "        <Parameter  name='Reference Temperature'       type='double'  value='10.0'/>"
            "        <Parameter  name='Fluid Thermal Diffusivity'   type='double'  value='0.5'/>"
            "        <Parameter  name='Solid Thermal Diffusivity'   type='double'  value='1.0'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Energy Natural Boundary Conditions'>"
            "    <ParameterList  name='Heat Flux'>"
            "      <Parameter  name='Type'   type='string'  value='Uniform'/>"
            "      <Parameter  name='Sides'  type='string'  value='x+'/>"
            "      <Parameter  name='Value'  type='double'  value='-1.0'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::EnergyPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    tDomain.setMaterialName("Madeuptinum");
    tDomain.setElementBlockName("block_1");
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);
    tSpatialModel.append(tDomain);

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    using ConfigT = EvaluationT::ConfigScalarType;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using ControlT = EvaluationT::ControlScalarType;
    auto tControl = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::mNumControlDofsPerCell) );
    Plato::blas2::fill(1.0, tControl->mData);
    tWorkSets.set("control", tControl);

    using PrevTempT = EvaluationT::PreviousMassScalarType;
    auto tPrevTemp = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevTempT> > >
        ( Plato::ScalarMultiVectorT<PrevTempT>("previous pressure", tNumCells, PhysicsT::mNumEnergyDofsPerCell) );
    auto tHostPrevTemp = Kokkos::create_mirror(tPrevTemp->mData);
    tHostPrevTemp(0, 0) = 1; tHostPrevTemp(1, 0) = 4;
    tHostPrevTemp(0, 1) = 2; tHostPrevTemp(1, 1) = 5;
    tHostPrevTemp(0, 2) = 3; tHostPrevTemp(1, 2) = 6;
    Kokkos::deep_copy(tPrevTemp->mData, tHostPrevTemp);
    tWorkSets.set("previous temperature", tPrevTemp);

    auto tTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("time step", tNumCells, tNumNodesPerCell) );
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep->mData);
    tHostTimeStep(0, 0) = 0.01; tHostTimeStep(1, 0) = 0.04;
    tHostTimeStep(0, 1) = 0.02; tHostTimeStep(1, 1) = 0.05;
    tHostTimeStep(0, 2) = 0.03; tHostTimeStep(1, 2) = 0.06;
    Kokkos::deep_copy(tTimeStep->mData, tHostTimeStep);
    tWorkSets.set("time steps", tTimeStep);

    // evaluate temperature increment residual
    Plato::DataMap tDataMap;
    Plato::ScalarMultiVectorT<EvaluationT::ResultScalarType> tResult("result", tNumCells, PhysicsT::mNumEnergyDofsPerCell);
    Plato::Fluids::SIMP::TemperatureResidual<PhysicsT,EvaluationT> tResidual(tDomain,tDataMap,tInputs.operator*());
    tResidual.evaluatePrescribed(tSpatialModel, tWorkSets, tResult);

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.0,0.01,0.015}, {0.0,0.0,0.0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < PhysicsT::mNumEnergyDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Brinkman_VelocityPredictorResidual_EvaluateBoundary)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Darcy Number'   type='double'        value='1.0'/>"
            "      <Parameter  name='Prandtl Number' type='double'        value='1.0'/>"
            "      <Parameter  name='Grashof Number' type='Array(double)' value='{0.0,1.0}'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Momentum Natural Boundary Conditions'>"
            "    <ParameterList  name='Traction Vector Boundary Condition'>"
            "      <Parameter  name='Type'   type='string'        value='Uniform'/>"
            "      <Parameter  name='Sides'  type='string'        value='x+'/>"
            "      <Parameter  name='Values' type='Array(double)' value='{0,-1.0}'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MomentumPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    auto tSpatialDomainName = std::string("my box");
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, tSpatialDomainName);
    auto tElementBlockName = std::string("body");
    tDomain.cellOrdinals(tElementBlockName);
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);
    tSpatialModel.append(tDomain);

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    using ConfigT = EvaluationT::ConfigScalarType;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using ControlT = EvaluationT::ControlScalarType;
    auto tControl = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::mNumControlDofsPerCell) );
    Plato::blas2::fill(1.0, tControl->mData);
    tWorkSets.set("control", tControl);

    using PrevVelT = EvaluationT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 1; tHostVelocity(1, 0) = 7;
    tHostVelocity(0, 1) = 2; tHostVelocity(1, 1) = 8;
    tHostVelocity(0, 2) = 3; tHostVelocity(1, 2) = 9;
    tHostVelocity(0, 3) = 4; tHostVelocity(1, 3) = 10;
    tHostVelocity(0, 4) = 5; tHostVelocity(1, 4) = 11;
    tHostVelocity(0, 5) = 6; tHostVelocity(1, 5) = 12;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    using PrevPressT = EvaluationT::PreviousMassScalarType;
    auto tPrevPress = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevPressT> > >
        ( Plato::ScalarMultiVectorT<PrevPressT>("previous pressure", tNumCells, PhysicsT::mNumMassDofsPerCell) );
    auto tHostPrevPress = Kokkos::create_mirror(tPrevPress->mData);
    tHostPrevPress(0, 0) = 1; tHostPrevPress(1, 0) = 4;
    tHostPrevPress(0, 1) = 2; tHostPrevPress(1, 1) = 5;
    tHostPrevPress(0, 2) = 3; tHostPrevPress(1, 2) = 6;
    Kokkos::deep_copy(tPrevPress->mData, tHostPrevPress);
    tWorkSets.set("previous pressure", tPrevPress);

    auto tTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("time step", tNumCells, tNumNodesPerCell) );
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep->mData);
    tHostTimeStep(0, 0) = 0.01; tHostTimeStep(1, 0) = 0.04;
    tHostTimeStep(0, 1) = 0.02; tHostTimeStep(1, 1) = 0.05;
    tHostTimeStep(0, 2) = 0.03; tHostTimeStep(1, 2) = 0.06;
    Kokkos::deep_copy(tTimeStep->mData, tHostTimeStep);
    tWorkSets.set("time steps", tTimeStep);

    // evaluate pressure increment residual
    Plato::DataMap tDataMap;
    Plato::ScalarMultiVectorT<EvaluationT::ResultScalarType> tResult("result", tNumCells, PhysicsT::mNumMomentumDofsPerCell);
    Plato::Fluids::Brinkman::VelocityPredictorResidual<PhysicsT,EvaluationT> tResidual(tDomain,tDataMap,tInputs.operator*());
    tResidual.evaluateBoundary(tSpatialModel, tWorkSets, tResult);

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.02,0.02,0.04,0.04,0.0,0.0}, {-0.08,-0.08,-0.1,-0.1,0.0,0.0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < PhysicsT::mNumMomentumDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Brinkman_VelocityPredictorResidual_EvaluatePrescribedBoundary)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Darcy Number'   type='double'        value='1.0'/>"
            "      <Parameter  name='Prandtl Number' type='double'        value='1.0'/>"
            "      <Parameter  name='Grashof Number' type='Array(double)' value='{0.0,1.0}'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Momentum Natural Boundary Conditions'>"
            "    <ParameterList  name='Traction Vector Boundary Condition'>"
            "      <Parameter  name='Type'   type='string'        value='Uniform'/>"
            "      <Parameter  name='Sides'  type='string'        value='x+'/>"
            "      <Parameter  name='Values' type='Array(double)' value='{0,-1.0}'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MomentumPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    auto tSpatialDomainName = std::string("my box");
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, tSpatialDomainName);
    auto tElementBlockName = std::string("body");
    tDomain.cellOrdinals(tElementBlockName);
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);
    tSpatialModel.append(tDomain);

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    using ConfigT = EvaluationT::ConfigScalarType;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using ControlT = EvaluationT::ControlScalarType;
    auto tControl = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::mNumControlDofsPerCell) );
    Plato::blas2::fill(1.0, tControl->mData);
    tWorkSets.set("control", tControl);

    using PrevVelT = EvaluationT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 1; tHostVelocity(1, 0) = 7;
    tHostVelocity(0, 1) = 2; tHostVelocity(1, 1) = 8;
    tHostVelocity(0, 2) = 3; tHostVelocity(1, 2) = 9;
    tHostVelocity(0, 3) = 4; tHostVelocity(1, 3) = 10;
    tHostVelocity(0, 4) = 5; tHostVelocity(1, 4) = 11;
    tHostVelocity(0, 5) = 6; tHostVelocity(1, 5) = 12;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    using PrevPressT = EvaluationT::PreviousMassScalarType;
    auto tPrevPress = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevPressT> > >
        ( Plato::ScalarMultiVectorT<PrevPressT>("previous pressure", tNumCells, PhysicsT::mNumMassDofsPerCell) );
    auto tHostPrevPress = Kokkos::create_mirror(tPrevPress->mData);
    tHostPrevPress(0, 0) = 1; tHostPrevPress(1, 0) = 4;
    tHostPrevPress(0, 1) = 2; tHostPrevPress(1, 1) = 5;
    tHostPrevPress(0, 2) = 3; tHostPrevPress(1, 2) = 6;
    Kokkos::deep_copy(tPrevPress->mData, tHostPrevPress);
    tWorkSets.set("previous pressure", tPrevPress);

    auto tTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("time step", tNumCells, tNumNodesPerCell) );
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep->mData);
    tHostTimeStep(0, 0) = 0.01; tHostTimeStep(1, 0) = 0.04;
    tHostTimeStep(0, 1) = 0.02; tHostTimeStep(1, 1) = 0.05;
    tHostTimeStep(0, 2) = 0.03; tHostTimeStep(1, 2) = 0.06;
    Kokkos::deep_copy(tTimeStep->mData, tHostTimeStep);
    tWorkSets.set("time steps", tTimeStep);

    // evaluate pressure increment residual
    Plato::DataMap tDataMap;
    Plato::ScalarMultiVectorT<EvaluationT::ResultScalarType> tResult("result", tNumCells, PhysicsT::mNumMomentumDofsPerCell);
    Plato::Fluids::Brinkman::VelocityPredictorResidual<PhysicsT,EvaluationT> tResidual(tDomain,tDataMap,tInputs.operator*());
    tResidual.evaluatePrescribed(tSpatialModel, tWorkSets, tResult);

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.0,0.0,-0.02,0.01,-0.03,0.015}, {0.0,0.0,0.0,0.0,0.0,0.0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < PhysicsT::mNumMomentumDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PressureIncrementResidual_EvaluateBoundary)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList  name='Momentum Essential Boundary Conditions'>"
            "    <ParameterList  name='Zero Velocity X-Dir'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Zero Velocity Y-Dir'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MassPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    auto tSpatialDomainName = std::string("my box");
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, tSpatialDomainName);
    auto tElementBlockName = std::string("body");
    tDomain.cellOrdinals(tElementBlockName);
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);
    tSpatialModel.append(tDomain);

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    using ConfigT = EvaluationT::ConfigScalarType;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using PrevVelT = EvaluationT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 1; tHostVelocity(1, 0) = 11;
    tHostVelocity(0, 1) = 2; tHostVelocity(1, 1) = 12;
    tHostVelocity(0, 2) = 3; tHostVelocity(1, 2) = 13;
    tHostVelocity(0, 3) = 4; tHostVelocity(1, 3) = 14;
    tHostVelocity(0, 4) = 5; tHostVelocity(1, 4) = 15;
    tHostVelocity(0, 5) = 6; tHostVelocity(1, 5) = 16;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    auto tPrescribedVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("prescribed velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostPrescribedVel = Kokkos::create_mirror(tPrescribedVel->mData);
    tHostPrescribedVel(0, 0) = 0.1; tHostPrescribedVel(1, 0) = 0.7;
    tHostPrescribedVel(0, 1) = 0.2; tHostPrescribedVel(1, 1) = 0.8;
    tHostPrescribedVel(0, 2) = 0.3; tHostPrescribedVel(1, 2) = 0.9;
    tHostPrescribedVel(0, 3) = 0.4; tHostPrescribedVel(1, 3) = 1.0;
    tHostPrescribedVel(0, 4) = 0.5; tHostPrescribedVel(1, 4) = 1.1;
    tHostPrescribedVel(0, 5) = 0.6; tHostPrescribedVel(1, 5) = 1.2;
    Kokkos::deep_copy(tPrescribedVel->mData, tHostPrescribedVel);
    tWorkSets.set("prescribed velocity", tPrescribedVel);

    auto tTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("time step", tNumCells, tNumNodesPerCell) );
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep->mData);
    tHostTimeStep(0, 0) = 0.01; tHostTimeStep(1, 0) = 0.04;
    tHostTimeStep(0, 1) = 0.02; tHostTimeStep(1, 1) = 0.05;
    tHostTimeStep(0, 2) = 0.03; tHostTimeStep(1, 2) = 0.06;
    Kokkos::deep_copy(tTimeStep->mData, tHostTimeStep);
    tWorkSets.set("time steps", tTimeStep);

    // evaluate pressure increment residual
    Plato::DataMap tDataMap;
    Plato::ScalarMultiVectorT<EvaluationT::ResultScalarType> tResult("result", tNumCells, PhysicsT::mNumMassDofsPerCell);
    Plato::Fluids::PressureResidual<PhysicsT,EvaluationT> tResidual(tDomain,tDataMap,tInputs.operator*());
    tResidual.evaluateBoundary(tSpatialModel, tWorkSets, tResult);

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.0,0.0,0.0},{0.0,0.15125,0.1815}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PressureResidual)
{
    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MassPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    auto tSpatialDomainName = std::string("my box");
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, tSpatialDomainName);
    auto tElementBlockName = std::string("body");
    tDomain.cellOrdinals(tElementBlockName);

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    using ConfigT = EvaluationT::ConfigScalarType;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using PrevVelT = EvaluationT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 1; tHostVelocity(1, 0) = 7;
    tHostVelocity(0, 1) = 2; tHostVelocity(1, 1) = 8;
    tHostVelocity(0, 2) = 3; tHostVelocity(1, 2) = 9;
    tHostVelocity(0, 3) = 4; tHostVelocity(1, 3) = 10;
    tHostVelocity(0, 4) = 5; tHostVelocity(1, 4) = 11;
    tHostVelocity(0, 5) = 6; tHostVelocity(1, 5) = 12;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    using PredictorT = EvaluationT::MomentumPredictorScalarType;
    auto tPredictor = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PredictorT> > >
        ( Plato::ScalarMultiVectorT<PredictorT>("predictor", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostPredictor = Kokkos::create_mirror(tPredictor->mData);
    tHostPredictor(0, 0) = 0.1; tHostPredictor(1, 0) = 0.7;
    tHostPredictor(0, 1) = 0.2; tHostPredictor(1, 1) = 0.8;
    tHostPredictor(0, 2) = 0.3; tHostPredictor(1, 2) = 0.9;
    tHostPredictor(0, 3) = 0.4; tHostPredictor(1, 3) = 1.0;
    tHostPredictor(0, 4) = 0.5; tHostPredictor(1, 4) = 1.1;
    tHostPredictor(0, 5) = 0.6; tHostPredictor(1, 5) = 1.2;
    Kokkos::deep_copy(tPredictor->mData, tHostPredictor);
    tWorkSets.set("current predictor", tPredictor);

    using PrevPressT = EvaluationT::PreviousMassScalarType;
    auto tPrevPress = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevPressT> > >
        ( Plato::ScalarMultiVectorT<PrevPressT>("previous pressure", tNumCells, PhysicsT::mNumMassDofsPerCell) );
    auto tHostPrevPress = Kokkos::create_mirror(tPrevPress->mData);
    tHostPrevPress(0, 0) = 1; tHostPrevPress(1, 0) = 4;
    tHostPrevPress(0, 1) = 2; tHostPrevPress(1, 1) = 5;
    tHostPrevPress(0, 2) = 3; tHostPrevPress(1, 2) = 6;
    Kokkos::deep_copy(tPrevPress->mData, tHostPrevPress);
    tWorkSets.set("previous pressure", tPrevPress);

    using CurPressT = EvaluationT::CurrentMassScalarType;
    auto tCurPress = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurPressT> > >
        ( Plato::ScalarMultiVectorT<CurPressT>("current pressure", tNumCells, PhysicsT::mNumMassDofsPerCell) );
    auto tHostCurPress = Kokkos::create_mirror(tCurPress->mData);
    tHostCurPress(0, 0) = 7; tHostCurPress(1, 0) = 10;
    tHostCurPress(0, 1) = 8; tHostCurPress(1, 1) = 11;
    tHostCurPress(0, 2) = 9; tHostCurPress(1, 2) = 12;
    Kokkos::deep_copy(tCurPress->mData, tHostCurPress);
    tWorkSets.set("current pressure", tCurPress);

    TEST_EQUALITY(3,tNumNodesPerCell);
    auto tTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("time step", tNumCells, tNumNodesPerCell) );
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep->mData);
    tHostTimeStep(0, 0) = 0.01; tHostTimeStep(1, 0) = 0.04;
    tHostTimeStep(0, 1) = 0.02; tHostTimeStep(1, 1) = 0.05;
    tHostTimeStep(0, 2) = 0.03; tHostTimeStep(1, 2) = 0.06;
    Kokkos::deep_copy(tTimeStep->mData, tHostTimeStep);
    tWorkSets.set("time steps", tTimeStep);

    auto tAC = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("artificial compressibility", tNumCells, tNumNodesPerCell) );
    auto tHostAC = Kokkos::create_mirror(tAC->mData);
    tHostAC(0, 0) = 0.1; tHostAC(1, 0) = 0.4;
    tHostAC(0, 1) = 0.2; tHostAC(1, 1) = 0.5;
    tHostAC(0, 2) = 0.3; tHostAC(1, 2) = 0.6;
    Kokkos::deep_copy(tAC->mData, tHostAC);
    tWorkSets.set("artificial compressibility", tAC);

    // evaluate pressure increment residual
    Plato::DataMap tDataMap;
    Plato::ScalarMultiVectorT<EvaluationT::ResultScalarType> tResult("result", tNumCells, PhysicsT::mNumMassDofsPerCell);
    Plato::Fluids::PressureResidual<PhysicsT,EvaluationT> tResidual(tDomain,tDataMap);
    tResidual.evaluate(tWorkSets, tResult);

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{10.008225,5.0055,3.30055833333333},{2.4006,1.98625,1.832566666666667}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PressureIncrementResidual_ThetaTwo_Set)
{
    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MassPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    auto tSpatialDomainName = std::string("my box");
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, tSpatialDomainName);
    auto tElementBlockName = std::string("body");
    tDomain.cellOrdinals(tElementBlockName);

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    using ConfigT = EvaluationT::ConfigScalarType;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using PrevVelT = EvaluationT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 1; tHostVelocity(1, 0) = 7;
    tHostVelocity(0, 1) = 2; tHostVelocity(1, 1) = 8;
    tHostVelocity(0, 2) = 3; tHostVelocity(1, 2) = 9;
    tHostVelocity(0, 3) = 4; tHostVelocity(1, 3) = 10;
    tHostVelocity(0, 4) = 5; tHostVelocity(1, 4) = 11;
    tHostVelocity(0, 5) = 6; tHostVelocity(1, 5) = 12;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    using PredictorT = EvaluationT::MomentumPredictorScalarType;
    auto tPredictor = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PredictorT> > >
        ( Plato::ScalarMultiVectorT<PredictorT>("predictor", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostPredictor = Kokkos::create_mirror(tPredictor->mData);
    tHostPredictor(0, 0) = 0.1; tHostPredictor(1, 0) = 0.7;
    tHostPredictor(0, 1) = 0.2; tHostPredictor(1, 1) = 0.8;
    tHostPredictor(0, 2) = 0.3; tHostPredictor(1, 2) = 0.9;
    tHostPredictor(0, 3) = 0.4; tHostPredictor(1, 3) = 1.0;
    tHostPredictor(0, 4) = 0.5; tHostPredictor(1, 4) = 1.1;
    tHostPredictor(0, 5) = 0.6; tHostPredictor(1, 5) = 1.2;
    Kokkos::deep_copy(tPredictor->mData, tHostPredictor);
    tWorkSets.set("current predictor", tPredictor);

    using PrevPressT = EvaluationT::PreviousMassScalarType;
    auto tPrevPress = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevPressT> > >
        ( Plato::ScalarMultiVectorT<PrevPressT>("previous pressure", tNumCells, PhysicsT::mNumMassDofsPerCell) );
    auto tHostPrevPress = Kokkos::create_mirror(tPrevPress->mData);
    tHostPrevPress(0, 0) = 1; tHostPrevPress(1, 0) = 4;
    tHostPrevPress(0, 1) = 2; tHostPrevPress(1, 1) = 5;
    tHostPrevPress(0, 2) = 3; tHostPrevPress(1, 2) = 6;
    Kokkos::deep_copy(tPrevPress->mData, tHostPrevPress);
    tWorkSets.set("previous pressure", tPrevPress);

    using CurPressT = EvaluationT::CurrentMassScalarType;
    auto tCurPress = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurPressT> > >
        ( Plato::ScalarMultiVectorT<CurPressT>("current pressure", tNumCells, PhysicsT::mNumMassDofsPerCell) );
    auto tHostCurPress = Kokkos::create_mirror(tCurPress->mData);
    tHostCurPress(0, 0) = 1; tHostCurPress(1, 0) = 3;
    tHostCurPress(0, 1) = 8; tHostCurPress(1, 1) = 11;
    tHostCurPress(0, 2) = 2; tHostCurPress(1, 2) = 4;
    Kokkos::deep_copy(tCurPress->mData, tHostCurPress);
    tWorkSets.set("current pressure", tCurPress);

    TEST_EQUALITY(3,tNumNodesPerCell);
    auto tTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("time step", tNumCells, tNumNodesPerCell) );
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep->mData);
    tHostTimeStep(0, 0) = 0.01; tHostTimeStep(1, 0) = 0.04;
    tHostTimeStep(0, 1) = 0.02; tHostTimeStep(1, 1) = 0.05;
    tHostTimeStep(0, 2) = 0.03; tHostTimeStep(1, 2) = 0.06;
    Kokkos::deep_copy(tTimeStep->mData, tHostTimeStep);
    tWorkSets.set("time steps", tTimeStep);

    auto tAC = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("artificial compressibility", tNumCells, tNumNodesPerCell) );
    auto tHostAC = Kokkos::create_mirror(tAC->mData);
    tHostAC(0, 0) = 0.1; tHostAC(1, 0) = 0.4;
    tHostAC(0, 1) = 0.2; tHostAC(1, 1) = 0.5;
    tHostAC(0, 2) = 0.3; tHostAC(1, 2) = 0.6;
    Kokkos::deep_copy(tAC->mData, tHostAC);
    tWorkSets.set("artificial compressibility", tAC);

    // evaluate pressure increment residual
    Plato::DataMap tDataMap;
    Plato::ScalarMultiVectorT<EvaluationT::ResultScalarType> tResult("result", tNumCells, PhysicsT::mNumMassDofsPerCell);
    Plato::Fluids::PressureResidual<PhysicsT,EvaluationT> tResidual(tDomain,tDataMap);
    tResidual.setThetaOne(0.5);
    tResidual.setThetaTwo(1.0);
    tResidual.evaluate(tWorkSets, tResult);

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{2.7858527,1.3956888,0.8915759},{0.31446667,0.3289583,0.43647778}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateInertialForces)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tBasisFunctions("basis functions", tNumNodesPerCell);
    Plato::blas1::fill(0.33333333333333333333333, tBasisFunctions);
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);
    Plato::ScalarVector tCurPress("current pressure", tNumCells);
    auto tHostCurPress = Kokkos::create_mirror(tCurPress);
    tHostCurPress(0) = 7; tHostCurPress(1) = 8;
    Kokkos::deep_copy(tCurPress, tHostCurPress);
    Plato::ScalarVector tPrevPress("previous pressure", tNumCells);
    auto tHostPrevPress = Kokkos::create_mirror(tPrevPress);
    tHostPrevPress(0) = 1; tHostPrevPress(1) = 2;
    Kokkos::deep_copy(tPrevPress, tHostPrevPress);
    Plato::ScalarMultiVector tArtificialCompress("artificial compressibility", tNumCells, tNumNodesPerCell);
    auto tHostArtificialCompress = Kokkos::create_mirror(tArtificialCompress);
    tHostArtificialCompress(0,0) = 0.1; tHostArtificialCompress(0,1) = 0.2; tHostArtificialCompress(0,2) = 0.3;
    tHostArtificialCompress(1,0) = 0.4; tHostArtificialCompress(1,1) = 0.5; tHostArtificialCompress(1,2) = 0.6;
    Kokkos::deep_copy(tArtificialCompress, tHostArtificialCompress);
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_inertial_forces<tNumNodesPerCell>
            (aCellOrdinal, tBasisFunctions, tCellVolume, tCurPress, tPrevPress, tArtificialCompress, tResult);
    }, "unit test integrate_inertial_forces");

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold = {{10.0,5.0,3.333333},{2.5,2.0,1.666667}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, DeltaPressureGradientDivergence)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);
    Plato::ScalarMultiVector tTimeStep("time step", tNumCells, tNumNodesPerCell);
    Plato::blas2::fill(0.1, tTimeStep);
    Plato::ScalarMultiVector tPrevPressGrad("previous pressure gradient", tNumCells, tSpaceDims);
    auto tHostPrevPressGrad = Kokkos::create_mirror(tPrevPressGrad);
    tHostPrevPressGrad(0,0) = 1; tHostPrevPressGrad(0,1) = 2;
    tHostPrevPressGrad(1,0) = 3; tHostPrevPressGrad(1,1) = 4;
    Kokkos::deep_copy(tPrevPressGrad, tHostPrevPressGrad);
    Plato::ScalarMultiVector tCurPressGrad("current pressure gradient", tNumCells, tSpaceDims);
    auto tHostCurPressGrad = Kokkos::create_mirror(tCurPressGrad);
    tHostCurPressGrad(0,0) = 5; tHostCurPressGrad(0,1) = 6;
    tHostCurPressGrad(1,0) = 7; tHostCurPressGrad(1,1) = 8;
    Kokkos::deep_copy(tCurPressGrad, tHostCurPressGrad);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);

    // call device function
    auto tMultiplier = 1.0;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::delta_pressure_gradient_divergence<tNumNodesPerCell,tSpaceDims>
            (aCellOrdinal, tMultiplier, tTimeStep, tGradient, tCellVolume, tCurPressGrad, tPrevPressGrad, tResult);
    }, "unit test delta_pressure_gradient_divergence");

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold = {{-0.2,0.0,0.2},{0.2,0.0,-0.2}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateScalarFieldGradient)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarMultiVector tResult("result", tNumCells, tSpaceDims);
    Plato::ScalarArray3D tGradient("gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarMultiVector tPressure("pressure", tNumCells, tNumNodesPerCell);
    auto tHostPressure = Kokkos::create_mirror(tPressure);
    tHostPressure(0,0) = 1; tHostPressure(0,1) = 2; tHostPressure(0,2) = 3;
    tHostPressure(1,0) = 4; tHostPressure(1,1) = 5; tHostPressure(1,2) = 6;
    Kokkos::deep_copy(tPressure, tHostPressure);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::calculate_scalar_field_gradient<tNumNodesPerCell,tSpaceDims>(aCellOrdinal, tGradient, tPressure, tResult);
    }, "unit test calculate_scalar_field_gradient");

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold = {{1.0,1.0},{-1.0,-1.0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tSpaceDims; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateVectorFieldGradient)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr Plato::OrdinalType tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarMultiVector tResult("result", tNumCells, tSpaceDims);
    Plato::ScalarArray3D tGradient("gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarMultiVector tVelocity("pressure", tNumCells, tNumDofsPerCell);
    auto tHostVelocity = Kokkos::create_mirror(tVelocity);
    tHostVelocity(0,0) = 1; tHostVelocity(0,1) = 2; tHostVelocity(0,2) = 3; tHostVelocity(0,3) = 4 ; tHostVelocity(0,4) = 5 ; tHostVelocity(0,5) = 6;
    tHostVelocity(1,0) = 7; tHostVelocity(1,1) = 8; tHostVelocity(1,2) = 9; tHostVelocity(1,3) = 10; tHostVelocity(1,4) = 11; tHostVelocity(1,5) = 12;
    Kokkos::deep_copy(tVelocity, tHostVelocity);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::calculate_vector_field_gradient<tNumNodesPerCell,tSpaceDims>(aCellOrdinal, tGradient, tVelocity, tResult);
    }, "unit test calculate_scalar_field_gradient");

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold = {{2.0,2.0},{-2.0,-2.0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tSpaceDims; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PreviousPressureGradientDivergence)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);
    Plato::ScalarMultiVector tTimeStep("time step", tNumCells, tNumNodesPerCell);
    Plato::blas2::fill(0.1, tTimeStep);
    Plato::ScalarMultiVector tPressureGrad("pressure gradient", tNumCells, tSpaceDims);
    auto tHostPressureGrad = Kokkos::create_mirror(tPressureGrad);
    tHostPressureGrad(0,0) = 1; tHostPressureGrad(0,1) = 2;
    tHostPressureGrad(1,0) = 3; tHostPressureGrad(1,1) = 4;
    Kokkos::deep_copy(tPressureGrad, tHostPressureGrad);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);

    // call device function
    auto tMultiplier = 1.0;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::previous_pressure_gradient_divergence<tNumNodesPerCell,tSpaceDims>
            (aCellOrdinal, tMultiplier, tTimeStep, tGradient, tCellVolume, tPressureGrad, tResult);
    }, "unit test previous_pressure_gradient_divergence");

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold = {{-0.05,-0.05,0.1},{0.15,0.05,-0.2}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateAdvectedForces)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);
    Plato::ScalarMultiVector tPrevVel("previous velocity", tNumCells, tSpaceDims);
    auto tHostPrevVel = Kokkos::create_mirror(tPrevVel);
    tHostPrevVel(0,0) = 1; tHostPrevVel(0,1) = 2;
    tHostPrevVel(1,0) = 3; tHostPrevVel(1,1) = 4;
    Kokkos::deep_copy(tPrevVel, tHostPrevVel);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_advected_forces<tNumNodesPerCell,tSpaceDims>
            (aCellOrdinal, tGradient, tCellVolume, tPrevVel, tResult);
    }, "unit test integrate_advected_forces");

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold = {{-0.5,-0.5,1.0},{1.5,0.5,-2.0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateDeltaAdvectedForces)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);
    Plato::ScalarMultiVector tPrevVel("previous velocity", tNumCells, tSpaceDims);
    auto tHostPrevVel = Kokkos::create_mirror(tPrevVel);
    tHostPrevVel(0,0) = 1; tHostPrevVel(0,1) = 2;
    tHostPrevVel(1,0) = 3; tHostPrevVel(1,1) = 4;
    Kokkos::deep_copy(tPrevVel, tHostPrevVel);
    Plato::ScalarMultiVector tPredVel("predicted velocity", tNumCells, tSpaceDims);
    auto tHostPredVel = Kokkos::create_mirror(tPredVel);
    tHostPredVel(0,0) = 11; tHostPredVel(0,1) = 12;
    tHostPredVel(1,0) = 13; tHostPredVel(1,1) = 14;
    Kokkos::deep_copy(tPredVel, tHostPredVel);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);

    // call device function
    auto tMultiplier = 1.0;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_delta_advected_forces<tNumNodesPerCell,tSpaceDims>
            (aCellOrdinal, tMultiplier, tGradient, tCellVolume, tPredVel, tPrevVel, tResult);
    }, "unit test integrate_delta_advected_forces");

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold = {{-5.0,0.0,5.0},{5.0,0.0,-5.0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MomentumSurfaceForces)
{
    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MassPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    auto tSpatialDomainName = std::string("my box");
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, tSpatialDomainName);
    auto tElementBlockName = std::string("body");
    tDomain.cellOrdinals(tElementBlockName);

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    using ConfigT = EvaluationT::ConfigScalarType;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using PrevVelT = EvaluationT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 1; tHostVelocity(1, 0) = 11;
    tHostVelocity(0, 1) = 2; tHostVelocity(1, 1) = 12;
    tHostVelocity(0, 2) = 3; tHostVelocity(1, 2) = 13;
    tHostVelocity(0, 3) = 4; tHostVelocity(1, 3) = 14;
    tHostVelocity(0, 4) = 5; tHostVelocity(1, 4) = 15;
    tHostVelocity(0, 5) = 6; tHostVelocity(1, 5) = 16;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    auto tPrescribedVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("prescribed velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostPrescribedVel = Kokkos::create_mirror(tPrescribedVel->mData);
    tHostPrescribedVel(0, 0) = 0.1; tHostPrescribedVel(1, 0) = 0.7;
    tHostPrescribedVel(0, 1) = 0.2; tHostPrescribedVel(1, 1) = 0.8;
    tHostPrescribedVel(0, 2) = 0.3; tHostPrescribedVel(1, 2) = 0.9;
    tHostPrescribedVel(0, 3) = 0.4; tHostPrescribedVel(1, 3) = 1.0;
    tHostPrescribedVel(0, 4) = 0.5; tHostPrescribedVel(1, 4) = 1.1;
    tHostPrescribedVel(0, 5) = 0.6; tHostPrescribedVel(1, 5) = 1.2;
    Kokkos::deep_copy(tPrescribedVel->mData, tHostPrescribedVel);
    tWorkSets.set("prescribed velocity", tPrescribedVel);

    // build momentum surface forces interface and call funciton
    auto tSideSetName = std::string("x-");
    Plato::Fluids::MomentumSurfaceForces<PhysicsT, EvaluationT> tMomentumBoundaryForces(tDomain, tSideSetName);
    Plato::ScalarMultiVectorT<EvaluationT::ResultScalarType> tResult("result", tNumCells, PhysicsT::mNumMassDofsPerCell);
    tMomentumBoundaryForces(tWorkSets, tResult);

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold = {{0.0,0.0,0.0},{0.0,6.05,6.05}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SIMP_TemperatureResidual)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Scenario' type='string' value='Density TO'/>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Characteristic Length'  type='double'  value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList name='Energy Conservation'>"
            "      <ParameterList name='Penalty Function'>"
            "        <Parameter  name='Heat Source Penalty Exponent'  type='double' value='3.0'/>"
            "        <Parameter  name='Thermal Diffusion Penalty Exponent'  type='double' value='3.0'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "<ParameterList name='Spatial Model'>"
            "  <ParameterList name='Domains'>"
            "    <ParameterList name='Design Volume'>"
            "      <Parameter name='Element Block' type='string' value='block_1'/>"
            "      <Parameter name='Material Model' type='string' value='Steel'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            "<ParameterList name='Material Models'>"
            "  <ParameterList name='Steel'>"
            "    <ParameterList name='Thermal Properties'>"
            "      <Parameter  name='Fluid Thermal Conductivity'  type='double'  value='1'/>"
            "      <Parameter  name='Reference Temperature'       type='double'  value='10.0'/>"
            "      <Parameter  name='Fluid Thermal Diffusivity'   type='double'  value='0.5'/>"
            "      <Parameter  name='Solid Thermal Diffusivity'   type='double'  value='1.0'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            "  <ParameterList  name='Heat Source'>"
            "    <Parameter  name='Constant'  type='double'  value='2.0'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    tDomain.setMaterialName("Steel");
    tDomain.setElementBlockName("block_1");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::EnergyPhysicsT;

    // set control variables
    auto tNumNodes = tMesh->nverts();
    Plato::ScalarVector tControls("control", tNumNodes);
    Plato::blas1::fill(1.0, tControls);

    // set state variables
    Plato::Primal tVariables;
    auto tNumVelDofs = tNumNodes * tNumSpaceDims;
    Plato::ScalarVector tPrevVel("previous velocity", tNumVelDofs);
    auto tHostPrevVel = Kokkos::create_mirror(tPrevVel);
    tHostPrevVel(0) = 1; tHostPrevVel(1) = 2; tHostPrevVel(2) = 3; tHostPrevVel(3) = 4;
    tHostPrevVel(4) = 5; tHostPrevVel(5) = 6; tHostPrevVel(6) = 7; tHostPrevVel(7) = 8;
    Kokkos::deep_copy(tPrevVel, tHostPrevVel);
    tVariables.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.0, tPrevPress);
    tVariables.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    auto tHostPrevTemp = Kokkos::create_mirror(tPrevTemp);
    tHostPrevTemp(0) = 1; tHostPrevTemp(1) = 2; tHostPrevTemp(2) = 3; tHostPrevTemp(3) = 4;
    Kokkos::deep_copy(tPrevTemp, tHostPrevTemp);
    tVariables.vector("previous temperature", tPrevTemp);

    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    tVariables.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    tVariables.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    tVariables.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    tVariables.vector("current temperature", tCurTemp);

    Plato::ScalarVector tTimeSteps("time step", tNumNodes);
    Plato::blas1::fill(0.1, tTimeSteps);
    tVariables.vector("time steps", tTimeSteps);
    Plato::ScalarVector tVelBCs("prescribed velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tVelBCs);
    tVariables.vector("prescribed velocity", tVelBCs);

    // allocate vector function
    Plato::DataMap tDataMap;
    std::string tFuncName("Temperature");
    Plato::Fluids::VectorFunction<PhysicsT> tVectorFunction(tFuncName, tModel, tDataMap, tInputs.operator*());

    // test vector function value
    auto tResidual = tVectorFunction.value(tControls, tVariables);

    auto tTol = 1e-4;
    auto tHostResidual = Kokkos::create_mirror(tResidual);
    Kokkos::deep_copy(tHostResidual, tResidual);
    std::vector<Plato::Scalar> tGold = {-0.615278,-0.0647222,-0.1075,0.000833333};
    for(auto& tValue : tGold)
    {
        auto tIndex = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue,tHostResidual(tIndex),tTol);
    }
    //Plato::print(tResidual, "residual");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateStabilizingScalarForces)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tStabForce("cell weight", tNumCells);
    auto tHostStabForce = Kokkos::create_mirror(tStabForce);
    tHostStabForce(0) = 1; tHostStabForce(1) = -1;
    Kokkos::deep_copy(tStabForce, tHostStabForce);
    Plato::ScalarVector tDivergence("cell weight", tNumCells);
    auto tHostDivergence = Kokkos::create_mirror(tDivergence);
    tHostDivergence(0) = 4; tHostDivergence(1) = -4;
    Kokkos::deep_copy(tDivergence, tHostDivergence);
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);
    Plato::ScalarVector tBasisFunctions("basis functions", tNumNodesPerCell);
    Plato::blas1::fill(0.33333333333333333333333, tBasisFunctions);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarMultiVector tPrevVelGP("previous velocities", tNumCells, tSpaceDims);
    auto tHostPrevVelGP = Kokkos::create_mirror(tPrevVelGP);
    tHostPrevVelGP(0,0) = 1; tHostPrevVelGP(0,1) = 2;
    tHostPrevVelGP(1,0) = 3; tHostPrevVelGP(1,1) = 4;
    Kokkos::deep_copy(tPrevVelGP, tHostPrevVelGP);
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_stabilizing_scalar_forces<tNumNodesPerCell,tSpaceDims>
            (aCellOrdinal, tBasisFunctions, tCellVolume, tDivergence, tGradient, tPrevVelGP, tStabForce, tResult);
    }, "unit test integrate_stabilizing_scalar_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.166666666666667,0.166666666666667,1.666666666666667},
         {-0.833333333333333,0.166666666666667,2.666666666666667}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGArray : tGold)
    {
        auto tCell = &tGArray - &tGold[0];
        for(auto& tGValue : tGArray)
        {
            auto tDof = &tGValue - &tGArray[0];
            TEST_FLOATING_EQUALITY(tGValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "result");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateInertialForces_ThermalResidual)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tVolume("result", tNumCells);
    Plato::blas1::fill(1.0, tVolume);
    Plato::ScalarVector tCurTemp("current temperature", tNumCells);
    auto tHostCurTemp = Kokkos::create_mirror(tCurTemp);
    tHostCurTemp(0,0) = 10; tHostCurTemp(0,1) = 10; tHostCurTemp(0,2) = 10;
    tHostCurTemp(1,0) = 12; tHostCurTemp(1,1) = 12; tHostCurTemp(1,2) = 12;
    Kokkos::deep_copy(tCurTemp, tHostCurTemp);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumCells);
    auto tHostPrevTemp = Kokkos::create_mirror(tPrevTemp);
    tHostPrevTemp(0,0) = 5; tHostPrevTemp(0,1) = 5; tHostPrevTemp(0,2) = 5;
    tHostPrevTemp(1,0) = 6; tHostPrevTemp(1,1) = 6; tHostPrevTemp(1,2) = 6;
    Kokkos::deep_copy(tPrevTemp, tHostPrevTemp);
    Plato::ScalarMultiVector tResult("flux", tNumCells, tNumNodesPerCell);

    // call device function
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::calculate_inertial_forces<tNumNodesPerCell>(aCellOrdinal, tBasisFunctions, tVolume, tCurTemp, tPrevTemp, tResult);
    }, "unit test calculate_inertial_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{1.666666666666667,1.666666666666667,1.666666666666667},
         {2.0,2.0,2.0}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGArray : tGold)
    {
        auto tCell = &tGArray - &tGold[0];
        for(auto& tGValue : tGArray)
        {
            auto tDof = &tGValue - &tGArray[0];
            TEST_FLOATING_EQUALITY(tGValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tFlux, "flux");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PenalizeHeatSourceConstant)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;

    constexpr auto tPenaltyExp = 3.0;
    constexpr auto tHeatSourceConst  = 4.0;
    Plato::ScalarVector tResult("result", tNumCells);
    Plato::ScalarMultiVector tControl("control", tNumCells, tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0,0) = 0.5; tHostControl(0,1) = 0.5; tHostControl(0,2) = 0.5;
    tHostControl(1,0) = 1.0; tHostControl(1,1) = 1.0; tHostControl(1,2) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tResult(aCellOrdinal) =
            Plato::Fluids::penalize_heat_source_constant<tNumNodesPerCell>(aCellOrdinal, tHeatSourceConst, tPenaltyExp, tControl);
    }, "unit test penalize_heat_source_constant");

    auto tTol = 1e-4;
    std::vector<Plato::Scalar> tGold = {3.5,0.0};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for (auto &tValue : tGold)
    {
        auto tCell = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue, tHostResult(tCell), tTol);
    }
    //Plato::print(tResult, "result");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PenalizeThermalDiffusivity)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tResult("result", tNumCells);
    Plato::ScalarMultiVector tControl("control", tNumCells, tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0,0) = 0.5; tHostControl(0,1) = 0.5; tHostControl(0,2) = 0.5;
    tHostControl(1,0) = 1.0; tHostControl(1,1) = 1.0; tHostControl(1,2) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);
    constexpr auto tFluidDiff  = 1.0;
    constexpr auto tSolidDiff  = 4.0;
    constexpr auto tPenaltyExp = 3.0;

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tResult(aCellOrdinal) =
            Plato::Fluids::penalize_thermal_diffusivity<tNumNodesPerCell>(aCellOrdinal, tFluidDiff, tSolidDiff, tPenaltyExp, tControl);
    }, "unit test penalize_thermal_diffusivity");

    auto tTol = 1e-4;
    std::vector<Plato::Scalar> tGold = {3.6250,1.0};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for (auto &tValue : tGold)
    {
        auto tCell = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue, tHostResult(tCell), tTol);
    }
    //Plato::print(tResult, "result");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateFlux)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarMultiVector tPrevTemp("previous temperature", tNumCells, tNumNodesPerCell);
    auto tHostPrevTemp = Kokkos::create_mirror(tPrevTemp);
    tHostPrevTemp(0,0) = 1; tHostPrevTemp(0,1) = 12; tHostPrevTemp(0,2) = 3;
    tHostPrevTemp(1,0) = 4; tHostPrevTemp(1,1) = 15; tHostPrevTemp(1,2) = 6;
    Kokkos::deep_copy(tPrevTemp, tHostPrevTemp);
    Plato::ScalarMultiVector tFlux("flux", tNumCells, tSpaceDims);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::calculate_flux<tNumNodesPerCell,tSpaceDims>(aCellOrdinal, tGradient, tPrevTemp, tFlux);
    }, "unit test calculate_flux");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{11.0,-9.0}, {-11.0,9.0}};
    auto tHostFlux = Kokkos::create_mirror(tFlux);
    Kokkos::deep_copy(tHostFlux, tFlux);
    for(auto& tGArray : tGold)
    {
        auto tCell = &tGArray - &tGold[0];
        for(auto& tGValue : tGArray)
        {
            auto tDof = &tGValue - &tGArray[0];
            TEST_FLOATING_EQUALITY(tGValue,tHostFlux(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tFlux, "flux");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateFluxDivergence)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);
    Plato::ScalarMultiVector tFlux("flux", tNumCells, tSpaceDims);
    auto tHostFlux = Kokkos::create_mirror(tFlux);
    tHostFlux(0,0) = 1; tHostFlux(0,1) = 2;
    tHostFlux(1,0) = 3; tHostFlux(1,1) = 4;
    Kokkos::deep_copy(tFlux, tHostFlux);

    // set functors for unit test
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    // call device function
    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        Plato::Fluids::calculate_flux_divergence<tNumNodesPerCell,tSpaceDims>(aCellOrdinal, tGradient, tCellVolume, tFlux, tResult);
    }, "unit test calculate_flux_divergence");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{-1.0,-1.0,2.0}, {3.0,1.0,-4.0}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGArray : tGold)
    {
        auto tCell = &tGArray - &tGold[0];
        for(auto& tGValue : tGArray)
        {
            auto tDof = &tGValue - &tGArray[0];
            TEST_FLOATING_EQUALITY(tGValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateScalarField)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);
    Plato::ScalarVector tSource("cell source", tNumCells);
    auto tHostSource = Kokkos::create_mirror(tSource);
    tHostSource(0) = 1; tHostSource(1) = 2;
    Kokkos::deep_copy(tSource, tHostSource);
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);

    // call device kernel
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_scalar_field<tNumNodesPerCell>(aCellOrdinal, tBasisFunctions, tCellVolume, tSource, tResult);
    }, "unit test integrate_scalar_field");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.166666666666667,0.166666666666667,0.166666666666667},
         {0.333333333333333,0.333333333333333,0.333333333333333}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGArray : tGold)
    {
        auto tCell = &tGArray - &tGold[0];
        for(auto& tGValue : tGArray)
        {
            auto tDof = &tGValue - &tGArray[0];
            TEST_FLOATING_EQUALITY(tGValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateConvectiveForces)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tPrevTemp("previous temperature", tNumCells, tNumNodesPerCell);
    auto tHostPrevTemp = Kokkos::create_mirror(tPrevTemp);
    tHostPrevTemp(0,0) = 1; tHostPrevTemp(0,1) = 2; tHostPrevTemp(0,2) = 3;
    tHostPrevTemp(1,0) = 4; tHostPrevTemp(1,1) = 5; tHostPrevTemp(1,2) = 6;
    Kokkos::deep_copy(tPrevTemp, tHostPrevTemp);
    Plato::ScalarMultiVector tPrevVelGP("previous velocities", tNumCells, tSpaceDims);
    auto tHostPrevVelGP = Kokkos::create_mirror(tPrevVelGP);
    tHostPrevVelGP(0,0) = 1; tHostPrevVelGP(0,1) = 2;
    tHostPrevVelGP(1,0) = 3; tHostPrevVelGP(1,1) = 4;
    Kokkos::deep_copy(tPrevVelGP, tHostPrevVelGP);
    Plato::ScalarVector tForces("internal force", tNumCells);

    // set functors for unit test
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        Plato::Fluids::calculate_convective_forces<tNumNodesPerCell, tSpaceDims>(aCellOrdinal, tGradient, tPrevVelGP, tPrevTemp, tForces);
    }, "unit test calculate_convective_forces");

    auto tTol = 1e-4;
    std::vector<Plato::Scalar> tGold = {3.0,-7.0};
    auto tHostForces = Kokkos::create_mirror(tForces);
    Kokkos::deep_copy(tHostForces, tForces);
    for (auto &tValue : tGold)
    {
        auto tCell = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue, tHostForces(tCell), tTol);
    }
    //Plato::print(tForces, "convective forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, VelocityCorrectorResidual)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural Convection'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Darcy Number'   type='double'        value='1.0'/>"
            "      <Parameter  name='Prandtl Number' type='double'        value='1.0'/>"
            "      <Parameter  name='Grashof Number' type='Array(double)' value='{0.0,1.0}'/>"
            "    </ParameterList>"
            "    <ParameterList name='Momentum Conservation'>"
            "      <ParameterList name='Penalty Function'>"
            "        <Parameter name='Brinkman Convexity Parameter' type='double' value='0.5'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Artificial Damping Two' type='double' value='0.2'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MomentumPhysicsT;

    // set control variables
    auto tNumNodes = tMesh->nverts();
    Plato::ScalarVector tControls("control", tNumNodes);
    Plato::blas1::fill(1.0, tControls);

    // set state variables
    Plato::Primal tVariables;
    auto tNumVelDofs = tNumNodes * tNumSpaceDims;
    Plato::ScalarVector tPrevVel("previous velocity", tNumVelDofs);
    auto tHostPrevVel = Kokkos::create_mirror(tPrevVel);
    tHostPrevVel(0) = 1; tHostPrevVel(1) = 2; tHostPrevVel(2) = 3;
    tHostPrevVel(3) = 4; tHostPrevVel(4) = 5; tHostPrevVel(5) = 6;
    Kokkos::deep_copy(tPrevVel, tHostPrevVel);
    tVariables.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.0, tPrevPress);
    tVariables.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    tVariables.vector("previous temperature", tPrevTemp);

    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    tVariables.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    auto tHostCurPred = Kokkos::create_mirror(tCurPred);
    tHostCurPred(0) = 7; tHostCurPred(1) = 8; tHostCurPred(2) = 9;
    tHostCurPred(3) = 10; tHostCurPred(4) = 11; tHostCurPred(5) = 12;
    Kokkos::deep_copy(tCurPred, tHostCurPred);
    tVariables.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    auto tHostCurPress = Kokkos::create_mirror(tCurPress);
    tHostCurPress(0) = 1; tHostCurPress(1) = 2;
    tHostCurPress(2) = 3; tHostCurPress(3) = 4;
    Kokkos::deep_copy(tCurPress, tHostCurPress);
    tVariables.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    tVariables.vector("current temperature", tCurTemp);

    Plato::ScalarVector tTimeSteps("time step", tNumNodes);
    Plato::blas1::fill(0.1, tTimeSteps);
    tVariables.vector("time steps", tTimeSteps);
    Plato::ScalarVector tVelBCs("prescribed velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tVelBCs);
    tVariables.vector("prescribed velocity", tVelBCs);

    // allocate vector function
    Plato::DataMap tDataMap;
    std::string tFuncName("Velocity");
    Plato::Fluids::VectorFunction<PhysicsT> tVectorFunction(tFuncName, tModel, tDataMap, tInputs.operator*());

    // test vector function value
    auto tResidual = tVectorFunction.value(tControls, tVariables);

    auto tTol = 1e-4;
    auto tHostResidual = Kokkos::create_mirror(tResidual);
    Kokkos::deep_copy(tHostResidual, tResidual);
    std::vector<Plato::Scalar> tGold = {-2.48717,-2.77761,-1.4955,-1.66333,-2.4845,-2.77561,-0.9885,-1.11444};
    for(auto& tValue : tGold)
    {
        auto tIndex = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue,tHostResidual(tIndex),tTol);
    }
    //Plato::print(tResidual, "residual");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculatePressureGradient)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tCurPress("current pressure", tNumCells, tNumNodesPerCell);
    auto tHostCurPress = Kokkos::create_mirror(tCurPress);
    tHostCurPress(0,0) = 1; tHostCurPress(0,1) = 2; tHostCurPress(0,2) = 3;
    tHostCurPress(1,0) = 4; tHostCurPress(1,1) = 5; tHostCurPress(1,2) = 6;
    Kokkos::deep_copy(tCurPress, tHostCurPress);
    Plato::ScalarMultiVector tPrevPress("previous pressure", tNumCells, tNumNodesPerCell);
    auto tHostPrevPress = Kokkos::create_mirror(tPrevPress);
    tHostPrevPress(0,0) = 1; tHostPrevPress(0,1) = 12; tHostPrevPress(0,2) = 3;
    tHostPrevPress(1,0) = 4; tHostPrevPress(1,1) = 15; tHostPrevPress(1,2) = 6;
    Kokkos::deep_copy(tPrevPress, tHostPrevPress);
    Plato::ScalarMultiVector tPressGrad("result", tNumCells, tSpaceDims);

    // set functors for unit test
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    // call device kernel
    auto tTheta = 0.2;
    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        Plato::Fluids::calculate_pressure_gradient<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tTheta, tGradient, tCurPress, tPrevPress, tPressGrad);
    }, "unit test calculate_pressure_gradient");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{9.0,-7.0}, {-9.0,7.0}};
    auto tHostPressGrad = Kokkos::create_mirror(tPressGrad);
    Kokkos::deep_copy(tHostPressGrad, tPressGrad);
    for(auto& tGoldVector : tGold)
    {
        auto tCell = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tDof = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostPressGrad(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tPressGrad, "pressure gradient");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateBrinkmanForces)
{
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tBrinkmanCoeff = 0.5;
    Plato::ScalarMultiVector tResult("results", tNumCells, tSpaceDims);
    Plato::ScalarMultiVector tPrevVelGP("previous velocities", tNumCells, tSpaceDims);
    auto tHostPrevVelGP = Kokkos::create_mirror(tPrevVelGP);
    tHostPrevVelGP(0,0) = 1; tHostPrevVelGP(0,1) = 2;
    tHostPrevVelGP(1,0) = 3; tHostPrevVelGP(1,1) = 4;
    Kokkos::deep_copy(tPrevVelGP, tHostPrevVelGP);

    // call device kernel
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::calculate_brinkman_forces<tSpaceDims>(aCellOrdinal, tBrinkmanCoeff, tPrevVelGP, tResult);
    }, "unit test calculate_brinkman_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{0.5,1.0},{1.5,2.0}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGoldVector : tGold)
    {
        auto tCell = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tDof = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "brinkman forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateStabilizingForces)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr auto tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarVector tDivergence("divergence", tNumCells);
    auto tHostDivergence = Kokkos::create_mirror(tDivergence);
    tHostDivergence(0) = 4; tHostDivergence(1) = -4;
    Kokkos::deep_copy(tDivergence, tHostDivergence);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tPrevVelGP("previous velocities", tNumCells, tSpaceDims);
    auto tHostPrevVelGP = Kokkos::create_mirror(tPrevVelGP);
    tHostPrevVelGP(0,0) = 1; tHostPrevVelGP(0,1) = 2;
    tHostPrevVelGP(1,0) = 3; tHostPrevVelGP(1,1) = 4;
    Kokkos::deep_copy(tPrevVelGP, tHostPrevVelGP);
    Plato::ScalarMultiVector tForce("internal force", tNumCells, tSpaceDims);
    Plato::blas2::fill(1.0,tForce);
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumDofsPerCell);

    // set functors for unit test
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    // call device kernel
    auto tCubWeight = tCubRule.getCubWeight();
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        tCellVolume(aCellOrdinal) *= tCubWeight;
        Plato::Fluids::integrate_stabilizing_vector_forces<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tBasisFunctions, tCellVolume, tDivergence, tGradient, tPrevVelGP, tForce, tResult);
    }, "unit test integrate_stabilizing_vector_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.166666666666667,0.166666666666667,0.166666666666667,0.166666666666667,1.666666666666667,1.666666666666667},
         {0.833333333333333,0.833333333333333,-0.166666666666667,-0.166666666666667,-2.666666666666667,-2.666666666666667}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGoldVector : tGold)
    {
        auto tCell = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tDof = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "stabilizing forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateInertialForces)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr auto tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumDofsPerCell);
    Plato::ScalarMultiVector tPredVelGP("predicted velocities", tNumCells, tSpaceDims);
    Plato::ScalarMultiVector tPrevVelGP("previous velocities", tNumCells, tSpaceDims);
    auto tHostPrevVelGP = Kokkos::create_mirror(tPrevVelGP);
    tHostPrevVelGP(0,0) = 1; tHostPrevVelGP(0,1) = 2;
    tHostPrevVelGP(1,0) = 3; tHostPrevVelGP(1,1) = 4;
    Kokkos::deep_copy(tPrevVelGP, tHostPrevVelGP);

    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::calculate_inertial_forces<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tBasisFunctions, tCellVolume, tPredVelGP, tPrevVelGP, tResult);
    }, "unit test calculate_inertial_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{-1.666667e-01,-3.333333e-01,-1.666667e-01,-3.333333e-01,-1.666667e-01,-3.333333e-01},
         {-5.000000e-01,-6.666667e-01,-5.000000e-01,-6.666667e-01,-5.000000e-01,-6.666667e-01}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGoldVector : tGold)
    {
        auto tCell = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tDof = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "inertial forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MultiplyTimeStep)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr auto tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumDofsPerCell);
    Plato::blas2::fill(1.0, tResult);
    Plato::ScalarMultiVector tTimeStep("time step", tNumCells, tNumNodesPerCell);
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep);
    tHostTimeStep(0,0) = 1; tHostTimeStep(0,1) = 2; tHostTimeStep(0,2) = 3;
    tHostTimeStep(1,0) = 4; tHostTimeStep(1,1) = 5; tHostTimeStep(1,2) = 6;
    Kokkos::deep_copy(tTimeStep, tHostTimeStep);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::multiply_time_step<tNumNodesPerCell, tSpaceDims>(aCellOrdinal, 0.5, tTimeStep, tResult);
    }, "unit test multiply_time_step");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.5,0.5,1.0,1.0,1.5,1.5},{2.0,2.0,2.5,2.5,3.0,3.0}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGoldVector : tGold)
    {
        auto tCell = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tDof = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "time step");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Divergence)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr auto tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarVector tResult("divergence", tNumCells);
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tPrevVelWS("previous velocity", tNumCells, tNumDofsPerCell);
    auto tHostPrevVelWS = Kokkos::create_mirror(tPrevVelWS);
    tHostPrevVelWS(0,0) = 1; tHostPrevVelWS(0,1) = 2; tHostPrevVelWS(0,2) = 3; tHostPrevVelWS(0,3) = 4 ; tHostPrevVelWS(0,4) = 5 ; tHostPrevVelWS(0,5) = 6;
    tHostPrevVelWS(1,0) = 7; tHostPrevVelWS(1,1) = 8; tHostPrevVelWS(1,2) = 9; tHostPrevVelWS(1,3) = 10; tHostPrevVelWS(1,4) = 11; tHostPrevVelWS(1,5) = 12;
    Kokkos::deep_copy(tPrevVelWS, tHostPrevVelWS);

    // set functors for unit test
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    // call device kernel
    auto tCubWeight = tCubRule.getCubWeight();
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        Plato::Fluids::divergence<tNumNodesPerCell, tSpaceDims>(aCellOrdinal, tGradient, tPrevVelWS, tResult);
    }, "unit test divergence");

    auto tTol = 1e-4;
    std::vector<Plato::Scalar> tGold = {4.0,-4.0};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tValue : tGold)
    {
        auto tCell = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue,tHostResult(tCell),tTol);
    }
    //Plato::print(tResult, "divergence");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Integrate)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr auto tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);
    Plato::ScalarMultiVector tResult("results", tNumCells, tNumDofsPerCell);
    Plato::ScalarMultiVector tInternalForces("internal forces", tNumCells, tSpaceDims);
    auto tHostInternalForces = Kokkos::create_mirror(tInternalForces);
    tHostInternalForces(0,0) = 26.0 ; tHostInternalForces(0,1) = 30.0;
    tHostInternalForces(1,0) = -74.0; tHostInternalForces(1,1) = -78.0;
    Kokkos::deep_copy(tInternalForces, tHostInternalForces);

    // call device kernel
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    auto tCubWeight = tCubRule.getCubWeight();
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_vector_field<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tBasisFunctions, tCellVolume, tInternalForces, tResult);
    }, "unit test integrate_vector_field");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{4.333333,5.0,4.333333,5.0,4.333333,5.0},{-12.33333,-13.0,-12.33333,-13.0,-12.33333,-13.0}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGoldVector : tGold)
    {
        auto tCell = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tDof = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "integrated forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateAdvectedInternalForces)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumVelDofsPerNode = tSpaceDims;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr auto tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tPrevVelGP("previous velocity at GP", tNumCells, tSpaceDims);
    Plato::ScalarMultiVector tPrevVelWS("previous velocity", tNumCells, tNumDofsPerCell);
    Plato::ScalarMultiVector tInternalForces("internal forces", tNumCells, tSpaceDims);
    auto tHostPrevVelWS = Kokkos::create_mirror(tPrevVelWS);
    tHostPrevVelWS(0,0) = 1; tHostPrevVelWS(0,1) = 2; tHostPrevVelWS(0,2) = 3; tHostPrevVelWS(0,3) = 4 ; tHostPrevVelWS(0,4) = 5 ; tHostPrevVelWS(0,5) = 6;
    tHostPrevVelWS(1,0) = 7; tHostPrevVelWS(1,1) = 8; tHostPrevVelWS(1,2) = 9; tHostPrevVelWS(1,3) = 10; tHostPrevVelWS(1,4) = 11; tHostPrevVelWS(1,5) = 12;
    Kokkos::deep_copy(tPrevVelWS, tHostPrevVelWS);

    // set functors for unit test
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    Plato::InterpolateFromNodal<tSpaceDims, tNumVelDofsPerNode, 0, tSpaceDims> tIntrplVectorField;

    // call device kernel
    auto tCubWeight = tCubRule.getCubWeight();
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        tCellVolume(aCellOrdinal) *= tCubWeight;

        tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
        Plato::Fluids::calculate_advected_internal_forces<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tGradient, tPrevVelWS, tPrevVelGP, tInternalForces);
    }, "unit test calculate_advected_internal_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{26.0,30.0},{-74.0,-78.0}};
    auto tHostInternalForces = Kokkos::create_mirror(tInternalForces);
    Kokkos::deep_copy(tHostInternalForces, tInternalForces);
    for(auto& tGoldVector : tGold)
    {
        auto tVecIndex = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tValIndex = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostInternalForces(tVecIndex,tValIndex),tTol);
        }
    }
    //Plato::print_array_2D(tInternalForces, "advected internal forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateNaturalConvectiveForces)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarVector tPrevTempGP("temperature at GP", tNumCells);
    Plato::blas1::fill(1.0, tPrevTempGP);
    Plato::ScalarMultiVector tResultGP("cell stabilized convective forces", tNumCells, tSpaceDims);
    Plato::Scalar tPenalizedPrNumTimesPrNum = 0.25;
    Plato::ScalarVector tPenalizedGrNum("Grashof Number", tSpaceDims);
    auto tHostPenalizedGrNum = Kokkos::create_mirror(tPenalizedGrNum);
    tHostPenalizedGrNum(1) = 1.0;
    Kokkos::deep_copy(tPenalizedGrNum, tHostPenalizedGrNum);

    // call device kernel
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::calculate_natural_convective_forces<tSpaceDims>
            (aCellOrdinal, tPenalizedPrNumTimesPrNum, tPenalizedGrNum, tPrevTempGP, tResultGP);
    }, "unit test calculate_natural_convective_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{0.0,0.25},{0.0,0.25}};
    auto tHostResultGP = Kokkos::create_mirror(tResultGP);
    Kokkos::deep_copy(tHostResultGP, tResultGP);
    for(auto& tGoldVector : tGold)
    {
        auto tVecIndex = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tValIndex = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResultGP(tVecIndex,tValIndex),tTol);
        }
    }
    //Plato::print_array_2D(tResultWS, "stabilized natural convective forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateViscousForces)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr auto tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarArray3D tStrainRate("cell strain rate", tNumCells, tSpaceDims, tSpaceDims);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tResultWS("cell viscous forces", tNumCells, tNumDofsPerCell);
    Plato::ScalarMultiVector tPrevVelWS("previous velocity workset", tNumCells, tNumDofsPerCell);
    auto tHostPrevVelWS = Kokkos::create_mirror(tPrevVelWS);
    tHostPrevVelWS(0,0) = 1; tHostPrevVelWS(0,1) = 2; tHostPrevVelWS(0,2) = 3; tHostPrevVelWS(0,3) = 4 ; tHostPrevVelWS(0,4) = 5 ; tHostPrevVelWS(0,5) = 6;
    tHostPrevVelWS(1,0) = 7; tHostPrevVelWS(1,1) = 8; tHostPrevVelWS(1,2) = 9; tHostPrevVelWS(1,3) = 10; tHostPrevVelWS(1,4) = 11; tHostPrevVelWS(1,5) = 12;
    Kokkos::deep_copy(tPrevVelWS, tHostPrevVelWS);
    Plato::Scalar tPenalizedPrNum = 0.5;

    // set functors for unit test
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    // call device kernel
    auto tCubWeight = tCubRule.getCubWeight();
    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        tCellVolume(aCellOrdinal) *= tCubWeight;

        Plato::Fluids::strain_rate<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tPrevVelWS, tGradient, tStrainRate);
        Plato::Fluids::integrate_viscous_forces<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tPenalizedPrNum, tCellVolume, tGradient, tStrainRate, tResultWS);
    }, "unit test integrate_viscous_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{-1.0,-1.0,0.0,0.0,1.0,1.0},{-1.0,-1.0,0.0,0.0,1.0,1.0}};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(auto& tGoldVector : tGold)
    {
        auto tVecIndex = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tValIndex = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResultWS(tVecIndex,tValIndex),tTol);
        }
    }
    //Plato::print_array_2D(tResultWS, "viscous forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS1_update)
{
    constexpr auto tNumCells = 2;
    constexpr auto tNumDofsPerCell = 6;
    Plato::ScalarMultiVector tVec1("vector one", tNumCells, tNumDofsPerCell);
    Plato::blas2::fill(1.0, tVec1);
    Plato::ScalarMultiVector tVec2("vector two", tNumCells, tNumDofsPerCell);
    Plato::blas2::fill(2.0, tVec2);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tConstant = static_cast<Plato::Scalar>(aCellOrdinal);
        Plato::blas1::update<tNumDofsPerCell>(aCellOrdinal, 2.0, tVec1, 3.0 + tConstant, tVec2);
    },"device_blas1_update");

    auto tTol = 1e-4;
    auto tHostVec2 = Kokkos::create_mirror(tVec2);
    Kokkos::deep_copy(tHostVec2, tVec2);
    std::vector<std::vector<Plato::Scalar>> tGold = { {8.0, 8.0, 8.0, 8.0, 8.0, 8.0}, {10.0, 10.0, 10.0, 10.0, 10.0, 10.0} };
    for(auto& tVector : tGold)
    {
        auto tCell = &tVector - &tGold[0];
        for(auto& tValue : tVector)
        {
            auto tDim = &tValue - &tVector[0];
            TEST_FLOATING_EQUALITY(tValue, tHostVec2(tCell, tDim), tTol);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, EntityFaceOrdinals)
{
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());

    // test: node sets
    auto tMyNodeSetOrdinals = Plato::get_entity_ordinals<Omega_h::NODE_SET>(tMeshSets, "x+");
    auto tLength = tMyNodeSetOrdinals.size();
    Plato::LocalOrdinalVector tNodeSetOrdinals("node set ordinals", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tNodeSetOrdinals(aOrdinal) = tMyNodeSetOrdinals[aOrdinal];
    }, "copy");
    auto tHostNodeSetOrdinals = Kokkos::create_mirror(tNodeSetOrdinals);
    Kokkos::deep_copy(tHostNodeSetOrdinals, tNodeSetOrdinals);
    TEST_EQUALITY(2, tHostNodeSetOrdinals(0));
    TEST_EQUALITY(3, tHostNodeSetOrdinals(1));
    //Plato::omega_h::print(tMyNodeSetOrdinals, "ordinals");

    // test: side sets
    auto tMySideSetOrdinals = Plato::get_entity_ordinals<Omega_h::SIDE_SET>(tMeshSets, "x+");
    tLength = tMySideSetOrdinals.size();
    Plato::LocalOrdinalVector tSideSetOrdinals("side set ordinals", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tSideSetOrdinals(aOrdinal) = tMySideSetOrdinals[aOrdinal];
    }, "copy");
    auto tHostSideSetOrdinals = Kokkos::create_mirror(tSideSetOrdinals);
    Kokkos::deep_copy(tHostSideSetOrdinals, tSideSetOrdinals);
    TEST_EQUALITY(4, tHostSideSetOrdinals(0));
    //Plato::omega_h::print(tMySideSetOrdinals, "ordinals");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IsEntitySetDefined)
{
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    TEST_EQUALITY(true, Plato::is_entity_set_defined<Omega_h::NODE_SET>(tMeshSets, "x+"));
    TEST_EQUALITY(false, Plato::is_entity_set_defined<Omega_h::NODE_SET>(tMeshSets, "dog"));

    TEST_EQUALITY(true, Plato::is_entity_set_defined<Omega_h::SIDE_SET>(tMeshSets, "x+"));
    TEST_EQUALITY(false, Plato::is_entity_set_defined<Omega_h::SIDE_SET>(tMeshSets, "dog"));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, NaturalConvectionBrinkman)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Scenario' type='string' value='Density TO'/>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Darcy Number'   type='double'        value='1.0'/>"
            "      <Parameter  name='Prandtl Number' type='double'        value='1.0'/>"
            "      <Parameter  name='Grashof Number' type='Array(double)' value='{0.0,1.0}'/>"
            "    </ParameterList>"
            "    <ParameterList name='Momentum Conservation'>"
            "      <ParameterList name='Penalty Function'>"
            "        <Parameter name='Brinkman Convexity Parameter' type='double' value='0.5'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Momentum Natural Boundary Conditions'>"
            "    <ParameterList  name='Traction Vector Boundary Condition'>"
            "      <Parameter  name='Type'   type='string'        value='Uniform'/>"
            "      <Parameter  name='Sides'  type='string'        value='x+'/>"
            "      <Parameter  name='Values' type='Array(double)' value='{0,-1.0,0}'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MomentumPhysicsT;

    // set control variables
    auto tNumNodes = tMesh->nverts();
    Plato::ScalarVector tControls("control", tNumNodes);
    Plato::blas1::fill(1.0, tControls);

    // set state variables
    Plato::Primal tVariables;
    auto tNumVelDofs = tNumNodes * tNumSpaceDims;
    Plato::ScalarVector tPrevVel("previous velocity", tNumVelDofs);
    auto tHostPrevVel = Kokkos::create_mirror(tPrevVel);
    tHostPrevVel(0) = 1.0; tHostPrevVel(1) = 1.1; tHostPrevVel(2) = 1.2;
    tHostPrevVel(3) = 1.3; tHostPrevVel(4) = 1.4; tHostPrevVel(5) = 1.5;
    Kokkos::deep_copy(tPrevVel, tHostPrevVel);
    tVariables.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.0, tPrevPress);
    tVariables.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    Plato::blas1::fill(1.0, tPrevTemp);
    tVariables.vector("previous temperature", tPrevTemp);

    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    tVariables.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    tVariables.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    tVariables.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    tVariables.vector("current temperature", tCurTemp);

    Plato::ScalarVector tTimeSteps("time step", tNumNodes);
    Plato::blas1::fill(0.1, tTimeSteps);
    tVariables.vector("time steps", tTimeSteps);
    Plato::ScalarVector tVelBCs("prescribed velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tVelBCs);
    tVariables.vector("prescribed velocity", tVelBCs);

    // allocate vector function
    Plato::DataMap tDataMap;
    std::string tFuncName("Velocity Predictor");
    Plato::Fluids::VectorFunction<PhysicsT> tVectorFunction(tFuncName, tModel, tDataMap, tInputs.operator*());

    // test vector function value
    auto tResidual = tVectorFunction.value(tControls, tVariables);

    auto tHostResidual = Kokkos::create_mirror(tResidual);
    Kokkos::deep_copy(tHostResidual, tResidual);
    std::vector<Plato::Scalar> tGold =
        {-0.212591, -0.190382, -0.163095, -0.161822, -0.293077, -0.0450344, -0.269574, -0.0480922};
    auto tTol = 1e-4;
    for(auto& tValue : tGold)
    {
        auto tIndex = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue,tHostResidual(tIndex),tTol);
    } 
    //Plato::print(tResidual, "residual");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, GetNumEntities)
{
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tNumEntities = Plato::get_num_entities(Omega_h::VERT, tMesh.operator*());
    TEST_EQUALITY(4, tNumEntities);
    tNumEntities = Plato::get_num_entities(Omega_h::EDGE, tMesh.operator*());
    TEST_EQUALITY(5, tNumEntities);
    tNumEntities = Plato::get_num_entities(Omega_h::FACE, tMesh.operator*());
    TEST_EQUALITY(2, tNumEntities);
    tNumEntities = Plato::get_num_entities(Omega_h::REGION, tMesh.operator*());
    TEST_EQUALITY(2, tNumEntities);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, FacesOnNonPrescribedBoundary)
{
    // build mesh, mesh sets, and spatial domain
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // test 1
    std::vector<std::string> tNames = {"x-","x+","y+","y-"};
    auto tFaceOrdinalsOnBoundaryOne = Plato::entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>(tNames, tDomain.Mesh, tDomain.MeshSets);
    TEST_EQUALITY(0, tFaceOrdinalsOnBoundaryOne.size());

    // test 2
    tNames = {"x+","y+","y-"};
    auto tFaceOrdinalsOnBoundaryTwo = Plato::entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>(tNames, tDomain.Mesh, tDomain.MeshSets);
    auto tLength = tFaceOrdinalsOnBoundaryTwo.size();
    TEST_EQUALITY(1, tLength);
    Plato::ScalarVector tValuesUseCaseTwo("use case two", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tValuesUseCaseTwo(aOrdinal) = tFaceOrdinalsOnBoundaryTwo[aOrdinal];
    },"data");
    auto tHostValuesUseCaseTwo = Kokkos::create_mirror(tValuesUseCaseTwo);
    Kokkos::deep_copy(tHostValuesUseCaseTwo, tValuesUseCaseTwo);
    TEST_EQUALITY(2, static_cast<Plato::OrdinalType>(tHostValuesUseCaseTwo(0)));

    // test 3
    tNames = {"y+","y-"};
    auto tFaceOrdinalsOnBoundaryThree = Plato::entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>(tNames, tDomain.Mesh, tDomain.MeshSets);
    tLength = tFaceOrdinalsOnBoundaryThree.size();
    TEST_EQUALITY(2, tLength);
    Plato::ScalarVector tValuesUseCaseThree("use case three", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tValuesUseCaseThree(aOrdinal) = tFaceOrdinalsOnBoundaryThree[aOrdinal];
    },"data");
    auto tHostValuesUseCaseThree = Kokkos::create_mirror(tValuesUseCaseThree);
    Kokkos::deep_copy(tHostValuesUseCaseThree, tValuesUseCaseThree);
    TEST_EQUALITY(2, static_cast<Plato::OrdinalType>(tHostValuesUseCaseThree(0)));
    TEST_EQUALITY(4, static_cast<Plato::OrdinalType>(tHostValuesUseCaseThree(1)));

    // test 4
    tNames = {"y-"};
    auto tFaceOrdinalsOnBoundaryFour = Plato::entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>(tNames, tDomain.Mesh, tDomain.MeshSets);
    tLength = tFaceOrdinalsOnBoundaryFour.size();
    TEST_EQUALITY(3, tLength);
    Plato::ScalarVector tValuesUseCaseFour("use case four", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tValuesUseCaseFour(aOrdinal) = tFaceOrdinalsOnBoundaryFour[aOrdinal];
    },"data");
    auto tHostValuesUseCaseFour = Kokkos::create_mirror(tValuesUseCaseFour);
    Kokkos::deep_copy(tHostValuesUseCaseFour, tValuesUseCaseFour);
    TEST_EQUALITY(2, static_cast<Plato::OrdinalType>(tHostValuesUseCaseFour(0)));
    TEST_EQUALITY(3, static_cast<Plato::OrdinalType>(tHostValuesUseCaseFour(1)));
    TEST_EQUALITY(4, static_cast<Plato::OrdinalType>(tHostValuesUseCaseFour(2)));

    // test 5
    tNames = {};
    auto tFaceOrdinalsOnBoundaryFive = Plato::entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>(tNames, tDomain.Mesh, tDomain.MeshSets);
    tLength = tFaceOrdinalsOnBoundaryFive.size();
    TEST_EQUALITY(4, tLength);
    Plato::ScalarVector tValuesUseCaseFive("use case five", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tValuesUseCaseFive(aOrdinal) = tFaceOrdinalsOnBoundaryFive[aOrdinal];
    },"data");
    auto tHostValuesUseCaseFive = Kokkos::create_mirror(tValuesUseCaseFive);
    Kokkos::deep_copy(tHostValuesUseCaseFive, tValuesUseCaseFive);
    TEST_EQUALITY(0, static_cast<Plato::OrdinalType>(tHostValuesUseCaseFive(0)));
    TEST_EQUALITY(2, static_cast<Plato::OrdinalType>(tHostValuesUseCaseFive(1)));
    TEST_EQUALITY(3, static_cast<Plato::OrdinalType>(tHostValuesUseCaseFive(2)));
    TEST_EQUALITY(4, static_cast<Plato::OrdinalType>(tHostValuesUseCaseFive(3)));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, DeviatoricSurfaceForces)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Prandtl Number' type='double' value='1.0'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Momentum Natural Boundary Conditions'>"
            "    <ParameterList  name='Traction Vector Boundary Condition'>"
            "      <Parameter  name='Type'   type='string'        value='Uniform'/>"
            "      <Parameter  name='Sides'  type='string'        value='x+'/>"
            "      <Parameter  name='Values' type='Array(double)' value='{0,-1.0}'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::MomentumConservation<tNumSpaceDims>;
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, mesh sets, and spatial domain
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tNumNodesPerCell = tNumSpaceDims + 1;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    using ConfigT = ResidualEvalT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using PrevVelT = ResidualEvalT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 0.12; tHostVelocity(1, 0) = 0.22;
    tHostVelocity(0, 1) = 0.41; tHostVelocity(1, 1) = 0.47;
    tHostVelocity(0, 2) = 0.25; tHostVelocity(1, 2) = 0.86;
    tHostVelocity(0, 3) = 0.15; tHostVelocity(1, 3) = 0.57;
    tHostVelocity(0, 4) = 0.12; tHostVelocity(1, 4) = 0.18;
    tHostVelocity(0, 5) = 0.43; tHostVelocity(1, 5) = 0.11;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    using ControlT = ResidualEvalT::ControlScalarType;
    auto tControl = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::mNumControlDofsPerCell) );
    Plato::blas2::fill(0.5, tControl->mData);
    tWorkSets.set("control", tControl);

    // build criterion
    Plato::ScalarMultiVectorT<ResidualEvalT::ResultScalarType>
        tResult("result", tNumCells, PhysicsT::mNumMomentumDofsPerCell);
    Plato::Fluids::DeviatoricSurfaceForces<PhysicsT, ResidualEvalT>
        tDeviatoricSurfaceForces(tDomain, tInputs.operator*());

    // test function
    tDeviatoricSurfaceForces(tWorkSets, tResult);
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{0.195,-0.28,0.195,-0.28,0,0},{0.64,-0.29,0.64,-0.29,0,0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < PhysicsT::mNumMomentumDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
            //printf("Results(Cell=%d,Dof=%d)=%f\n", tCell, tDof, tHostResult(tCell, tDof));
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, DeviatoricSurfaceForces_Zeros)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Prandtl Number' type='double' value='1.0'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Momentum Natural Boundary Conditions'>"
            "    <ParameterList  name='Traction Vector Boundary Condition - X+'>"
            "      <Parameter  name='Sides'  type='string'        value='x+'/>"
            "      <Parameter  name='Values' type='Array(double)' value='{0,-1}'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Traction Vector Boundary Condition - X-'>"
            "      <Parameter  name='Sides'  type='string'        value='x-'/>"
            "      <Parameter  name='Values' type='Array(double)' value='{0,-1}'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Traction Vector Boundary Condition - Y-'>"
            "      <Parameter  name='Sides'  type='string'        value='y-'/>"
            "      <Parameter  name='Values' type='Array(double)' value='{0,-1}'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Traction Vector Boundary Condition - Y+'>"
            "      <Parameter  name='Sides'  type='string'        value='y+'/>"
            "      <Parameter  name='Values' type='Array(double)' value='{0,-1}'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::MomentumConservation<tNumSpaceDims>;
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, mesh sets, and spatial domain
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tNumNodesPerCell = tNumSpaceDims + 1;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    using ConfigT = ResidualEvalT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using PrevVelT = ResidualEvalT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 0.12; tHostVelocity(1, 0) = 0.22;
    tHostVelocity(0, 1) = 0.41; tHostVelocity(1, 1) = 0.47;
    tHostVelocity(0, 2) = 0.25; tHostVelocity(1, 2) = 0.86;
    tHostVelocity(0, 3) = 0.15; tHostVelocity(1, 3) = 0.57;
    tHostVelocity(0, 4) = 0.12; tHostVelocity(1, 4) = 0.18;
    tHostVelocity(0, 5) = 0.43; tHostVelocity(1, 5) = 0.11;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    using ControlT = ResidualEvalT::ControlScalarType;
    auto tControl = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::mNumControlDofsPerCell) );
    Plato::blas2::fill(0.5, tControl->mData);
    tWorkSets.set("control", tControl);

    // build criterion
    Plato::ScalarMultiVectorT<ResidualEvalT::ResultScalarType>
        tResult("result", tNumCells, PhysicsT::mNumMomentumDofsPerCell);
    Plato::Fluids::DeviatoricSurfaceForces<PhysicsT, ResidualEvalT>
        tDeviatoricSurfaceForces(tDomain, tInputs.operator*());

    // test function
    tDeviatoricSurfaceForces(tWorkSets, tResult);
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    auto tTol = 1e-4;
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < PhysicsT::mNumMomentumDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.0, tHostResult(tCell, tDof), tTol);
            //printf("Results(Cell=%d,Dof=%d)=%f\n", tCell, tDof, tHostResult(tCell, tDof));
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PressureSurfaceForces)
{
    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::MomentumConservation<tNumSpaceDims>;
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, mesh sets, and spatial domain
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tNumNodesPerCell = 3;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    using ConfigT = ResidualEvalT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using PrevPressT = ResidualEvalT::PreviousMassScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevPressT> > >
        ( Plato::ScalarMultiVectorT<PrevPressT>("previous pressure", tNumCells, PhysicsT::mNumMassDofsPerCell) );
    Plato::blas2::fill(1, tPrevVel->mData);
    tWorkSets.set("previous pressure", tPrevVel);

    // build criterion
    Plato::ScalarMultiVectorT<ResidualEvalT::ResultScalarType>
        tResult("result", tNumCells, PhysicsT::mNumMomentumDofsPerCell);
    Plato::Fluids::PressureSurfaceForces<PhysicsT, ResidualEvalT>
        tPressureSurfaceForces(tDomain, "x+");

    // test function
    tPressureSurfaceForces(tWorkSets, tResult);
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    
    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{0,0,0.5,0,0.5,0},{0,0,0,0,0,0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < PhysicsT::mNumMomentumDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
            //printf("Results(Cell=%d,Dof=%d)=%f\n", tCell, tDof, tHostResult(tCell, tDof));
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StrainRate)
{
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    constexpr Plato::OrdinalType tNumNodesPerCell = 3;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    TEST_EQUALITY(2, tMesh->nelems());

    auto const tNumCells = tMesh->nelems();
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    Plato::ScalarArray3D tConfig("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims);
    Plato::workset_config_scalar<tNumSpaceDims,tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig);

    Plato::ScalarVector tVolume("volume", tNumCells);
    Plato::ScalarArray3D tGradient("gradient", tNumCells, tNumNodesPerCell, tNumSpaceDims);
    Plato::ScalarArray3D tStrainRate("strain rate", tNumCells, tNumNodesPerCell, tNumSpaceDims);
    Plato::ComputeGradientWorkset<tNumSpaceDims> tComputeGradient;

    auto tNumDofsPerCell = tNumSpaceDims * tNumNodesPerCell;
    Plato::ScalarMultiVector tVelocity("velocity", tNumCells, tNumDofsPerCell);
    auto tHostVelocity = Kokkos::create_mirror(tVelocity);
    tHostVelocity(0, 0) = 0.12; tHostVelocity(1, 0) = 0.22;
    tHostVelocity(0, 1) = 0.41; tHostVelocity(1, 1) = 0.47;
    tHostVelocity(0, 2) = 0.25; tHostVelocity(1, 2) = 0.86;
    tHostVelocity(0, 3) = 0.15; tHostVelocity(1, 3) = 0.57;
    tHostVelocity(0, 4) = 0.12; tHostVelocity(1, 4) = 0.18;
    tHostVelocity(0, 5) = 0.43; tHostVelocity(1, 5) = 0.11;
    Kokkos::deep_copy(tVelocity, tHostVelocity);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfig, tVolume);
        Plato::Fluids::strain_rate<tNumNodesPerCell, tNumSpaceDims>(aCellOrdinal, tVelocity, tGradient, tStrainRate);
    }, "strain_rate unit test");

    auto tTol = 1e-6;
    auto tHostStrainRate = Kokkos::create_mirror(tStrainRate);
    Kokkos::deep_copy(tHostStrainRate, tStrainRate);
    // cell 1
    TEST_FLOATING_EQUALITY(0.13,   tHostStrainRate(0, 0, 0), tTol);
    TEST_FLOATING_EQUALITY(-0.195, tHostStrainRate(0, 0, 1), tTol);
    TEST_FLOATING_EQUALITY(-0.195, tHostStrainRate(0, 1, 0), tTol);
    TEST_FLOATING_EQUALITY(0.28,   tHostStrainRate(0, 1, 1), tTol);
    // cell 2
    TEST_FLOATING_EQUALITY(-0.64, tHostStrainRate(1, 0, 0), tTol);
    TEST_FLOATING_EQUALITY(0.29,  tHostStrainRate(1, 0, 1), tTol);
    TEST_FLOATING_EQUALITY(0.29,  tHostStrainRate(1, 1, 0), tTol);
    TEST_FLOATING_EQUALITY(0.46,  tHostStrainRate(1, 1, 1), tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS1_DeviceScale)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarMultiVector tInput("input", tNumCells, tNumSpaceDims);
    Plato::blas2::fill(1.0, tInput);
    Plato::ScalarMultiVector tOutput("output", tNumCells, tNumSpaceDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas1::scale<tNumSpaceDims>(aCellOrdinal, 4.0, tInput, tOutput);
    }, "device blas1::scale");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDim = 0; tDim < tNumSpaceDims; tDim++)
        {
            TEST_FLOATING_EQUALITY(4.0, tHostOutput(tCell, tDim), tTol);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS1_DeviceScale_Version2)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarMultiVector tInput("input", tNumCells, tNumSpaceDims);
    Plato::blas2::fill(1.0, tInput);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas1::scale<tNumSpaceDims>(aCellOrdinal, 4.0, tInput);
    }, "device blas1::scale");

    auto tTol = 1e-6;
    auto tHostInput = Kokkos::create_mirror(tInput);
    Kokkos::deep_copy(tHostInput, tInput);
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDim = 0; tDim < tNumSpaceDims; tDim++)
        {
            TEST_FLOATING_EQUALITY(4.0, tHostInput(tCell, tDim), tTol);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS1_Dot)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarMultiVector tInputA("input", tNumCells, tNumSpaceDims);
    Plato::blas2::fill(1.0, tInputA);
    Plato::ScalarMultiVector tInputB("input", tNumCells, tNumSpaceDims);
    Plato::blas2::fill(4.0, tInputB);
    Plato::ScalarVector tOutput("output", tNumCells, tNumSpaceDims, tNumSpaceDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas1::dot<tNumSpaceDims>(aCellOrdinal, tInputA, tInputB, tOutput);
    }, "device blas1::dot");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        TEST_FLOATING_EQUALITY(8.0, tHostOutput(tCell), tTol);
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS2_DeviceScale)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarArray3D tInput("input", tNumCells, tNumSpaceDims, tNumSpaceDims);
    Plato::blas3::fill<tNumSpaceDims, tNumSpaceDims>(tNumCells, 1.0, tInput);
    Plato::ScalarArray3D tOutput("output", tNumCells, tNumSpaceDims, tNumSpaceDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas2::scale<tNumSpaceDims, tNumSpaceDims>(aCellOrdinal, 4.0, tInput, tOutput);
    }, "device blas2::scale");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDimI = 0; tDimI < tNumSpaceDims; tDimI++)
        {
            for (Plato::OrdinalType tDimJ = 0; tDimJ < tNumSpaceDims; tDimJ++)
            {
                TEST_FLOATING_EQUALITY(4.0, tHostOutput(tCell, tDimI, tDimJ), tTol);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS2_Dot)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarArray3D tInputA("input", tNumCells, tNumSpaceDims, tNumSpaceDims);
    Plato::blas3::fill<tNumSpaceDims, tNumSpaceDims>(tNumCells, 1.0, tInputA);
    Plato::ScalarArray3D tInputB("input", tNumCells, tNumSpaceDims, tNumSpaceDims);
    Plato::blas3::fill<tNumSpaceDims, tNumSpaceDims>(tNumCells, 4.0, tInputB);
    Plato::ScalarVector tOutput("output", tNumCells, tNumSpaceDims, tNumSpaceDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas2::dot<tNumSpaceDims, tNumSpaceDims>(aCellOrdinal, tInputA, tInputB, tOutput);
    }, "device blas2::dot");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        TEST_FLOATING_EQUALITY(16.0, tHostOutput(tCell), tTol);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BrinkmanPenalization)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumNodesPerCell = 4;
    Plato::Scalar tPhysicalNum = 1.0;
    Plato::Scalar tConvexityParam = 0.5;
    Plato::ScalarVector tOutput("output", tNumCells);
    Plato::ScalarMultiVector tControlWS("control", tNumCells, tNumNodesPerCell);
    Plato::blas2::fill(0.5, tControlWS);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tOutput(aCellOrdinal) =
            Plato::Fluids::brinkman_penalization<tNumNodesPerCell>(aCellOrdinal, tPhysicalNum, tConvexityParam, tControlWS);
    }, "brinkman_penalization unit test");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for(Plato::OrdinalType tIndex = 0; tIndex < tOutput.size(); tIndex++)
    {
        TEST_FLOATING_EQUALITY(0.4, tHostOutput(tIndex), tTol);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, RampPenalization)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumNodesPerCell = 4;
    Plato::Scalar tPhysicalNum = 1.0;
    Plato::Scalar tConvexityParam = 0.5;
    Plato::ScalarVector tOutput("output", tNumCells);
    Plato::ScalarMultiVector tControlWS("control", tNumCells, tNumNodesPerCell);
    Plato::blas2::fill(0.5, tControlWS);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tOutput(aCellOrdinal) =
            Plato::Fluids::ramp_penalization<tNumNodesPerCell>(aCellOrdinal, tPhysicalNum, tConvexityParam, tControlWS);
    }, "ramp_penalization unit test");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for(Plato::OrdinalType tIndex = 0; tIndex < tOutput.size(); tIndex++)
    {
        TEST_FLOATING_EQUALITY(0.6, tHostOutput(tIndex), tTol);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, InternalDissipationEnergyIncompressible_Value)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList  name='Criteria'>"
            "    <Parameter  name='Type'    type='string'    value='Scalar Function'/>"
            "    <ParameterList name='My Criteria'>"
            "      <Parameter  name='Flow'                 type='string'    value='Incompressible'/>"
            "      <Parameter  name='Type'                 type='string'    value='Scalar Function'/>"
            "      <Parameter  name='Scalar Function Type' type='string'    value='Internal Dissipation Energy'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Dimensionless Properties'>"
            "    <Parameter  name='Darcy Number'    type='double'    value='1.0'/>"
            "    <Parameter  name='Prandtl Number' type='double'    value='1.0'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tMesh->dim();
    Plato::ScalarVector tControl("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(0.1, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(1.5, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::Fluids::PhysicsScalarFunction<PhysicsT>
        tCriterion(tModel, tDataMap, tInputs.operator*(), tFuncName);
    TEST_EQUALITY("My Criteria", tCriterion.name());

    // test criterion value
    auto tTol = 1e-4;
    auto tValue = tCriterion.value(tControl, tPrimal);
    TEST_FLOATING_EQUALITY(0.222222, tValue, tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, InternalDissipationEnergyIncompressible_GradControl)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList  name='Criteria'>"
            "    <Parameter  name='Type'    type='string'    value='Scalar Function'/>"
            "    <ParameterList name='My Criteria'>"
            "      <Parameter  name='Flow'                 type='string'    value='Incompressible'/>"
            "      <Parameter  name='Type'                 type='string'    value='Scalar Function'/>"
            "      <Parameter  name='Scalar Function Type' type='string'    value='Internal Dissipation Energy'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Dimensionless Properties'>"
            "    <Parameter  name='Darcy Number'    type='double'    value='1.0'/>"
            "    <Parameter  name='Prandtl Number'  type='double'    value='1.0'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tMesh->dim();
    Plato::ScalarVector tControl("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    auto tHostCurVel = Kokkos::create_mirror(tCurVel);
    tHostCurVel(0, 0) = 0.12; tHostCurVel(1, 0) = 0.22;
    tHostCurVel(0, 1) = 0.41; tHostCurVel(1, 1) = 0.47;
    tHostCurVel(0, 2) = 0.25; tHostCurVel(1, 2) = 0.86;
    tHostCurVel(0, 3) = 0.15; tHostCurVel(1, 3) = 0.57;
    tHostCurVel(0, 4) = 0.12; tHostCurVel(1, 4) = 0.18;
    tHostCurVel(0, 5) = 0.43; tHostCurVel(1, 5) = 0.11;
    Kokkos::deep_copy(tCurVel, tHostCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(0.1, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(1.5, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::Fluids::PhysicsScalarFunction<PhysicsT>
        tCriterion(tModel, tDataMap, tInputs.operator*(), tFuncName);
    TEST_EQUALITY("My Criteria", tCriterion.name());

    // test criterion value
    auto tTol = 1e-4;
    auto tGradient = tCriterion.gradientControl(tControl, tPrimal);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    Kokkos::deep_copy(tHostGradient, tGradient);

    std::vector<Plato::Scalar> tGold = {-0.00350222, 0.0, -0.00350222, -0.00350222};
    for(Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostGradient(tNode), tTol);
        //printf("Results(Node=%d)=%f\n", tNode, tHostGradient(tNode));
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, InternalDissipationEnergyIncompressible_GradConfig)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList  name='Criteria'>"
            "    <Parameter  name='Type'    type='string'    value='Scalar Function'/>"
            "    <ParameterList name='My Criteria'>"
            "      <Parameter  name='Flow'                 type='string'    value='Incompressible'/>"
            "      <Parameter  name='Type'                 type='string'    value='Scalar Function'/>"
            "      <Parameter  name='Scalar Function Type' type='string'    value='Internal Dissipation Energy'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Dimensionless Properties'>"
            "    <Parameter  name='Darcy Number'    type='double'    value='1.0'/>"
            "    <Parameter  name='Prandtl Number'  type='double'    value='1.0'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tMesh->dim();
    Plato::ScalarVector tControl("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    auto tHostCurVel = Kokkos::create_mirror(tCurVel);
    tHostCurVel(0, 0) = 0.12; tHostCurVel(1, 0) = 0.22;
    tHostCurVel(0, 1) = 0.41; tHostCurVel(1, 1) = 0.47;
    tHostCurVel(0, 2) = 0.25; tHostCurVel(1, 2) = 0.86;
    tHostCurVel(0, 3) = 0.15; tHostCurVel(1, 3) = 0.57;
    tHostCurVel(0, 4) = 0.12; tHostCurVel(1, 4) = 0.18;
    tHostCurVel(0, 5) = 0.43; tHostCurVel(1, 5) = 0.11;
    Kokkos::deep_copy(tCurVel, tHostCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(0.1, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(1.5, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::Fluids::PhysicsScalarFunction<PhysicsT>
        tCriterion(tModel, tDataMap, tInputs.operator*(), tFuncName);
    TEST_EQUALITY("My Criteria", tCriterion.name());

    // test criterion value
    auto tTol = 1e-4;
    auto tGradient = tCriterion.gradientConfig(tControl, tPrimal);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    Kokkos::deep_copy(tHostGradient, tGradient);

    std::vector<Plato::Scalar> tGold = {0.377522, 0.2091, -0.2091, -0.1145};
    for(Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostGradient(tNode), tTol);
        //printf("Results(Node=%d)=%f\n", tNode, tHostGradient(tNode));
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AverageSurfacePressure_Value)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList  name='Criteria'>"
            "    <Parameter  name='Type'    type='string'    value='Scalar Function'/>"
            "    <ParameterList name='My Criteria'>"
            "      <Parameter  name='Type'                 type='string'        value='Scalar Function'/>"
            "      <Parameter  name='Sides'                type='Array(string)' value='{x+}'/>"
            "      <Parameter  name='Scalar Function Type' type='string'        value='Average Surface Pressure'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tMesh->dim();
    Plato::ScalarVector tControl("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(0.1, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(1.5, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::Fluids::PhysicsScalarFunction<PhysicsT>
        tCriterion(tModel, tDataMap, tInputs.operator*(), tFuncName);
    TEST_EQUALITY("My Criteria", tCriterion.name());

    // test criterion value
    auto tTol = 1e-4;
    auto tValue = tCriterion.value(tControl, tPrimal);
    TEST_FLOATING_EQUALITY(0.133333, tValue, tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AverageSurfacePressure_GradCurPress)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList  name='Criteria'>"
            "    <Parameter  name='Type'    type='string'    value='Scalar Function'/>"
            "    <ParameterList name='My Criteria'>"
            "      <Parameter  name='Type'                 type='string'        value='Scalar Function'/>"
            "      <Parameter  name='Sides'                type='Array(string)' value='{x+}'/>"
            "      <Parameter  name='Scalar Function Type' type='string'        value='Average Surface Pressure'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tMesh->dim();
    Plato::ScalarVector tControl("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(0.1, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(1.5, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::Fluids::PhysicsScalarFunction<PhysicsT>
        tCriterion(tModel, tDataMap, tInputs.operator*(), tFuncName);
    TEST_EQUALITY("My Criteria", tCriterion.name());

    // test criterion value
    auto tGradCurPress = tCriterion.gradientCurrentPress(tControl, tPrimal);
    auto tHostGradCurPress = Kokkos::create_mirror(tGradCurPress);
    Kokkos::deep_copy(tHostGradCurPress, tGradCurPress);

    auto tTol = 1e-6;
    std::vector<Plato::Scalar> tGold = {0.5, 0.0, 0.333333333, 0.5};
    for(Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostGradCurPress(tNode), tTol);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AverageSurfaceTemperature_Value)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList  name='Criteria'>"
            "    <Parameter  name='Type'    type='string'    value='Scalar Function'/>"
            "    <ParameterList name='My Criteria'>"
            "      <Parameter  name='Type'                 type='string'        value='Scalar Function'/>"
            "      <Parameter  name='Sides'                type='Array(string)' value='{x+}'/>"
            "      <Parameter  name='Scalar Function Type' type='string'        value='Average Surface Temperature'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tMesh->dim();
    Plato::ScalarVector tControl("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(0.1, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(1.5, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::Fluids::PhysicsScalarFunction<PhysicsT>
        tCriterion(tModel, tDataMap, tInputs.operator*(), tFuncName);
    TEST_EQUALITY("My Criteria", tCriterion.name());

    // test criterion value
    auto tTol = 1e-6;
    auto tValue = tCriterion.value(tControl, tPrimal);
    TEST_FLOATING_EQUALITY(2.0, tValue, tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AverageSurfaceTemperature_GradCurTemp)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList  name='Criteria'>"
            "    <Parameter  name='Type'    type='string'    value='Scalar Function'/>"
            "    <ParameterList name='My Criteria'>"
            "      <Parameter  name='Type'                 type='string'        value='Scalar Function'/>"
            "      <Parameter  name='Sides'                type='Array(string)' value='{x+}'/>"
            "      <Parameter  name='Scalar Function Type' type='string'        value='Average Surface Temperature'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tMesh->dim();
    Plato::ScalarVector tControl("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(0.1, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(1.5, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::Fluids::PhysicsScalarFunction<PhysicsT>
        tCriterion(tModel, tDataMap, tInputs.operator*(), tFuncName);
    TEST_EQUALITY("My Criteria", tCriterion.name());

    // test criterion value
    auto tGradCurTemp = tCriterion.gradientCurrentTemp(tControl, tPrimal);
    auto tHostGradCurTemp = Kokkos::create_mirror(tGradCurTemp);
    Kokkos::deep_copy(tHostGradCurTemp, tGradCurTemp);

    auto tTol = 1e-6;
    std::vector<Plato::Scalar> tGold = {0.5, 0.0, 0.333333333, 0.5};
    for(Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostGradCurTemp(tNode), tTol);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BuildScalarFunctionWorksets_SpatialDomain)
{
    // build mesh and spatial domain
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tNumSpaceDim;
    Plato::ScalarVector tControls("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControls);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(2.0, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(3.0, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // call build_scalar_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);
    Plato::Fluids::build_scalar_function_worksets<ResidualEvalT>
        (tDomain, tControls, tPrimal, tOrdinalMaps, tWorkSets);

    // test current velocity results
    auto tCurVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMomentumScalarType>>(tWorkSets.get("current velocity"));
    TEST_EQUALITY(tNumCells, tCurVelWS.extent(0));
    auto tNumVelDofsPerCell = PhysicsT::mNumMomentumDofsPerCell;
    TEST_EQUALITY(tNumVelDofsPerCell, tCurVelWS.extent(1));
    auto tHostCurVelWS = Kokkos::create_mirror(tCurVelWS);
    Kokkos::deep_copy(tHostCurVelWS, tCurVelWS);
    const Plato::Scalar tTol = 1e-6;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.0, tHostCurVelWS(tCell, tDof), tTol);
        }
    }

    // test current pressure results
    auto tCurPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMassScalarType>>(tWorkSets.get("current pressure"));
    TEST_EQUALITY(tNumCells, tCurPressWS.extent(0));
    auto tNumPressDofsPerCell = PhysicsT::mNumMassDofsPerCell;
    TEST_EQUALITY(tNumPressDofsPerCell, tCurPressWS.extent(1));
    auto tHostCurPressWS = Kokkos::create_mirror(tCurPressWS);
    Kokkos::deep_copy(tHostCurPressWS, tCurPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.0, tHostCurPressWS(tCell, tDof), tTol);
        }
    }

    // test current temperature results
    auto tCurTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentEnergyScalarType>>(tWorkSets.get("current temperature"));
    TEST_EQUALITY(tNumCells, tCurTempWS.extent(0));
    auto tNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell;
    TEST_EQUALITY(tNumTempDofsPerCell, tCurTempWS.extent(1));
    auto tHostCurTempWS = Kokkos::create_mirror(tCurTempWS);
    Kokkos::deep_copy(tHostCurTempWS, tCurTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(3.0, tHostCurTempWS(tCell, tDof), tTol);
        }
    }

    // test time steps results
    auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("time steps"));
    TEST_EQUALITY(tNumCells, tTimeStepWS.extent(0));
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    TEST_EQUALITY(tNumNodesPerCell, tTimeStepWS.extent(1));
    auto tHostTimeStepWS = Kokkos::create_mirror(tTimeStepWS);
    Kokkos::deep_copy(tHostTimeStepWS, tTimeStepWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(4.0, tHostTimeStepWS(tCell, tDof), tTol);
        }
    }

    // test controls results
    auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::ControlScalarType>>(tWorkSets.get("control"));
    TEST_EQUALITY(tNumCells, tControlWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tControlWS.extent(1));
    auto tHostControlWS = Kokkos::create_mirror(tControlWS);
    Kokkos::deep_copy(tHostControlWS, tControlWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.5, tHostControlWS(tCell, tDof), tTol);
        }
    }

    // test configuration results
    auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ResidualEvalT::ConfigScalarType>>(tWorkSets.get("configuration"));
    TEST_EQUALITY(tNumCells, tConfigWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tConfigWS.extent(1));
    auto tNumConfigDofsPerNode = PhysicsT::mNumConfigDofsPerNode;
    TEST_EQUALITY(tNumConfigDofsPerNode, tConfigWS.extent(2));
    auto tHostConfigWS = Kokkos::create_mirror(tConfigWS);
    Kokkos::deep_copy(tHostConfigWS, tConfigWS);
    Plato::ScalarArray3D tGoldConfigWS("gold configuration", tNumCells, tNumNodesPerCell, tNumConfigDofsPerNode);
    auto tHostGoldConfigWS = Kokkos::create_mirror(tGoldConfigWS);
    tHostGoldConfigWS(0,0,0) = 0; tHostGoldConfigWS(0,1,0) = 1; tHostGoldConfigWS(0,2,0) = 1;
    tHostGoldConfigWS(1,0,0) = 1; tHostGoldConfigWS(1,1,0) = 0; tHostGoldConfigWS(1,2,0) = 0;
    tHostGoldConfigWS(0,0,1) = 0; tHostGoldConfigWS(0,1,1) = 0; tHostGoldConfigWS(0,2,1) = 1;
    tHostGoldConfigWS(1,0,1) = 1; tHostGoldConfigWS(1,1,1) = 1; tHostGoldConfigWS(1,2,1) = 0;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tNode = 0; tNode < tNumNodesPerCell; tNode++)
        {
            for (decltype(tNumConfigDofsPerNode) tDof = 0; tDof < tNumConfigDofsPerNode; tDof++)
            {
                TEST_FLOATING_EQUALITY(tHostGoldConfigWS(tCell, tNode, tDof), tHostConfigWS(tCell, tNode, tDof), tTol);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BuildVectorFunctionWorksets_SpatialDomain)
{
    // build mesh and spatial domain
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tNumSpaceDim;
    Plato::ScalarVector tControls("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControls);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    Plato::blas1::fill(0.1, tCurPred);
    tPrimal.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(2.0, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(3.0, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tPrevVel("previous velocity", tNumVelDofs);
    Plato::blas1::fill(0.8, tPrevVel);
    tPrimal.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.8, tPrevPress);
    tPrimal.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    Plato::blas1::fill(2.8, tPrevTemp);
    tPrimal.vector("previous temperature", tPrevTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);
    Plato::ScalarVector tArtCompress("artificial compressibility", tNumNodes);
    Plato::blas1::fill(5.0, tArtCompress);
    tPrimal.vector("artificial compressibility", tArtCompress);
    Plato::ScalarVector tVelBCs("prescribed velocity", tNumVelDofs);
    Plato::blas1::fill(6.0, tVelBCs);
    tPrimal.vector("prescribed velocity", tVelBCs);

    // call build_vector_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);
    Plato::Fluids::build_vector_function_worksets<ResidualEvalT>
        (tNumCells, tControls, tPrimal, tOrdinalMaps, tWorkSets);

    // test current velocity results
    auto tCurVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMomentumScalarType>>(tWorkSets.get("current velocity"));
    TEST_EQUALITY(tNumCells, tCurVelWS.extent(0));
    auto tNumVelDofsPerCell = PhysicsT::mNumMomentumDofsPerCell;
    TEST_EQUALITY(tNumVelDofsPerCell, tCurVelWS.extent(1));
    auto tHostCurVelWS = Kokkos::create_mirror(tCurVelWS);
    Kokkos::deep_copy(tHostCurVelWS, tCurVelWS);
    const Plato::Scalar tTol = 1e-6;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.0, tHostCurVelWS(tCell, tDof), tTol);
        }
    }

    // test current pressure results
    auto tCurPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMassScalarType>>(tWorkSets.get("current pressure"));
    TEST_EQUALITY(tNumCells, tCurPressWS.extent(0));
    auto tNumPressDofsPerCell = PhysicsT::mNumMassDofsPerCell;
    TEST_EQUALITY(tNumPressDofsPerCell, tCurPressWS.extent(1));
    auto tHostCurPressWS = Kokkos::create_mirror(tCurPressWS);
    Kokkos::deep_copy(tHostCurPressWS, tCurPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.0, tHostCurPressWS(tCell, tDof), tTol);
        }
    }

    // test current temperature results
    auto tCurTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentEnergyScalarType>>(tWorkSets.get("current temperature"));
    TEST_EQUALITY(tNumCells, tCurTempWS.extent(0));
    auto tNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell;
    TEST_EQUALITY(tNumTempDofsPerCell, tCurTempWS.extent(1));
    auto tHostCurTempWS = Kokkos::create_mirror(tCurTempWS);
    Kokkos::deep_copy(tHostCurTempWS, tCurTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(3.0, tHostCurTempWS(tCell, tDof), tTol);
        }
    }

    // test previous velocity results
    auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousMomentumScalarType>>(tWorkSets.get("previous velocity"));
    TEST_EQUALITY(tNumCells, tPrevVelWS.extent(0));
    TEST_EQUALITY(tNumVelDofsPerCell, tPrevVelWS.extent(1));
    auto tHostPrevVelWS = Kokkos::create_mirror(tPrevVelWS);
    Kokkos::deep_copy(tHostPrevVelWS, tPrevVelWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.8, tHostPrevVelWS(tCell, tDof), tTol);
        }
    }

    // test previous pressure results
    auto tPrevPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousMassScalarType>>(tWorkSets.get("previous pressure"));
    TEST_EQUALITY(tNumCells, tPrevPressWS.extent(0));
    TEST_EQUALITY(tNumPressDofsPerCell, tPrevPressWS.extent(1));
    auto tHostPrevPressWS = Kokkos::create_mirror(tPrevPressWS);
    Kokkos::deep_copy(tHostPrevPressWS, tPrevPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.8, tHostPrevPressWS(tCell, tDof), tTol);
        }
    }

    // test previous temperature results
    auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousEnergyScalarType>>(tWorkSets.get("previous temperature"));
    TEST_EQUALITY(tNumCells, tPrevTempWS.extent(0));
    TEST_EQUALITY(tNumTempDofsPerCell, tPrevTempWS.extent(1));
    auto tHostPrevTempWS = Kokkos::create_mirror(tPrevTempWS);
    Kokkos::deep_copy(tHostPrevTempWS, tPrevTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.8, tHostPrevTempWS(tCell, tDof), tTol);
        }
    }

    // test time steps results
    auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("time steps"));
    TEST_EQUALITY(tNumCells, tTimeStepWS.extent(0));
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    TEST_EQUALITY(tNumNodesPerCell, tTimeStepWS.extent(1));
    auto tHostTimeStepWS = Kokkos::create_mirror(tTimeStepWS);
    Kokkos::deep_copy(tHostTimeStepWS, tTimeStepWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(4.0, tHostTimeStepWS(tCell, tDof), tTol);
        }
    }

    // test artificial compressibility results
    auto tArtCompressWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("artificial compressibility"));
    TEST_EQUALITY(tNumCells, tArtCompressWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tArtCompressWS.extent(1));
    auto tHostArtCompressWS = Kokkos::create_mirror(tArtCompressWS);
    Kokkos::deep_copy(tHostArtCompressWS, tArtCompressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(5.0, tHostArtCompressWS(tCell, tDof), tTol);
        }
    }

    // test prescribed velocity results
    auto tVelBCsWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("prescribed velocity"));
    TEST_EQUALITY(tNumCells, tVelBCsWS.extent(0));
    TEST_EQUALITY(tNumVelDofsPerCell, tVelBCsWS.extent(1));
    auto tHostVelBCsWS = Kokkos::create_mirror(tVelBCsWS);
    Kokkos::deep_copy(tHostVelBCsWS, tVelBCsWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(6.0, tHostVelBCsWS(tCell, tDof), tTol);
        }
    }

    // test controls results
    auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::ControlScalarType>>(tWorkSets.get("control"));
    TEST_EQUALITY(tNumCells, tControlWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tControlWS.extent(1));
    auto tHostControlWS = Kokkos::create_mirror(tControlWS);
    Kokkos::deep_copy(tHostControlWS, tControlWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.5, tHostControlWS(tCell, tDof), tTol);
        }
    }

    // test configuration results
    auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ResidualEvalT::ConfigScalarType>>(tWorkSets.get("configuration"));
    TEST_EQUALITY(tNumCells, tConfigWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tConfigWS.extent(1));
    auto tNumConfigDofsPerNode = PhysicsT::mNumConfigDofsPerNode;
    TEST_EQUALITY(tNumConfigDofsPerNode, tConfigWS.extent(2));
    auto tHostConfigWS = Kokkos::create_mirror(tConfigWS);
    Kokkos::deep_copy(tHostConfigWS, tConfigWS);
    Plato::ScalarArray3D tGoldConfigWS("gold configuration", tNumCells, tNumNodesPerCell, tNumConfigDofsPerNode);
    auto tHostGoldConfigWS = Kokkos::create_mirror(tGoldConfigWS);
    tHostGoldConfigWS(0,0,0) = 0; tHostGoldConfigWS(0,1,0) = 1; tHostGoldConfigWS(0,2,0) = 1;
    tHostGoldConfigWS(1,0,0) = 1; tHostGoldConfigWS(1,1,0) = 0; tHostGoldConfigWS(1,2,0) = 0;
    tHostGoldConfigWS(0,0,1) = 0; tHostGoldConfigWS(0,1,1) = 0; tHostGoldConfigWS(0,2,1) = 1;
    tHostGoldConfigWS(1,0,1) = 1; tHostGoldConfigWS(1,1,1) = 1; tHostGoldConfigWS(1,2,1) = 0;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tNode = 0; tNode < tNumNodesPerCell; tNode++)
        {
            for (decltype(tNumConfigDofsPerNode) tDof = 0; tDof < tNumConfigDofsPerNode; tDof++)
            {
                TEST_FLOATING_EQUALITY(tHostGoldConfigWS(tCell, tNode, tDof), tHostConfigWS(tCell, tNode, tDof), tTol);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BuildVectorFunctionWorksets)
{
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = 2;
    auto tNumNodes = 4;
    auto tNumVelDofs = tNumNodes * tNumSpaceDim;
    Plato::ScalarVector tControls("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControls);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    Plato::blas1::fill(0.1, tCurPred);
    tPrimal.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(2.0, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(3.0, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tPrevVel("previous velocity", tNumVelDofs);
    Plato::blas1::fill(0.8, tPrevVel);
    tPrimal.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.8, tPrevPress);
    tPrimal.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    Plato::blas1::fill(2.8, tPrevTemp);
    tPrimal.vector("previous temperature", tPrevTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);
    Plato::ScalarVector tArtCompress("artificial compressibility", tNumNodes);
    Plato::blas1::fill(5.0, tArtCompress);
    tPrimal.vector("artificial compressibility", tArtCompress);
    Plato::ScalarVector tVelBCs("prescribed velocity", tNumVelDofs);
    Plato::blas1::fill(6.0, tVelBCs);
    tPrimal.vector("prescribed velocity", tVelBCs);

    // set ordinal maps;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);

    // call build_vector_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::Fluids::build_vector_function_worksets<ResidualEvalT>
        (tNumCells, tControls, tPrimal, tOrdinalMaps, tWorkSets);

    // test current velocity results
    auto tCurVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMomentumScalarType>>(tWorkSets.get("current velocity"));
    TEST_EQUALITY(tNumCells, tCurVelWS.extent(0));
    auto tNumVelDofsPerCell = PhysicsT::mNumMomentumDofsPerCell;
    TEST_EQUALITY(tNumVelDofsPerCell, tCurVelWS.extent(1));
    auto tHostCurVelWS = Kokkos::create_mirror(tCurVelWS);
    Kokkos::deep_copy(tHostCurVelWS, tCurVelWS);
    const Plato::Scalar tTol = 1e-6;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.0, tHostCurVelWS(tCell, tDof), tTol);
        }
    }

    // test current pressure results
    auto tCurPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMassScalarType>>(tWorkSets.get("current pressure"));
    TEST_EQUALITY(tNumCells, tCurPressWS.extent(0));
    auto tNumPressDofsPerCell = PhysicsT::mNumMassDofsPerCell;
    TEST_EQUALITY(tNumPressDofsPerCell, tCurPressWS.extent(1));
    auto tHostCurPressWS = Kokkos::create_mirror(tCurPressWS);
    Kokkos::deep_copy(tHostCurPressWS, tCurPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.0, tHostCurPressWS(tCell, tDof), tTol);
        }
    }

    // test current temperature results
    auto tCurTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentEnergyScalarType>>(tWorkSets.get("current temperature"));
    TEST_EQUALITY(tNumCells, tCurTempWS.extent(0));
    auto tNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell;
    TEST_EQUALITY(tNumTempDofsPerCell, tCurTempWS.extent(1));
    auto tHostCurTempWS = Kokkos::create_mirror(tCurTempWS);
    Kokkos::deep_copy(tHostCurTempWS, tCurTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(3.0, tHostCurTempWS(tCell, tDof), tTol);
        }
    }

    // test previous velocity results
    auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousMomentumScalarType>>(tWorkSets.get("previous velocity"));
    TEST_EQUALITY(tNumCells, tPrevVelWS.extent(0));
    TEST_EQUALITY(tNumVelDofsPerCell, tPrevVelWS.extent(1));
    auto tHostPrevVelWS = Kokkos::create_mirror(tPrevVelWS);
    Kokkos::deep_copy(tHostPrevVelWS, tPrevVelWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.8, tHostPrevVelWS(tCell, tDof), tTol);
        }
    }

    // test previous pressure results
    auto tPrevPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousMassScalarType>>(tWorkSets.get("previous pressure"));
    TEST_EQUALITY(tNumCells, tPrevPressWS.extent(0));
    TEST_EQUALITY(tNumPressDofsPerCell, tPrevPressWS.extent(1));
    auto tHostPrevPressWS = Kokkos::create_mirror(tPrevPressWS);
    Kokkos::deep_copy(tHostPrevPressWS, tPrevPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.8, tHostPrevPressWS(tCell, tDof), tTol);
        }
    }

    // test previous temperature results
    auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousEnergyScalarType>>(tWorkSets.get("previous temperature"));
    TEST_EQUALITY(tNumCells, tPrevTempWS.extent(0));
    TEST_EQUALITY(tNumTempDofsPerCell, tPrevTempWS.extent(1));
    auto tHostPrevTempWS = Kokkos::create_mirror(tPrevTempWS);
    Kokkos::deep_copy(tHostPrevTempWS, tPrevTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.8, tHostPrevTempWS(tCell, tDof), tTol);
        }
    }

    // test time steps results
    auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("time steps"));
    TEST_EQUALITY(tNumCells, tTimeStepWS.extent(0));
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    TEST_EQUALITY(tNumNodesPerCell, tTimeStepWS.extent(1));
    auto tHostTimeStepWS = Kokkos::create_mirror(tTimeStepWS);
    Kokkos::deep_copy(tHostTimeStepWS, tTimeStepWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(4.0, tHostTimeStepWS(tCell, tDof), tTol);
        }
    }

    // test artificial compressibility results
    auto tArtCompressWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("artificial compressibility"));
    TEST_EQUALITY(tNumCells, tArtCompressWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tArtCompressWS.extent(1));
    auto tHostArtCompressWS = Kokkos::create_mirror(tArtCompressWS);
    Kokkos::deep_copy(tHostArtCompressWS, tArtCompressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(5.0, tHostArtCompressWS(tCell, tDof), tTol);
        }
    }

    // test controls results
    auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::ControlScalarType>>(tWorkSets.get("control"));
    TEST_EQUALITY(tNumCells, tControlWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tControlWS.extent(1));
    auto tHostControlWS = Kokkos::create_mirror(tControlWS);
    Kokkos::deep_copy(tHostControlWS, tControlWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.5, tHostControlWS(tCell, tDof), tTol);
        }
    }

    // test prescribed velocity results
    auto tVelBCsWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("prescribed velocity"));
    TEST_EQUALITY(tNumCells, tVelBCsWS.extent(0));
    TEST_EQUALITY(tNumVelDofsPerCell, tVelBCsWS.extent(1));
    auto tHostVelBCsWS = Kokkos::create_mirror(tVelBCsWS);
    Kokkos::deep_copy(tHostVelBCsWS, tVelBCsWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(6.0, tHostVelBCsWS(tCell, tDof), tTol);
        }
    }

    // test configuration results
    auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ResidualEvalT::ConfigScalarType>>(tWorkSets.get("configuration"));
    TEST_EQUALITY(tNumCells, tConfigWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tConfigWS.extent(1));
    auto tNumConfigDofsPerNode = PhysicsT::mNumConfigDofsPerNode;
    TEST_EQUALITY(tNumConfigDofsPerNode, tConfigWS.extent(2));
    auto tHostConfigWS = Kokkos::create_mirror(tConfigWS);
    Kokkos::deep_copy(tHostConfigWS, tConfigWS);
    Plato::ScalarArray3D tGoldConfigWS("gold configuration", tNumCells, tNumNodesPerCell, tNumConfigDofsPerNode);
    auto tHostGoldConfigWS = Kokkos::create_mirror(tGoldConfigWS);
    tHostGoldConfigWS(0,0,0) = 0; tHostGoldConfigWS(0,1,0) = 1; tHostGoldConfigWS(0,2,0) = 1;
    tHostGoldConfigWS(1,0,0) = 1; tHostGoldConfigWS(1,1,0) = 0; tHostGoldConfigWS(1,2,0) = 0;
    tHostGoldConfigWS(0,0,1) = 0; tHostGoldConfigWS(0,1,1) = 0; tHostGoldConfigWS(0,2,1) = 1;
    tHostGoldConfigWS(1,0,1) = 1; tHostGoldConfigWS(1,1,1) = 1; tHostGoldConfigWS(1,2,1) = 0;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tNode = 0; tNode < tNumNodesPerCell; tNode++)
        {
            for (decltype(tNumConfigDofsPerNode) tDof = 0; tDof < tNumConfigDofsPerNode; tDof++)
            {
                TEST_FLOATING_EQUALITY(tHostGoldConfigWS(tCell, tNode, tDof), tHostConfigWS(tCell, tNode, tDof), tTol);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BuildVectorFunctionWorksetsTwo)
{
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = 2;
    auto tNumNodes = 4;
    auto tNumVelDofs = tNumNodes * tNumSpaceDim;
    Plato::ScalarVector tControls("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControls);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    Plato::blas1::fill(0.1, tCurPred);
    tPrimal.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(2.0, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(3.0, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tPrevVel("previous velocity", tNumVelDofs);
    Plato::blas1::fill(0.8, tPrevVel);
    tPrimal.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.8, tPrevPress);
    tPrimal.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    Plato::blas1::fill(2.8, tPrevTemp);
    tPrimal.vector("previous temperature", tPrevTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);
    Plato::ScalarVector tVelBCs("prescribed velocity", tNumVelDofs);
    Plato::blas1::fill(6.0, tVelBCs);
    tPrimal.vector("prescribed velocity", tVelBCs);

    // set ordinal maps;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);

    // call build_vector_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::Fluids::build_vector_function_worksets<ResidualEvalT>
        (tNumCells, tControls, tPrimal, tOrdinalMaps, tWorkSets);
    TEST_EQUALITY(tWorkSets.defined("artifical compressibility"), false);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BuildScalarFunctionWorksets)
{
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = 2;
    auto tNumNodes = 4;
    auto tNumVelDofs = tNumNodes * tNumSpaceDim;
    Plato::ScalarVector tControls("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControls);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(2.0, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(3.0, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // set ordinal maps;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);

    // call build_scalar_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::Fluids::build_scalar_function_worksets<ResidualEvalT>
        (tNumCells, tControls, tPrimal, tOrdinalMaps, tWorkSets);

    // test current velocity results
    auto tCurVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMomentumScalarType>>(tWorkSets.get("current velocity"));
    TEST_EQUALITY(tNumCells, tCurVelWS.extent(0));
    auto tNumVelDofsPerCell = PhysicsT::mNumMomentumDofsPerCell;
    TEST_EQUALITY(tNumVelDofsPerCell, tCurVelWS.extent(1));
    auto tHostCurVelWS = Kokkos::create_mirror(tCurVelWS);
    Kokkos::deep_copy(tHostCurVelWS, tCurVelWS);
    const Plato::Scalar tTol = 1e-6;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.0, tHostCurVelWS(tCell, tDof), tTol);
        }
    }

    // test current pressure results
    auto tCurPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMassScalarType>>(tWorkSets.get("current pressure"));
    TEST_EQUALITY(tNumCells, tCurPressWS.extent(0));
    auto tNumPressDofsPerCell = PhysicsT::mNumMassDofsPerCell;
    TEST_EQUALITY(tNumPressDofsPerCell, tCurPressWS.extent(1));
    auto tHostCurPressWS = Kokkos::create_mirror(tCurPressWS);
    Kokkos::deep_copy(tHostCurPressWS, tCurPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.0, tHostCurPressWS(tCell, tDof), tTol);
        }
    }

    // test current temperature results
    auto tCurTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentEnergyScalarType>>(tWorkSets.get("current temperature"));
    TEST_EQUALITY(tNumCells, tCurTempWS.extent(0));
    auto tNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell;
    TEST_EQUALITY(tNumTempDofsPerCell, tCurTempWS.extent(1));
    auto tHostCurTempWS = Kokkos::create_mirror(tCurTempWS);
    Kokkos::deep_copy(tHostCurTempWS, tCurTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(3.0, tHostCurTempWS(tCell, tDof), tTol);
        }
    }

    // test time steps results
    auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("time steps"));
    TEST_EQUALITY(tNumCells, tTimeStepWS.extent(0));
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    TEST_EQUALITY(tNumNodesPerCell, tTimeStepWS.extent(1));
    auto tHostTimeStepWS = Kokkos::create_mirror(tTimeStepWS);
    Kokkos::deep_copy(tHostTimeStepWS, tTimeStepWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(4.0, tHostTimeStepWS(tCell, tDof), tTol);
        }
    }

    // test controls results
    auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::ControlScalarType>>(tWorkSets.get("control"));
    TEST_EQUALITY(tNumCells, tControlWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tControlWS.extent(1));
    auto tHostControlWS = Kokkos::create_mirror(tControlWS);
    Kokkos::deep_copy(tHostControlWS, tControlWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.5, tHostControlWS(tCell, tDof), tTol);
        }
    }

    // test configuration results
    auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ResidualEvalT::ConfigScalarType>>(tWorkSets.get("configuration"));
    TEST_EQUALITY(tNumCells, tConfigWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tConfigWS.extent(1));
    auto tNumConfigDofsPerNode = PhysicsT::mNumConfigDofsPerNode;
    TEST_EQUALITY(tNumConfigDofsPerNode, tConfigWS.extent(2));
    auto tHostConfigWS = Kokkos::create_mirror(tConfigWS);
    Kokkos::deep_copy(tHostConfigWS, tConfigWS);
    Plato::ScalarArray3D tGoldConfigWS("gold configuration", tNumCells, tNumNodesPerCell, tNumConfigDofsPerNode);
    auto tHostGoldConfigWS = Kokkos::create_mirror(tGoldConfigWS);
    tHostGoldConfigWS(0,0,0) = 0; tHostGoldConfigWS(0,1,0) = 1; tHostGoldConfigWS(0,2,0) = 1;
    tHostGoldConfigWS(1,0,0) = 1; tHostGoldConfigWS(1,1,0) = 0; tHostGoldConfigWS(1,2,0) = 0;
    tHostGoldConfigWS(0,0,1) = 0; tHostGoldConfigWS(0,1,1) = 0; tHostGoldConfigWS(0,2,1) = 1;
    tHostGoldConfigWS(1,0,1) = 1; tHostGoldConfigWS(1,1,1) = 1; tHostGoldConfigWS(1,2,1) = 0;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tNode = 0; tNode < tNumNodesPerCell; tNode++)
        {
            for (decltype(tNumConfigDofsPerNode) tDof = 0; tDof < tNumConfigDofsPerNode; tDof++)
            {
                TEST_FLOATING_EQUALITY(tHostGoldConfigWS(tCell, tNode, tDof), tHostConfigWS(tCell, tNode, tDof), tTol);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ParseArray)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
        Teuchos::getParametersFromXmlString(
            "<ParameterList  name='Criteria'>"
            "  <Parameter  name='Type'         type='string'         value='Weighted Sum'/>"
            "  <Parameter  name='Functions'    type='Array(string)'  value='{My Inlet Pressure, My Outlet Pressure}'/>"
            "  <Parameter  name='Weights'      type='Array(double)'  value='{1.0,-1.0}'/>"
            "  <ParameterList  name='My Inlet Pressure'>"
            "    <Parameter  name='Type'                   type='string'           value='Scalar Function'/>"
            "    <Parameter  name='Scalar Function Type'   type='string'           value='Average Surface Pressure'/>"
            "    <Parameter  name='Sides'                  type='Array(string)'    value='{ss_1}'/>"
            "  </ParameterList>"
            "  <ParameterList  name='My Outlet Pressure'>"
            "    <Parameter  name='Type'                   type='string'           value='Scalar Function'/>"
            "    <Parameter  name='Scalar Function Type'   type='string'           value='Average Surface Pressure'/>"
            "    <Parameter  name='Sides'                  type='Array(string)'    value='{ss_2}'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );
    auto tNames = Plato::parse_array<std::string>("Functions", tParams.operator*());

    std::vector<std::string> tGoldNames = {"My Inlet Pressure", "My Outlet Pressure"};
    for(auto& tName : tNames)
    {
        auto tIndex = &tName - &tNames[0];
        TEST_EQUALITY(tGoldNames[tIndex], tName);
    }

    auto tWeights = Plato::parse_array<Plato::Scalar>("Weights", *tParams);
    std::vector<Plato::Scalar> tGoldWeights = {1.0, -1.0};
    for(auto& tWeight : tWeights)
    {
        auto tIndex = &tWeight - &tWeights[0];
        TEST_EQUALITY(tGoldWeights[tIndex], tWeight);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, WorkStes)
{
    Plato::WorkSets tWorkSets;

    Plato::OrdinalType tNumCells = 1;
    Plato::OrdinalType tNumVelDofs = 12;
    Plato::ScalarMultiVector tVelWS("velocity", tNumCells, tNumVelDofs);
    Plato::blas2::fill(1.0, tVelWS);
    auto tVelPtr = std::make_shared<Plato::MetaData<Plato::ScalarMultiVector>>( tVelWS );
    tWorkSets.set("velocity", tVelPtr);

    Plato::OrdinalType tNumPressDofs = 4;
    Plato::ScalarMultiVector tPressWS("pressure", tNumCells, tNumPressDofs);
    Plato::blas2::fill(2.0, tPressWS);
    auto tPressPtr = std::make_shared<Plato::MetaData<Plato::ScalarMultiVector>>( tPressWS );
    tWorkSets.set("pressure", tPressPtr);

    // TEST VALUES
    tVelWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("velocity"));
    TEST_EQUALITY(tNumCells, tVelWS.extent(0));
    TEST_EQUALITY(tNumVelDofs, tVelWS.extent(1));
    auto tHostVelWS = Kokkos::create_mirror(tVelWS);
    Kokkos::deep_copy(tHostVelWS, tVelWS);
    const Plato::Scalar tTol = 1e-6;
    for(decltype(tNumVelDofs) tIndex = 0; tIndex < tNumVelDofs; tIndex++)
    {
        TEST_FLOATING_EQUALITY(1.0, tHostVelWS(0, tIndex), tTol);
    }

    tPressWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("pressure"));
    TEST_EQUALITY(tNumCells, tPressWS.extent(0));
    TEST_EQUALITY(tNumPressDofs, tPressWS.extent(1));
    auto tHostPressWS = Kokkos::create_mirror(tPressWS);
    Kokkos::deep_copy(tHostPressWS, tPressWS);
    for(decltype(tNumPressDofs) tIndex = 0; tIndex < tNumPressDofs; tIndex++)
    {
        TEST_FLOATING_EQUALITY(2.0, tHostPressWS(0, tIndex), tTol);
    }

    // TEST TAGS
    auto tTags = tWorkSets.tags();
    std::vector<std::string> tGoldTags = {"velocity", "pressure"};
    for(auto& tTag : tTags)
    {
        auto tGoldItr = std::find(tGoldTags.begin(), tGoldTags.end(), tTag);
        if(tGoldItr != tGoldTags.end())
        {
            TEST_EQUALITY(tGoldItr.operator*(), tTag);
        }
        else
        {
            TEST_EQUALITY("failed", tTag);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, LocalOrdinalMaps)
{
    constexpr Plato::OrdinalType tNumSpaceDim = 3;
    using PhysicsT = Plato::MomentumConservation<tNumSpaceDim>;
    auto tMesh = PlatoUtestHelpers::build_3d_box_mesh(1.0, 1.0, 1.0, 1, 1, 1);
    Plato::LocalOrdinalMaps<PhysicsT> tLocalOrdinalMaps(tMesh.operator*());

    auto tNumCells = tMesh->nelems();
    Plato::ScalarArray3D tCoords("coordinates", tNumCells, PhysicsT::mNumNodesPerCell, tNumSpaceDim);
    Plato::ScalarMultiVector tControlOrdinals("control", tNumCells, PhysicsT::mNumNodesPerCell);
    Plato::ScalarMultiVector tScalarFieldOrdinals("scalar field ordinals", tNumCells, PhysicsT::mNumNodesPerCell);
    Plato::ScalarArray3D tVectorFieldOrdinals("vector field ordinals", tNumCells, PhysicsT::mNumNodesPerCell, PhysicsT::mNumMomentumDofsPerNode);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNode = 0; tNode < PhysicsT::mNumNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDim = 0; tDim < PhysicsT::mNumSpatialDims; tDim++)
            {
                tCoords(aCellOrdinal, tNode, tDim) = tLocalOrdinalMaps.mNodeCoordinate(aCellOrdinal, tNode, tDim);
                tVectorFieldOrdinals(aCellOrdinal, tNode, tDim) = tLocalOrdinalMaps.mVectorFieldOrdinalsMap(aCellOrdinal, tNode, tDim);
            }
        }

        for(Plato::OrdinalType tNode = 0; tNode < PhysicsT::mNumNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDim = 0; tDim < PhysicsT::mNumControlDofsPerNode; tDim++)
            {
                tControlOrdinals(aCellOrdinal, tNode) = tLocalOrdinalMaps.mControlOrdinalsMap(aCellOrdinal, tNode, tDim);
                tScalarFieldOrdinals(aCellOrdinal, tNode) = tLocalOrdinalMaps.mScalarFieldOrdinalsMap(aCellOrdinal, tNode, tDim);
            }
        }

    },"test");

    // TEST 3D ARRAYS
    Plato::ScalarArray3D tGoldCoords("coordinates", tNumCells, PhysicsT::mNumNodesPerCell, tNumSpaceDim);
    auto tHostGoldCoords = Kokkos::create_mirror(tGoldCoords);
    tHostGoldCoords(0,0,0) = 0; tHostGoldCoords(0,1,0) = 1; tHostGoldCoords(0,2,0) = 0; tHostGoldCoords(0,3,0) = 1;
    tHostGoldCoords(1,0,0) = 0; tHostGoldCoords(1,1,0) = 0; tHostGoldCoords(1,2,0) = 0; tHostGoldCoords(1,3,0) = 1;
    tHostGoldCoords(2,0,0) = 0; tHostGoldCoords(2,1,0) = 0; tHostGoldCoords(2,2,0) = 0; tHostGoldCoords(2,3,0) = 1;
    tHostGoldCoords(3,0,0) = 0; tHostGoldCoords(3,1,0) = 1; tHostGoldCoords(3,2,0) = 1; tHostGoldCoords(3,3,0) = 0;
    tHostGoldCoords(4,0,0) = 1; tHostGoldCoords(4,1,0) = 1; tHostGoldCoords(4,2,0) = 1; tHostGoldCoords(4,3,0) = 0;
    tHostGoldCoords(5,0,0) = 1; tHostGoldCoords(5,1,0) = 1; tHostGoldCoords(5,2,0) = 1; tHostGoldCoords(5,3,0) = 0;
    tHostGoldCoords(0,0,1) = 0; tHostGoldCoords(0,1,1) = 1; tHostGoldCoords(0,2,1) = 1; tHostGoldCoords(0,3,1) = 1;
    tHostGoldCoords(1,0,1) = 0; tHostGoldCoords(1,1,1) = 1; tHostGoldCoords(1,2,1) = 1; tHostGoldCoords(1,3,1) = 1;
    tHostGoldCoords(2,0,1) = 0; tHostGoldCoords(2,1,1) = 1; tHostGoldCoords(2,2,1) = 0; tHostGoldCoords(2,3,1) = 1;
    tHostGoldCoords(3,0,1) = 0; tHostGoldCoords(3,1,1) = 0; tHostGoldCoords(3,2,1) = 1; tHostGoldCoords(3,3,1) = 0;
    tHostGoldCoords(4,0,1) = 0; tHostGoldCoords(4,1,1) = 0; tHostGoldCoords(4,2,1) = 1; tHostGoldCoords(4,3,1) = 0;
    tHostGoldCoords(5,0,1) = 0; tHostGoldCoords(5,1,1) = 1; tHostGoldCoords(5,2,1) = 1; tHostGoldCoords(5,3,1) = 0;
    tHostGoldCoords(0,0,2) = 0; tHostGoldCoords(0,1,2) = 0; tHostGoldCoords(0,2,2) = 0; tHostGoldCoords(0,3,2) = 1;
    tHostGoldCoords(1,0,2) = 0; tHostGoldCoords(1,1,2) = 0; tHostGoldCoords(1,2,2) = 1; tHostGoldCoords(1,3,2) = 1;
    tHostGoldCoords(2,0,2) = 0; tHostGoldCoords(2,1,2) = 1; tHostGoldCoords(2,2,2) = 1; tHostGoldCoords(2,3,2) = 1;
    tHostGoldCoords(3,0,2) = 0; tHostGoldCoords(3,1,2) = 1; tHostGoldCoords(3,2,2) = 1; tHostGoldCoords(3,3,2) = 1;
    tHostGoldCoords(4,0,2) = 0; tHostGoldCoords(4,1,2) = 1; tHostGoldCoords(4,2,2) = 1; tHostGoldCoords(4,3,2) = 0;
    tHostGoldCoords(5,0,2) = 0; tHostGoldCoords(5,1,2) = 1; tHostGoldCoords(5,2,2) = 0; tHostGoldCoords(5,3,2) = 0;
    auto tHostCoords = Kokkos::create_mirror(tCoords);
    Kokkos::deep_copy(tHostCoords, tCoords);

    Plato::ScalarArray3D tGoldVectorOrdinals("vector field", tNumCells, PhysicsT::mNumNodesPerCell, PhysicsT::mNumMomentumDofsPerNode);
    auto tHostGoldVecOrdinals = Kokkos::create_mirror(tGoldVectorOrdinals);
    tHostGoldVecOrdinals(0,0,0) = 0;  tHostGoldVecOrdinals(0,1,0) = 12; tHostGoldVecOrdinals(0,2,0) = 9;  tHostGoldVecOrdinals(0,3,0) = 15;
    tHostGoldVecOrdinals(1,0,0) = 0;  tHostGoldVecOrdinals(1,1,0) = 9;  tHostGoldVecOrdinals(1,2,0) = 6;  tHostGoldVecOrdinals(1,3,0) = 15;
    tHostGoldVecOrdinals(2,0,0) = 0;  tHostGoldVecOrdinals(2,1,0) = 6;  tHostGoldVecOrdinals(2,2,0) = 3;  tHostGoldVecOrdinals(2,3,0) = 15;
    tHostGoldVecOrdinals(3,0,0) = 0;  tHostGoldVecOrdinals(3,1,0) = 18; tHostGoldVecOrdinals(3,2,0) = 15; tHostGoldVecOrdinals(3,3,0) = 3;
    tHostGoldVecOrdinals(4,0,0) = 21; tHostGoldVecOrdinals(4,1,0) = 18; tHostGoldVecOrdinals(4,2,0) = 15; tHostGoldVecOrdinals(4,3,0) = 0;
    tHostGoldVecOrdinals(5,0,0) = 21; tHostGoldVecOrdinals(5,1,0) = 15; tHostGoldVecOrdinals(5,2,0) = 12; tHostGoldVecOrdinals(5,3,0) = 0;
    tHostGoldVecOrdinals(0,0,1) = 1;  tHostGoldVecOrdinals(0,1,1) = 13; tHostGoldVecOrdinals(0,2,1) = 10; tHostGoldVecOrdinals(0,3,1) = 16;
    tHostGoldVecOrdinals(1,0,1) = 1;  tHostGoldVecOrdinals(1,1,1) = 10; tHostGoldVecOrdinals(1,2,1) = 7;  tHostGoldVecOrdinals(1,3,1) = 16;
    tHostGoldVecOrdinals(2,0,1) = 1;  tHostGoldVecOrdinals(2,1,1) = 7;  tHostGoldVecOrdinals(2,2,1) = 4;  tHostGoldVecOrdinals(2,3,1) = 16;
    tHostGoldVecOrdinals(3,0,1) = 1;  tHostGoldVecOrdinals(3,1,1) = 19; tHostGoldVecOrdinals(3,2,1) = 16; tHostGoldVecOrdinals(3,3,1) = 4;
    tHostGoldVecOrdinals(4,0,1) = 22; tHostGoldVecOrdinals(4,1,1) = 19; tHostGoldVecOrdinals(4,2,1) = 16; tHostGoldVecOrdinals(4,3,1) = 1;
    tHostGoldVecOrdinals(5,0,1) = 22; tHostGoldVecOrdinals(5,1,1) = 16; tHostGoldVecOrdinals(5,2,1) = 13; tHostGoldVecOrdinals(5,3,1) = 1;
    tHostGoldVecOrdinals(0,0,2) = 2;  tHostGoldVecOrdinals(0,1,2) = 14; tHostGoldVecOrdinals(0,2,2) = 11; tHostGoldVecOrdinals(0,3,2) = 17;
    tHostGoldVecOrdinals(1,0,2) = 2;  tHostGoldVecOrdinals(1,1,2) = 11; tHostGoldVecOrdinals(1,2,2) = 8;  tHostGoldVecOrdinals(1,3,2) = 17;
    tHostGoldVecOrdinals(2,0,2) = 2;  tHostGoldVecOrdinals(2,1,2) = 8;  tHostGoldVecOrdinals(2,2,2) = 5;  tHostGoldVecOrdinals(2,3,2) = 17;
    tHostGoldVecOrdinals(3,0,2) = 2;  tHostGoldVecOrdinals(3,1,2) = 20; tHostGoldVecOrdinals(3,2,2) = 17; tHostGoldVecOrdinals(3,3,2) = 5;
    tHostGoldVecOrdinals(4,0,2) = 23; tHostGoldVecOrdinals(4,1,2) = 20; tHostGoldVecOrdinals(4,2,2) = 17; tHostGoldVecOrdinals(4,3,2) = 2;
    tHostGoldVecOrdinals(5,0,2) = 23; tHostGoldVecOrdinals(5,1,2) = 17; tHostGoldVecOrdinals(5,2,2) = 14; tHostGoldVecOrdinals(5,3,2) = 2;
    auto tHostVectorFieldOrdinals = Kokkos::create_mirror(tVectorFieldOrdinals);
    Kokkos::deep_copy(tHostVectorFieldOrdinals, tVectorFieldOrdinals);

    auto tTol = 1e-6;
    for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for(Plato::OrdinalType tNode = 0; tNode < PhysicsT::mNumNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDim = 0; tDim < PhysicsT::mNumSpatialDims; tDim++)
            {
                TEST_FLOATING_EQUALITY(tHostGoldCoords(tCell, tNode, tDim), tHostCoords(tCell, tNode, tDim), tTol);
                TEST_FLOATING_EQUALITY(tHostGoldVecOrdinals(tCell, tNode, tDim), tHostVectorFieldOrdinals(tCell, tNode, tDim), tTol);
            }
        }
    }

    // TEST 2D ARRAYS
    Plato::ScalarMultiVector tGoldControlOrdinals("control", tNumCells, PhysicsT::mNumNodesPerCell);
    auto tHostGoldControlOrdinals = Kokkos::create_mirror(tGoldControlOrdinals);
    tHostGoldControlOrdinals(0,0) = 0; tHostGoldControlOrdinals(0,1) = 4; tHostGoldControlOrdinals(0,2) = 3; tHostGoldControlOrdinals(0,3) = 5;
    tHostGoldControlOrdinals(1,0) = 0; tHostGoldControlOrdinals(1,1) = 3; tHostGoldControlOrdinals(1,2) = 2; tHostGoldControlOrdinals(1,3) = 5;
    tHostGoldControlOrdinals(2,0) = 0; tHostGoldControlOrdinals(2,1) = 2; tHostGoldControlOrdinals(2,2) = 1; tHostGoldControlOrdinals(2,3) = 5;
    tHostGoldControlOrdinals(3,0) = 0; tHostGoldControlOrdinals(3,1) = 6; tHostGoldControlOrdinals(3,2) = 5; tHostGoldControlOrdinals(3,3) = 1;
    tHostGoldControlOrdinals(4,0) = 7; tHostGoldControlOrdinals(4,1) = 6; tHostGoldControlOrdinals(4,2) = 5; tHostGoldControlOrdinals(4,3) = 0;
    tHostGoldControlOrdinals(5,0) = 7; tHostGoldControlOrdinals(5,1) = 5; tHostGoldControlOrdinals(5,2) = 4; tHostGoldControlOrdinals(5,3) = 0;
    auto tHostControlOrdinals = Kokkos::create_mirror(tControlOrdinals);
    Kokkos::deep_copy(tHostControlOrdinals, tControlOrdinals);

    Plato::ScalarMultiVector tGoldScalarOrdinals("scalar field", tNumCells, PhysicsT::mNumNodesPerCell);
    auto tHostGoldScalarOrdinals = Kokkos::create_mirror(tGoldScalarOrdinals);
    tHostGoldScalarOrdinals(0,0) = 0; tHostGoldScalarOrdinals(0,1) = 4; tHostGoldScalarOrdinals(0,2) = 3; tHostGoldScalarOrdinals(0,3) = 5;
    tHostGoldScalarOrdinals(1,0) = 0; tHostGoldScalarOrdinals(1,1) = 3; tHostGoldScalarOrdinals(1,2) = 2; tHostGoldScalarOrdinals(1,3) = 5;
    tHostGoldScalarOrdinals(2,0) = 0; tHostGoldScalarOrdinals(2,1) = 2; tHostGoldScalarOrdinals(2,2) = 1; tHostGoldScalarOrdinals(2,3) = 5;
    tHostGoldScalarOrdinals(3,0) = 0; tHostGoldScalarOrdinals(3,1) = 6; tHostGoldScalarOrdinals(3,2) = 5; tHostGoldScalarOrdinals(3,3) = 1;
    tHostGoldScalarOrdinals(4,0) = 7; tHostGoldScalarOrdinals(4,1) = 6; tHostGoldScalarOrdinals(4,2) = 5; tHostGoldScalarOrdinals(4,3) = 0;
    tHostGoldScalarOrdinals(5,0) = 7; tHostGoldScalarOrdinals(5,1) = 5; tHostGoldScalarOrdinals(5,2) = 4; tHostGoldScalarOrdinals(5,3) = 0;
    auto tHostScalarFieldOrdinals = Kokkos::create_mirror(tScalarFieldOrdinals);
    Kokkos::deep_copy(tHostScalarFieldOrdinals, tScalarFieldOrdinals);

    for(Plato::OrdinalType tNode = 0; tNode < PhysicsT::mNumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < PhysicsT::mNumSpatialDims; tDim++)
        {
            TEST_FLOATING_EQUALITY(tHostGoldControlOrdinals(tNode, tDim), tHostControlOrdinals(tNode, tDim), tTol);
            TEST_FLOATING_EQUALITY(tHostGoldScalarOrdinals(tNode, tDim), tHostScalarFieldOrdinals(tNode, tDim), tTol);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IsValidFunction)
{
    // 1. test throw
    TEST_THROW(Plato::is_valid_function("some function"), std::runtime_error);

    // 2. test scalar function
    auto tOutput = Plato::is_valid_function("scalar function");
    TEST_COMPARE(tOutput, ==, "scalar function");

    // 2. test vector function
    tOutput = Plato::is_valid_function("vector function");
    TEST_COMPARE(tOutput, ==, "vector function");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SidesetNames)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
        "<ParameterList  name='Natural Boundary Conditions'>"
        "  <ParameterList  name='Traction Vector Boundary Condition 1'>"
        "    <Parameter  name='Type'     type='string'        value='Uniform'/>"
        "    <Parameter  name='Values'   type='Array(double)' value='{0.0, -3.0e3, 0.0}'/>"
        "    <Parameter  name='Sides'    type='string'        value='ss_1'/>"
        "  </ParameterList>"
        "  <ParameterList  name='Traction Vector Boundary Condition 2'>"
        "    <Parameter  name='Type'     type='string'        value='Uniform'/>"
        "    <Parameter  name='Values'   type='Array(double)' value='{0.0, -3.0e3, 0.0}'/>"
        "    <Parameter  name='Sides'    type='string'        value='ss_2'/>"
        "  </ParameterList>"
        "</ParameterList>"
    );

    auto tBCs = tParams->sublist("Natural Boundary Conditions");
    auto tOutput = Plato::sideset_names(tBCs);

    std::vector<std::string> tGold = {"ss_1", "ss_2"};
    for(auto& tName : tOutput)
    {
        auto tIndex = &tName - &tOutput[0];
        TEST_COMPARE(tName, ==, tGold[tIndex]);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ParseDimensionlessProperty)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
        "<ParameterList  name='Plato Problem'>"
        "  <ParameterList  name='Dimensionless Properties'>"
        "    <Parameter  name='Prandtl'   type='double'        value='2.1'/>"
        "    <Parameter  name='Grashof'   type='Array(double)' value='{0.0, 1.5, 0.0}'/>"
        "    <Parameter  name='Darcy'     type='double'        value='2.2'/>"
        "  </ParameterList>"
        "</ParameterList>"
    );

    // Prandtl #
    auto tScalarOutput = Plato::parse_parameter<Plato::Scalar>("Prandtl", "Dimensionless Properties", tParams.operator*());
    auto tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tScalarOutput, 2.1, tTolerance);

    // Darcy #
    tScalarOutput = Plato::parse_parameter<Plato::Scalar>("Darcy", "Dimensionless Properties", tParams.operator*());
    TEST_FLOATING_EQUALITY(tScalarOutput, 2.2, tTolerance);

    // Grashof #
    auto tArrayOutput = Plato::parse_parameter<Teuchos::Array<Plato::Scalar>>("Grashof", "Dimensionless Properties", tParams.operator*());
    TEST_EQUALITY(3, tArrayOutput.size());
    TEST_FLOATING_EQUALITY(tArrayOutput[0], 0.0, tTolerance);
    TEST_FLOATING_EQUALITY(tArrayOutput[1], 1.5, tTolerance);
    TEST_FLOATING_EQUALITY(tArrayOutput[2], 0.0, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SolutionsStruct)
{
    Plato::Solutions tSolution;
    constexpr Plato::OrdinalType tNumTimeSteps = 2;

    // set velocity
    constexpr Plato::OrdinalType tNumVelDofs = 12;
    Plato::ScalarMultiVector tGoldVel("velocity", tNumTimeSteps, tNumVelDofs);
    auto tHostGoldVel = Kokkos::create_mirror(tGoldVel);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumVelDofs; tDof++)
        {
            tHostGoldVel(tStep, tDof) = (tStep * tNumTimeSteps) + tDof;
        }
    }
    Kokkos::deep_copy(tGoldVel, tHostGoldVel);
    tSolution.set("velocity", tGoldVel);

    // set pressure
    constexpr Plato::OrdinalType tNumPressDofs = 6;
    Plato::ScalarMultiVector tGoldPress("pressure", tNumTimeSteps, tNumPressDofs);
    auto tHostGoldPress = Kokkos::create_mirror(tGoldPress);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumPressDofs; tDof++)
        {
            tHostGoldPress(tStep, tDof) = (tStep * tNumTimeSteps) + tDof;
        }
    }
    Kokkos::deep_copy(tGoldPress, tHostGoldPress);
    tSolution.set("pressure", tGoldPress);

    // set temperature
    constexpr Plato::OrdinalType tNumTempDofs = 6;
    Plato::ScalarMultiVector tGoldTemp("temperature", tNumTimeSteps, tNumTempDofs);
    auto tHostGoldTemp = Kokkos::create_mirror(tGoldTemp);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumTempDofs; tDof++)
        {
            tHostGoldTemp(tStep, tDof) = (tStep * tNumTimeSteps) + tDof;
        }
    }
    Kokkos::deep_copy(tGoldTemp, tHostGoldTemp);
    tSolution.set("temperature", tGoldTemp);

    // ********** test velocity **********
    auto tTolerance = 1e-6;
    auto tVel   = tSolution.get("velocity");
    auto tHostVel = Kokkos::create_mirror(tVel);
    Kokkos::deep_copy(tHostVel, tVel);
    tHostGoldVel  = Kokkos::create_mirror(tGoldVel);
    Kokkos::deep_copy(tHostGoldVel, tGoldVel);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumVelDofs; tDof++)
        {
            TEST_FLOATING_EQUALITY(tHostGoldVel(tStep, tDof), tHostVel(tStep, tDof), tTolerance);
        }
    }

    // ********** test pressure **********
    auto tPress = tSolution.get("pressure");
    auto tHostPress = Kokkos::create_mirror(tPress);
    Kokkos::deep_copy(tHostPress, tPress);
    tHostGoldPress  = Kokkos::create_mirror(tGoldPress);
    Kokkos::deep_copy(tHostGoldPress, tGoldPress);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumPressDofs; tDof++)
        {
            TEST_FLOATING_EQUALITY(tHostGoldPress(tStep, tDof), tHostPress(tStep, tDof), tTolerance);
        }
    }

    // ********** test temperature **********
    auto tTemp  = tSolution.get("temperature");
    auto tHostTemp = Kokkos::create_mirror(tTemp);
    Kokkos::deep_copy(tHostTemp, tTemp);
    tHostGoldTemp  = Kokkos::create_mirror(tGoldTemp);
    Kokkos::deep_copy(tHostGoldTemp, tGoldTemp);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumTempDofs; tDof++)
        {
            TEST_FLOATING_EQUALITY(tHostGoldTemp(tStep, tDof), tHostTemp(tStep, tDof), tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StatesStruct)
{
    Plato::Variables tStates;
    TEST_COMPARE(tStates.isScalarMapEmpty(), ==, true);
    TEST_COMPARE(tStates.isVectorMapEmpty(), ==, true);

    // set time step index
    tStates.scalar("step", 1);
    TEST_COMPARE(tStates.isScalarMapEmpty(), ==, false);

    // set velocity
    constexpr Plato::OrdinalType tNumVelDofs = 12;
    Plato::ScalarVector tGoldVel("velocity", tNumVelDofs);
    auto tHostGoldVel = Kokkos::create_mirror(tGoldVel);
    for(auto tDof = 0; tDof < tNumVelDofs; tDof++)
    {
        tHostGoldVel(tDof) = tDof;
    }
    Kokkos::deep_copy(tGoldVel, tHostGoldVel);
    tStates.vector("velocity", tGoldVel);
    TEST_COMPARE(tStates.isVectorMapEmpty(), ==, false);

    // set pressure
    constexpr Plato::OrdinalType tNumPressDofs = 6;
    Plato::ScalarVector tGoldPress("pressure", tNumPressDofs);
    auto tHostGoldPress = Kokkos::create_mirror(tGoldPress);
    for(auto tDof = 0; tDof < tNumPressDofs; tDof++)
    {
        tHostGoldPress(tDof) = tDof;
    }
    Kokkos::deep_copy(tGoldPress, tHostGoldPress);
    tStates.vector("pressure", tGoldPress);

    // test empty funciton
    TEST_COMPARE(tStates.defined("velocity"), ==, true);
    TEST_COMPARE(tStates.defined("temperature"), ==, false);

    // test metadata
    auto tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(1.0, tStates.scalar("step"), tTolerance);

    auto tVel  = tStates.vector("velocity");
    auto tHostVel = Kokkos::create_mirror(tVel);
    tHostGoldVel  = Kokkos::create_mirror(tGoldVel);
    for(auto tDof = 0; tDof < tNumVelDofs; tDof++)
    {
        TEST_FLOATING_EQUALITY(tHostGoldVel(tDof), tHostVel(tDof), tTolerance);
    }

    auto tPress  = tStates.vector("pressure");
    auto tHostPress = Kokkos::create_mirror(tPress);
    tHostGoldPress  = Kokkos::create_mirror(tGoldPress);
    for(auto tDof = 0; tDof < tNumPressDofs; tDof++)
    {
        TEST_FLOATING_EQUALITY(tHostGoldPress(tDof), tHostPress(tDof), tTolerance);
    }
}

}
