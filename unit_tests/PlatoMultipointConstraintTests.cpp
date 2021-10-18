#include "PlatoTestHelpers.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "Mechanics.hpp"
#include "EssentialBCs.hpp"
#include "elliptic/VectorFunction.hpp"
#include "ApplyConstraints.hpp"
#include "SimplexMechanics.hpp"
#include "LinearElasticMaterial.hpp"
#include "alg/PlatoSolverFactory.hpp"
#include "MultipointConstraints.hpp"

#include "PlatoStaticsTypes.hpp"
#include "PlatoMathHelpers.hpp"

#include "SpatialModel.hpp"

#ifdef HAVE_AMGX
#include <alg/AmgXSparseLinearProblem.hpp>
#endif


#include <fenv.h>
#include <memory>
#include <typeinfo>
#include <vector>

namespace PlatoDevel {

/******************************************************************************/
/*! 
  \brief Set Kokkos::View data from std::vector
*/
/******************************************************************************/
template <typename DataType>
void setViewFromVector( Plato::ScalarVectorT<DataType> aView, std::vector<DataType> aVector)
{
  Kokkos::View<DataType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tHostView(aVector.data(),aVector.size());
  Kokkos::deep_copy(aView, tHostView);
}

/******************************************************************************/
/*! 
  \brief Set matrix data from provided views
*/
/******************************************************************************/
void setMatrixData(
  Teuchos::RCP<Plato::CrsMatrixType> aMatrix,
  std::vector<Plato::OrdinalType>    aRowMap,
  std::vector<Plato::OrdinalType>    aColMap,
  std::vector<Plato::Scalar>         aValues )
{
  Plato::ScalarVectorT<Plato::OrdinalType> tRowMap("row map", aRowMap.size());
  setViewFromVector(tRowMap, aRowMap);
  aMatrix->setRowMap(tRowMap);

  Plato::ScalarVectorT<Plato::OrdinalType> tColMap("col map", aColMap.size());
  setViewFromVector(tColMap, aColMap);
  aMatrix->setColumnIndices(tColMap);

  Plato::ScalarVectorT<Plato::Scalar> tValues("values", aValues.size());
  setViewFromVector(tValues, aValues);
  aMatrix->setEntries(tValues);
}

// set CRS matrix from full 2D vector
void fromFull( Teuchos::RCP<Plato::CrsMatrixType>            aOutMatrix,
          const std::vector<std::vector<Plato::Scalar>> aInMatrix )
{
    using Plato::OrdinalType;
    using Plato::Scalar;

    if( aOutMatrix->numRows() != aInMatrix.size()    ) { THROWERR("matrices have incompatible shapes"); }
    if( aOutMatrix->numCols() != aInMatrix[0].size() ) { THROWERR("matrices have incompatible shapes"); }

    auto tNumRowsPerBlock = aOutMatrix->numRowsPerBlock();
    auto tNumColsPerBlock = aOutMatrix->numColsPerBlock();
    auto tNumBlockRows = aOutMatrix->numRows() / tNumRowsPerBlock;
    auto tNumBlockCols = aOutMatrix->numCols() / tNumColsPerBlock;

    std::vector<OrdinalType> tBlockRowMap(tNumBlockRows+1);;

    tBlockRowMap[0] = 0;
    std::vector<OrdinalType> tColumnIndices;
    std::vector<Scalar> tBlockEntries;
    for( OrdinalType iBlockRowIndex=0; iBlockRowIndex<tNumBlockRows; iBlockRowIndex++)
    {
        for( OrdinalType iBlockColIndex=0; iBlockColIndex<tNumBlockCols; iBlockColIndex++)
        {
             bool blockIsNonZero = false;
             std::vector<Scalar> tLocalEntries;
             for( OrdinalType iLocalBlockRowIndex=0; iLocalBlockRowIndex<tNumRowsPerBlock; iLocalBlockRowIndex++)
             {
                 for( OrdinalType iLocalBlockColIndex=0; iLocalBlockColIndex<tNumColsPerBlock; iLocalBlockColIndex++)
                 {
                      auto tMatrixRow = iBlockRowIndex * tNumRowsPerBlock + iLocalBlockRowIndex;
                      auto tMatrixCol = iBlockColIndex * tNumColsPerBlock + iLocalBlockColIndex;
                      tLocalEntries.push_back( aInMatrix[tMatrixRow][tMatrixCol] );
                      if( aInMatrix[tMatrixRow][tMatrixCol] != 0.0 ) blockIsNonZero = true;
                 }
             }
             if( blockIsNonZero )
             {
                 tColumnIndices.push_back( iBlockColIndex );
                 tBlockEntries.insert(tBlockEntries.end(), tLocalEntries.begin(), tLocalEntries.end());
             }
        }
        tBlockRowMap[iBlockRowIndex+1] = tColumnIndices.size();
    }

    setMatrixData(aOutMatrix, tBlockRowMap, tColumnIndices, tBlockEntries);
}

// slow dumb matrix matrix multiply
void SlowDumbMatrixMatrixMultiply( 
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo,
            Teuchos::RCP<Plato::CrsMatrixType> & aOutMatrix)
{
    auto tNumOutMatrixRows = aInMatrixOne->numRows();
    auto tNumOutMatrixCols = aInMatrixTwo->numCols();

    if( aInMatrixOne->numCols() != aInMatrixTwo->numRows() ) { THROWERR("input matrices have incompatible shapes"); }

    auto tNumInner = aInMatrixOne->numCols();

    using Plato::Scalar;
    std::vector<std::vector<Scalar>> tFullMatrix(tNumOutMatrixRows,std::vector<Scalar>(tNumOutMatrixCols, 0.0));

    auto F1 = ::PlatoUtestHelpers::toFull(aInMatrixOne);
    auto F2 = ::PlatoUtestHelpers::toFull(aInMatrixTwo);

    for (auto iRow=0; iRow<tNumOutMatrixRows; iRow++)
    {
        for (auto iCol=0; iCol<tNumOutMatrixCols; iCol++)
        {
            tFullMatrix[iRow][iCol] = 0.0;
            for (auto iK=0; iK<tNumInner; iK++)
            {
                tFullMatrix[iRow][iCol] += F1[iRow][iK]*F2[iK][iCol];
            }
        }
    }

    fromFull(aOutMatrix, tFullMatrix);
}


} // end namespace PlatoDevel

template <typename DataType>
bool is_same(
      const Plato::ScalarVectorT<DataType> & aView,
      const std::vector<DataType>          & aVec)
 {
    auto tView_host = Kokkos::create_mirror(aView);
    Kokkos::deep_copy(tView_host, aView);

    if( aView.extent(0) != aVec.size() ) return false;

    for (unsigned int i = 0; i < aVec.size(); ++i)
    {
        if(tView_host(i) != aVec[i])
        {
            return false;
        }
    }
    return true;
 }

template <typename DataType>
bool is_same(
      const Plato::ScalarVectorT<DataType> & aViewA,
      const Plato::ScalarVectorT<DataType> & aViewB)
 {
    if( aViewA.extent(0) != aViewB.extent(0) ) return false;
    auto tViewA_host = Kokkos::create_mirror(aViewA);
    Kokkos::deep_copy(tViewA_host, aViewA);
    auto tViewB_host = Kokkos::create_mirror(aViewB);
    Kokkos::deep_copy(tViewB_host, aViewB);
    for (unsigned int i = 0; i < aViewA.extent(0); ++i)
    {
        if(tViewA_host(i) != tViewB_host(i)) return false;
    }
    return true;
 }

bool is_equivalent(
      const Plato::ScalarVectorT<Plato::OrdinalType> & aRowMap,
      const Plato::ScalarVectorT<Plato::OrdinalType> & aColMapA,
      const Plato::ScalarVectorT<Plato::Scalar>      & aValuesA,
      const Plato::ScalarVectorT<Plato::OrdinalType> & aColMapB,
      const Plato::ScalarVectorT<Plato::Scalar>      & aValuesB,
      Plato::Scalar tolerance = 1.0e-14)
 {
    if( aColMapA.extent(0) != aColMapB.extent(0) ) return false;
    if( aValuesA.extent(0) != aValuesB.extent(0) ) return false;

    auto tRowMap  = PlatoUtestHelpers::get(aRowMap);
    auto tColMapA = PlatoUtestHelpers::get(aColMapA);
    auto tValuesA = PlatoUtestHelpers::get(aValuesA);
    auto tColMapB = PlatoUtestHelpers::get(aColMapB);
    auto tValuesB = PlatoUtestHelpers::get(aValuesB);

    Plato::OrdinalType tANumEntriesPerBlock = aValuesA.extent(0) / aColMapA.extent(0);
    Plato::OrdinalType tBNumEntriesPerBlock = aValuesB.extent(0) / aColMapB.extent(0);
    if( tANumEntriesPerBlock != tBNumEntriesPerBlock ) return false;

    auto tNumRows = tRowMap.extent(0)-1;
    for (unsigned int i = 0; i < tNumRows; ++i)
    {
        auto tFrom = tRowMap(i);
        auto tTo = tRowMap(i+1);
        for (auto iColMapEntryA=tFrom; iColMapEntryA<tTo; iColMapEntryA++)
        {
            auto tColumnIndexA = tColMapA(iColMapEntryA);
            for (auto iColMapEntryB=tFrom; iColMapEntryB<tTo; iColMapEntryB++)
            {
                if (tColumnIndexA == tColMapB(iColMapEntryB) )
                {
                    for (auto iBlockEntry=0; iBlockEntry<tANumEntriesPerBlock; iBlockEntry++)
                    {
                        auto tBlockEntryIndexA = iColMapEntryA*tANumEntriesPerBlock+iBlockEntry;
                        auto tBlockEntryIndexB = iColMapEntryB*tBNumEntriesPerBlock+iBlockEntry;
                        Plato::Scalar tSum = fabs(tValuesA(tBlockEntryIndexA)) + fabs(tValuesB(tBlockEntryIndexB));
                        Plato::Scalar tDif = fabs(tValuesA(tBlockEntryIndexA) - tValuesB(tBlockEntryIndexB));
                        Plato::Scalar tRelVal = (tSum != 0.0) ? 2.0*tDif/tSum : 0.0;
                        if (tRelVal > tolerance)
                        {
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
 }

/******************************************************************************/
/*!
  \brief test operations that form condensed systen

  Construct a linear system with tie multipoint constraints.
  Test passes if transformed Jacobian and residual have correct sizes
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( MultipointConstraintTests, BuildCondensedSystem )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // specify parameter input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                    \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>     \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>             \n"
    "  <ParameterList name='Elliptic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>            \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                   \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>          \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>   \n"
    "      <Parameter name='Values' type='Array(double)' value='{1e3, 0}'/>  \n"
    "      <Parameter name='Sides'  type='string'        value='Load'/>      \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Essential Boundary Conditions'>                 \n"
    "    <ParameterList  name='X Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='0'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='1'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Multipoint Constraints'>                        \n"
    "    <ParameterList  name='Node Tie Constraint 1'>                       \n"
    "      <Parameter  name='Type'     type='string'    value='Tie'/>        \n"
    "      <Parameter  name='Child'    type='string'    value='MPC Child'/>  \n"
    "      <Parameter  name='Parent'   type='string'    value='MPC Parent'/> \n"
    "      <Parameter  name='Value'    type='double'    value='4.2'/>        \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "</ParameterList>                                                        \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = tMesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);

  // assign edges for load and boundary conditions
  //
  Plato::DataMap tDataMap;
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Omega_h::Read<Omega_h::I8> tMarksLoad = Omega_h::mark_class_closure(tMesh.get(), Omega_h::EDGE, Omega_h::EDGE, 5 /* class id */);
  tMeshSets[Omega_h::SIDE_SET]["Load"] = Omega_h::collect_marked(tMarksLoad);

  Omega_h::Read<Omega_h::I8> tMarksFix = Omega_h::mark_class_closure(tMesh.get(), Omega_h::VERT, Omega_h::EDGE, 3 /* class id */);
  tMeshSets[Omega_h::NODE_SET]["Fix"] = Omega_h::collect_marked(tMarksFix);
  
  // assign edges for MPCs
  //
  Omega_h::Read<Omega_h::I8> tMarksParent = Omega_h::mark_class_closure(tMesh.get(), Omega_h::VERT, Omega_h::EDGE, 1 /* class id */); // bottom edge is parent
  tMeshSets[Omega_h::NODE_SET]["MPC Parent"] = Omega_h::collect_marked(tMarksParent);

  Omega_h::Read<Omega_h::I8> tMarksChild = Omega_h::mark_class_closure(tMesh.get(), Omega_h::VERT, Omega_h::EDGE, 7 /* class id */); // top edge is child
  tMeshSets[Omega_h::NODE_SET]["MPC Child"] = Omega_h::collect_marked(tMarksChild);
  
  // parse essential BCs
  //
  Plato::LocalOrdinalVector mBcDofs;
  Plato::ScalarVector mBcValues;
  Plato::EssentialBCs<SimplexPhysics>
      tEssentialBoundaryConditions(params->sublist("Essential Boundary Conditions",false), tMeshSets);
  tEssentialBoundaryConditions.get(mBcDofs, mBcValues);

  // create vector function
  //
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *params);
  Plato::Elliptic::VectorFunction<SimplexPhysics>
    vectorFunction(tSpatialModel, tDataMap, *params, params->get<std::string>("PDE Constraint"));

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);
  
  // parse multipoint constraints
  //
  std::shared_ptr<Plato::MultipointConstraints> tMPCs = std::make_shared<Plato::MultipointConstraints>(tSpatialModel, tNumDofsPerNode, params->sublist("Multipoint Constraints", false));
  tMPCs->setupTransform();

  // apply essential BCs
  //
  Plato::applyBlockConstraints<SimplexPhysics::mNumDofsPerNode>(jacobian, residual, mBcDofs, mBcValues);

  // setup transformation
  //
  //Teuchos::RCP<Plato::CrsMatrixType> aA(&jacobian, /*hasOwnership=*/ false);
  Teuchos::RCP<Plato::CrsMatrixType> aA = jacobian;
  const Plato::OrdinalType tNumCondensedNodes = tMPCs->getNumCondensedNodes();
  auto tNumCondensedDofs = tNumCondensedNodes*tNumDofsPerNode;
  
  // get MPC condensation matrices and RHS
  Teuchos::RCP<Plato::CrsMatrixType> tTransformMatrix = tMPCs->getTransformMatrix();
  Teuchos::RCP<Plato::CrsMatrixType> tTransformMatrixTranspose = tMPCs->getTransformMatrixTranspose();
  Plato::ScalarVector tMpcRhs = tMPCs->getRhsVector();
  
  // build condensed matrix
  auto tCondensedALeft = Teuchos::rcp( new Plato::CrsMatrixType(tNumDofs, tNumCondensedDofs, tNumDofsPerNode, tNumDofsPerNode) );
  auto tCondensedA     = Teuchos::rcp( new Plato::CrsMatrixType(tNumCondensedDofs, tNumCondensedDofs, tNumDofsPerNode, tNumDofsPerNode) );
      
  Plato::MatrixMatrixMultiply(aA, tTransformMatrix, tCondensedALeft);
  Plato::MatrixMatrixMultiply(tTransformMatrixTranspose, tCondensedALeft, tCondensedA);

  // build condensed vector
  Plato::ScalarVector tInnerB = residual;
  Plato::blas1::scale(-1.0, tMpcRhs);
  Plato::MatrixTimesVectorPlusVector(aA, tMpcRhs, tInnerB);
  
  Plato::ScalarVector tCondensedB("Condensed RHS Vector", tNumCondensedDofs);
  Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tCondensedB);
  
  Plato::MatrixTimesVectorPlusVector(tTransformMatrixTranspose, tInnerB, tCondensedB);

  // Compute condensed jacobian with slow dumb
  auto tSlowDumbCondensedALeft = Teuchos::rcp( new Plato::CrsMatrixType(tNumDofs, tNumCondensedDofs, tNumDofsPerNode, tNumDofsPerNode) );
  auto tSlowDumbCondensedA     = Teuchos::rcp( new Plato::CrsMatrixType(tNumCondensedDofs, tNumCondensedDofs, tNumDofsPerNode, tNumDofsPerNode) );
  
  PlatoDevel::SlowDumbMatrixMatrixMultiply( aA, tTransformMatrix, tSlowDumbCondensedALeft);
  PlatoDevel::SlowDumbMatrixMatrixMultiply( tTransformMatrixTranspose, tSlowDumbCondensedALeft, tSlowDumbCondensedA);

  // test lenths
  TEST_EQUALITY(tCondensedA->rowMap().size(), tSlowDumbCondensedA->rowMap().size());
}

/******************************************************************************/
/*!
  \brief 2D Elastic problem with Tie multipoint constraints

  Construct a linear system with tie multipoint constraints.
  Test passes if nodal displacements are offset by specified amount in MPC
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( MultipointConstraintTests, Elastic2DTieMPC )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // specify parameter input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                    \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>     \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>             \n"
    "  <ParameterList name='Elliptic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>            \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                   \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>          \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>   \n"
    "      <Parameter name='Values' type='Array(double)' value='{1e3, 0}'/>  \n"
    "      <Parameter name='Sides'  type='string'        value='Load'/>      \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Essential Boundary Conditions'>                 \n"
    "    <ParameterList  name='X Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='0'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='1'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Multipoint Constraints'>                        \n"
    "    <ParameterList  name='Node Tie Constraint 1'>                       \n"
    "      <Parameter  name='Type'     type='string'    value='Tie'/>        \n"
    "      <Parameter  name='Child'    type='string'    value='MPC Child'/>  \n"
    "      <Parameter  name='Parent'   type='string'    value='MPC Parent'/> \n"
    "      <Parameter  name='Value'    type='double'    value='4.2'/>        \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "</ParameterList>                                                        \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = tMesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);

  // assign edges for load and boundary conditions
  //
  Plato::DataMap tDataMap;
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Omega_h::Read<Omega_h::I8> tMarksLoad = Omega_h::mark_class_closure(tMesh.get(), Omega_h::EDGE, Omega_h::EDGE, 5 /* class id */);
  tMeshSets[Omega_h::SIDE_SET]["Load"] = Omega_h::collect_marked(tMarksLoad);

  Omega_h::Read<Omega_h::I8> tMarksFix = Omega_h::mark_class_closure(tMesh.get(), Omega_h::VERT, Omega_h::EDGE, 3 /* class id */);
  tMeshSets[Omega_h::NODE_SET]["Fix"] = Omega_h::collect_marked(tMarksFix);
  
  // assign edges for MPCs
  //
  Omega_h::Read<Omega_h::I8> tMarksParent = Omega_h::mark_class_closure(tMesh.get(), Omega_h::VERT, Omega_h::EDGE, 1 /* class id */); // bottom edge is parent
  tMeshSets[Omega_h::NODE_SET]["MPC Parent"] = Omega_h::collect_marked(tMarksParent);

  Omega_h::Read<Omega_h::I8> tMarksChild = Omega_h::mark_class_closure(tMesh.get(), Omega_h::VERT, Omega_h::EDGE, 7 /* class id */); // top edge is child
  tMeshSets[Omega_h::NODE_SET]["MPC Child"] = Omega_h::collect_marked(tMarksChild);
  
  // parse essential BCs
  //
  Plato::LocalOrdinalVector mBcDofs;
  Plato::ScalarVector mBcValues;
  Plato::EssentialBCs<SimplexPhysics>
      tEssentialBoundaryConditions(params->sublist("Essential Boundary Conditions",false), tMeshSets);
  tEssentialBoundaryConditions.get(mBcDofs, mBcValues);

  // create vector function
  //
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *params);
  Plato::Elliptic::VectorFunction<SimplexPhysics>
    vectorFunction(tSpatialModel, tDataMap, *params, params->get<std::string>("PDE Constraint"));

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);
  
  // parse multipoint constraints
  //
  std::shared_ptr<Plato::MultipointConstraints> tMPCs = std::make_shared<Plato::MultipointConstraints>(tSpatialModel, tNumDofsPerNode, params->sublist("Multipoint Constraints", false));
  tMPCs->setupTransform();
  
  // create solver
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                              \n"
    "  <Parameter name='Solver' type='string' value='AztecOO'/>        \n"
    "  <Parameter name='Display Iterations' type='int' value='0'/>     \n"
    "  <Parameter name='Iterations' type='int' value='50'/>            \n"
    "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "</ParameterList>                                                  \n"
  );
  Plato::SolverFactory tSolverFactory(*tSolverParams);

  auto tSolver = tSolverFactory.create(tMesh->nverts(), tMachine, tNumDofsPerNode, tMPCs);

  // apply essential BCs
  //
  Plato::applyBlockConstraints<SimplexPhysics::mNumDofsPerNode>(jacobian, residual, mBcDofs, mBcValues);

  // solve linear system
  //
  tSolver->solve(*jacobian, state, residual);

  // create mirror view of displacement solution
  //
  Plato::ScalarVector statesView("State",tNumDofs);
  Kokkos::deep_copy(statesView, state);

  auto stateView_host = Kokkos::create_mirror_view(statesView);
  Kokkos::deep_copy(stateView_host, statesView);

  // test difference between constrained nodes
  //
  Plato::OrdinalType checkChildNode = 4;
  Plato::OrdinalType checkParentNode = 7;
  Plato::Scalar      checkValue = 4.2;

  Plato::OrdinalType checkChildDof0 = checkChildNode*tNumDofsPerNode;
  Plato::OrdinalType checkChildDof1 = checkChildNode*tNumDofsPerNode + 1;

  Plato::OrdinalType checkParentDof0 = checkParentNode*tNumDofsPerNode;
  Plato::OrdinalType checkParentDof1 = checkParentNode*tNumDofsPerNode + 1;

  Plato::Scalar checkDifferenceDof0 = stateView_host(checkChildDof0) - stateView_host(checkParentDof0);
  Plato::Scalar checkDifferenceDof1 = stateView_host(checkChildDof1) - stateView_host(checkParentDof1);

  TEST_FLOATING_EQUALITY(checkDifferenceDof0, checkValue, 1.0e-12);
  TEST_FLOATING_EQUALITY(checkDifferenceDof1, checkValue, 1.0e-12);

}

/******************************************************************************/
/*!
  \brief 2D Elastic problem with PBC multipoint constraints

  Construct a linear system with PBC multipoint constraints.
  Test passes if nodal displacements are offset by specified amount in MPC
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( MultipointConstraintTests, Elastic3DPbcMPC )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = tMesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);
  
  auto tMeshObject = *tMesh;

  // specify parameter input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                    \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>     \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>             \n"
    "  <ParameterList name='Elliptic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>            \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                   \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>          \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>   \n"
    "      <Parameter name='Values' type='Array(double)' value='{1e3, 0, 0}'/>  \n"
    "      <Parameter name='Sides'  type='string'        value='Load'/>      \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Essential Boundary Conditions'>                 \n"
    "    <ParameterList  name='X Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='0'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='1'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Z Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='2'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Multipoint Constraints'>                        \n"
    "    <ParameterList  name='PBC Constraint 1'>                            \n"
    "      <Parameter  name='Type'     type='string'    value='PBC'/>        \n"
    "      <Parameter  name='Child'    type='string'    value='MPC Child'/>  \n"
    "      <Parameter  name='Parent'   type='string'    value='Design Volume'/>  \n"
    "      <Parameter  name='Vector'  type='Array(double)' value='{0, 1, 0}'/>  \n"
    "      <Parameter  name='Value'    type='double'    value='0.0'/>        \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "</ParameterList>                                                        \n"
  );

  // assign edges for load and boundary conditions
  //
  Plato::DataMap tDataMap;
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Omega_h::Read<Omega_h::I8> tMarksLoad = Omega_h::mark_class_closure(tMesh.get(), Omega_h::EDGE, Omega_h::FACE, 14 /* class id */);
  tMeshSets[Omega_h::SIDE_SET]["Load"] = Omega_h::collect_marked(tMarksLoad);

  Omega_h::Read<Omega_h::I8> tMarksFix = Omega_h::mark_class_closure(tMesh.get(), Omega_h::VERT, Omega_h::FACE, 12 /* class id */);
  tMeshSets[Omega_h::NODE_SET]["Fix"] = Omega_h::collect_marked(tMarksFix);
  
  // assign faces for MPCs
  //
  Omega_h::Read<Omega_h::I8> tMarksChild = Omega_h::mark_class_closure(tMesh.get(), Omega_h::VERT, Omega_h::FACE, 10 /* class id */); // y0 face is child
  tMeshSets[Omega_h::NODE_SET]["MPC Child"] = Omega_h::collect_marked(tMarksChild);
  
  // parse essential BCs
  //
  Plato::LocalOrdinalVector mBcDofs;
  Plato::ScalarVector mBcValues;
  Plato::EssentialBCs<SimplexPhysics>
      tEssentialBoundaryConditions(params->sublist("Essential Boundary Conditions",false), tMeshSets);
  tEssentialBoundaryConditions.get(mBcDofs, mBcValues);

  // create vector function
  //
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *params);
  Plato::Elliptic::VectorFunction<SimplexPhysics>
    vectorFunction(tSpatialModel, tDataMap, *params, params->get<std::string>("PDE Constraint"));

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);
  
  // parse multipoint constraints
  //
  std::shared_ptr<Plato::MultipointConstraints> tMPCs = std::make_shared<Plato::MultipointConstraints>(tSpatialModel, tNumDofsPerNode, params->sublist("Multipoint Constraints", false));
  tMPCs->setupTransform();
  
  // create solver
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                              \n"
    "  <Parameter name='Solver' type='string' value='AztecOO'/>        \n"
    "  <Parameter name='Display Iterations' type='int' value='0'/>     \n"
    "  <Parameter name='Iterations' type='int' value='50'/>            \n"
    "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "</ParameterList>                                                  \n"
  );
  Plato::SolverFactory tSolverFactory(*tSolverParams);

  auto tSolver = tSolverFactory.create(tMesh->nverts(), tMachine, tNumDofsPerNode, tMPCs);

  // apply essential BCs
  //
  Plato::applyBlockConstraints<SimplexPhysics::mNumDofsPerNode>(jacobian, residual, mBcDofs, mBcValues);

  // solve linear system
  //
  tSolver->solve(*jacobian, state, residual);

  // create mirror view of displacement solution
  //
  Plato::ScalarVector statesView("State",tNumDofs);
  Kokkos::deep_copy(statesView, state);

  auto stateView_host = Kokkos::create_mirror_view(statesView);
  Kokkos::deep_copy(stateView_host, statesView);

  // test difference between constrained nodes
  //
  Plato::OrdinalType checkChildNode = 22;
  Plato::OrdinalType checkParentNode = 15;
  Plato::Scalar      checkValue = 0.0;

  Plato::OrdinalType checkChildDof0 = checkChildNode*tNumDofsPerNode;
  Plato::OrdinalType checkChildDof1 = checkChildNode*tNumDofsPerNode + 1;
  Plato::OrdinalType checkChildDof2 = checkChildNode*tNumDofsPerNode + 2;

  Plato::OrdinalType checkParentDof0 = checkParentNode*tNumDofsPerNode;
  Plato::OrdinalType checkParentDof1 = checkParentNode*tNumDofsPerNode + 1;
  Plato::OrdinalType checkParentDof2 = checkParentNode*tNumDofsPerNode + 2;

  Plato::Scalar checkDifferenceDof0 = stateView_host(checkChildDof0) - stateView_host(checkParentDof0);
  Plato::Scalar checkDifferenceDof1 = stateView_host(checkChildDof1) - stateView_host(checkParentDof1);
  Plato::Scalar checkDifferenceDof2 = stateView_host(checkChildDof2) - stateView_host(checkParentDof2);

  TEST_FLOATING_EQUALITY(checkDifferenceDof0, checkValue, 1.0e-8);
  TEST_FLOATING_EQUALITY(checkDifferenceDof1, checkValue, 1.0e-8);
  TEST_FLOATING_EQUALITY(checkDifferenceDof2, checkValue, 1.0e-8);

}

