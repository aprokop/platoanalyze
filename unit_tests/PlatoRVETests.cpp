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

/******************************************************************************//**
 *
  \brief "zip" values of two vectors together into a vector of pairs
 *
 **********************************************************************************/
template <typename A, typename B>
void zip(
    const std::vector<A> &a,
    const std::vector<B> &b,
    std::vector<std::pair<A,B>> &zipped)
{
    for(size_t i=0; i<a.size(); ++i)
    {
        zipped.push_back(std::make_pair(a[i], b[i]));
    }
}

/******************************************************************************//**
 *
  \brief "unzip" a vector of pairs back into two separate vectors
 *
 **********************************************************************************/
template <typename A, typename B>
void unzip(
    const std::vector<std::pair<A, B>> &zipped,
    std::vector<A> &a,
    std::vector<B> &b)
{
    for(size_t i=0; i<a.size(); i++)
    {
        a[i] = zipped[i].first;
        b[i] = zipped[i].second;
    }
}

/******************************************************************************//**
 *
  \brief scale std::vector
 *
 **********************************************************************************/
template <typename DataType>
void scalarTimesVector(std::vector<DataType> &v, const DataType k)
{
    std::transform(v.begin(), v.end(), v.begin(), [k](DataType &c){ return c*k; });
}

/******************************************************************************//**
 *
  \brief add two std::vectors
 *
 **********************************************************************************/
template <typename DataType>
void vectorPlusVector(std::vector<DataType> &v1, const std::vector<DataType> &v2)
{
    std::transform (v1.begin(), v1.end(), v2.begin(), v1.begin(), std::plus<DataType>());
}

/******************************************************************************//**
 *
  \brief Set std::vector data from data on device \n
    overloaded for Omega_h::LOs
 *
 **********************************************************************************/
std::vector<Plato::OrdinalType> setVectorFromDeviceData( const Omega_h::LOs & aInput )
{
    auto tRange = aInput.size();
    Plato::LocalOrdinalVector tView("view of device data",tRange);

    Kokkos::parallel_for("set view data", tRange, LAMBDA_EXPRESSION(const int & aIndex)
    {
        tView(aIndex) = aInput[aIndex]; 
    });

    auto tView_host = Kokkos::create_mirror_view(tView);
    Kokkos::deep_copy(tView_host, tView); 

    std::vector<Plato::OrdinalType> tVector(tRange);
    tVector.assign(tView_host.data(),tView_host.data()+tView_host.extent(0));
    return tVector;
}

/******************************************************************************//**
 *
  \brief Set std::vector data from data on device \n
    overloaded for kokkos views
 *
 **********************************************************************************/
template <typename DataType>
std::vector<DataType> setVectorFromDeviceData( const Plato::ScalarVectorT<DataType> & aInput )
{
    auto tInput_host = Kokkos::create_mirror_view(aInput);
    Kokkos::deep_copy(tInput_host, aInput); 

    std::vector<DataType> tVector(aInput.size());
    tVector.assign(tInput_host.data(),tInput_host.data()+tInput_host.extent(0));
    return tVector;
}

/******************************************************************************//**
 *
  \brief Set std::vector of coordinate data \n
    overloaded for Omega_h::LOs
 *
 **********************************************************************************/
std::vector<Plato::Scalar> getCoordinateVector(const Omega_h::Mesh& aMesh, const Omega_h::LOs& aNodeOrdinals, const int aCoordOffset)
{
    auto tSpaceDim = aMesh.dim();
    assert(tSpaceDim == static_cast<Omega_h::Int>(3));
    auto tCoords = aMesh.coords();
    auto tNumNodes = aNodeOrdinals.size();

    Plato::ScalarVector tView("coordinates", tNumNodes);
    // the following will only work well in serial mode on host -- this is just for basic sanity checking
    Kokkos::parallel_for("store coordinates", tNumNodes, LAMBDA_EXPRESSION(const int & aNodeIndex)
    {
        auto tVertexNumber = aNodeOrdinals[aNodeIndex];
        auto tEntryOffset = tVertexNumber * tSpaceDim;
        auto tCoordinate = tCoords[tEntryOffset + aCoordOffset];
        tView(aNodeIndex) = tCoordinate;
    });

    auto tVector = setVectorFromDeviceData(tView);

    return tVector;
}

/******************************************************************************//**
 *
  \brief Set std::vector of coordinate data \n
    overloaded for std::vector
 *
 **********************************************************************************/
std::vector<Plato::Scalar> getCoordinateVector(const Omega_h::Mesh& aMesh, const std::vector<Plato::OrdinalType>& aNodeOrdinals, const int aCoordOffset)
{
    auto tSpaceDim = aMesh.dim();
    assert(tSpaceDim == static_cast<Omega_h::Int>(3));
    auto tCoords = aMesh.coords();
    auto tNumNodes = aNodeOrdinals.size();

    Plato::ScalarVector tCoordsView("coordinates", tCoords.size());
    // the following will only work well in serial mode on host -- this is just for basic sanity checking
    Kokkos::parallel_for("store coordinates", tCoords.size(), LAMBDA_EXPRESSION(const int & aCoordIndex)
    {
        tCoordsView(aCoordIndex) = tCoords[aCoordIndex];
    });

    auto tCoordsVector = setVectorFromDeviceData(tCoordsView);

    std::vector<Plato::Scalar> tVector(tNumNodes);
    for( Plato::OrdinalType iOrdinal=0; iOrdinal < tNumNodes; iOrdinal++)
    {
        auto tVertexNumber = aNodeOrdinals[iOrdinal];
        auto tEntryOffset = tVertexNumber * tSpaceDim;
        auto tCoordinate = tCoordsVector[tEntryOffset + aCoordOffset];
        tVector[iOrdinal] = tCoordinate;
    };

    return tVector;
}

/******************************************************************************//**
 *
  \brief Sort node ordinals first along one direction and then along a second
 *
 **********************************************************************************/
void sortOrdinalsByCoordinates(std::vector<Plato::OrdinalType> & aOrdinals,
                               std::vector<Plato::Scalar> & aCoordsDim1,
                               const std::vector<Plato::Scalar> & aCoordsDim2)
{
    auto tNumNodes = aOrdinals.size();
    assert(aCoordsDim1.size() == tNumNodes);
    assert(aCoordsDim2.size() == tNumNodes);

    auto tMaxDim1 = *std::max_element(aCoordsDim1.begin(), aCoordsDim1.end());
    auto tMinDim1 = *std::min_element(aCoordsDim1.begin(), aCoordsDim1.end());
    auto tRangeDim1 = tMaxDim1 - tMinDim1;

    tRangeDim1 *= 10.0;

    std::vector<Plato::Scalar> tTransformedCoords = aCoordsDim1;
    scalarTimesVector(tTransformedCoords, tRangeDim1);
    vectorPlusVector(tTransformedCoords, aCoordsDim2);

    std::vector<std::pair<Plato::OrdinalType,Plato::Scalar>> tZipped;
    zip(aOrdinals, tTransformedCoords, tZipped);

    std::sort(std::begin(tZipped), std::end(tZipped),
    [&](const auto& a, const auto& b)
    {
        return a.second > b.second;
    });

    unzip(tZipped, aOrdinals, tTransformedCoords);
}

/******************************************************************************//**
 *
  \brief find ordinals of face nodes at edge corresponding to max or min of given dimension
 *
 **********************************************************************************/
std::vector<Plato::OrdinalType> getLowerDimensionalOrdinals(const std::string aTag,
                                                            const std::vector<Plato::OrdinalType> & aOrdinals,
                                                            const std::vector<Plato::Scalar> & aCoords)
{
    constexpr Plato::Scalar cThreshold = 1e-8;

    auto tNumNodes = aOrdinals.size();
    assert(aCoords.size() == tNumNodes);

    Plato::Scalar tTarget;
    if ( aTag == "max" )
        tTarget = *std::max_element(aCoords.begin(), aCoords.end()); 
    else if (aTag == "min" )
        tTarget = *std::min_element(aCoords.begin(), aCoords.end()); 
    else
        THROWERR(std::string("IN getLowerDimensionalOrdinals: INVALID TAG. FIRST ARGUMENT MUST BE max OR min"))

    std::vector<Plato::OrdinalType> tEdgeOrdinals;
    for ( size_t iOrdinal = 0; iOrdinal < tNumNodes; iOrdinal++)
    {
        auto tTol = aCoords[iOrdinal] - tTarget;
        if ( tTol*tTol < cThreshold*cThreshold )
            tEdgeOrdinals.push_back(aOrdinals[iOrdinal]);
    }

    if ( tEdgeOrdinals.size() > 0 )
        return tEdgeOrdinals;
    else
        THROWERR(std::string("IN getLowerDimensionalOrdinals: NO ORDINALS FOUND"))
}

/******************************************************************************//**
 *
  \brief Print std::vector
 *
 **********************************************************************************/
template <typename DataType>
void printVector(const std::vector<DataType> & aVector)
{
    std::cout << "\n Vector length " << aVector.size() << std::endl;
    std::cout << "\n Vector contents \n";
    for (size_t iOrdinal = 0; iOrdinal < aVector.size(); iOrdinal++)
        std::cout << aVector[iOrdinal] << std::endl;
}

} // end namespace PlatoDevel

/******************************************************************************/
/*!
  \brief test scaling of std::vector
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( RVETests, ScaleVector )
{
  std::vector<Plato::Scalar> tVector = {1.0,5.2,8.3,100.43,-12.4};
  Plato::Scalar tScale = 45.3;
  PlatoDevel::scalarTimesVector(tVector, tScale);

  std::vector<Plato::Scalar> tGoldValues = {45.3,235.56,375.99,4549.479,-561.72};
  for (size_t iOrdinal = 0; iOrdinal < tGoldValues.size(); iOrdinal++)
      TEST_EQUALITY(tGoldValues[iOrdinal], tVector[iOrdinal]);
}

/******************************************************************************/
/*!
  \brief test addition of two std::vectors
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( RVETests, AddVectors )
{
  std::vector<Plato::Scalar> tVector1 = {1.0,5.2,8.3,100.43,-12.4};
  std::vector<Plato::Scalar> tVector2 = {5.6,9.3,7.2,15.3,102.5};
  PlatoDevel::vectorPlusVector(tVector1, tVector2);

  std::vector<Plato::Scalar> tGoldValues = {6.6,14.5,15.5,115.73,90.1};
  for (size_t iOrdinal = 0; iOrdinal < tGoldValues.size(); iOrdinal++)
      TEST_EQUALITY(tGoldValues[iOrdinal], tVector1[iOrdinal]);
}

/******************************************************************************/
/*!
  \brief test zipping and unzipping of two std::vectors
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( RVETests, ZipVectors )
{
  std::vector<Plato::OrdinalType> tOrdinalVector = {1,5,8,100,2};
  std::vector<Plato::Scalar> tScalarVector = {5.6,9.3,7.2,15.3,102.5};
  std::vector<std::pair<Plato::OrdinalType,Plato::Scalar>> tZipped;
  PlatoDevel::zip(tOrdinalVector, tScalarVector, tZipped);

  for (size_t iOrdinal = 0; iOrdinal < tZipped.size(); iOrdinal++)
      TEST_EQUALITY(tOrdinalVector[iOrdinal], tZipped[iOrdinal].first);

  for (size_t iOrdinal = 0; iOrdinal < tZipped.size(); iOrdinal++)
      TEST_EQUALITY(tScalarVector[iOrdinal], tZipped[iOrdinal].second);

  std::vector<Plato::OrdinalType> tOrdinalVectorOut(tZipped.size());
  std::vector<Plato::Scalar> tScalarVectorOut(tZipped.size());
  PlatoDevel::unzip(tZipped, tOrdinalVectorOut, tScalarVectorOut);

  for (size_t iOrdinal = 0; iOrdinal < tOrdinalVector.size(); iOrdinal++)
      TEST_EQUALITY(tOrdinalVector[iOrdinal], tOrdinalVectorOut[iOrdinal]);

  for (size_t iOrdinal = 0; iOrdinal < tScalarVector.size(); iOrdinal++)
      TEST_EQUALITY(tScalarVector[iOrdinal], tScalarVectorOut[iOrdinal]);

}

/******************************************************************************/
/*!
  \brief test sorting of x0 face
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( RVETests, SortX0Face )
{
  constexpr int meshWidth=4;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  auto tMeshObject = *tMesh;
  auto tMarks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 12 /* class id */);
  auto tOrdinals = Omega_h::collect_marked(tMarks);

  auto tOrdinalVector = PlatoDevel::setVectorFromDeviceData(tOrdinals);
  auto tCoordVectorY = PlatoDevel::getCoordinateVector(tMeshObject,tOrdinals,1);
  auto tCoordVectorZ = PlatoDevel::getCoordinateVector(tMeshObject,tOrdinals,2);

  PlatoDevel::sortOrdinalsByCoordinates(tOrdinalVector,tCoordVectorY,tCoordVectorZ);

  std::vector<Plato::OrdinalType> tGoldOrdinals = {25,26,27,40,46,24,23,28,41,47,22,21,20,39,48,18,17,19,6,1,10,9,8,7,0};
  TEST_EQUALITY(tGoldOrdinals.size(), tOrdinalVector.size());
  for (size_t iOrdinal = 0; iOrdinal < tGoldOrdinals.size(); iOrdinal++)
      TEST_EQUALITY(tGoldOrdinals[iOrdinal], tOrdinalVector[iOrdinal]);
}

/******************************************************************************/
/*!
  \brief test finding x0-face z-edges and corners
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( RVETests, FindX0FaceZEdgesAndCorners )
{
  // create test mesh
  //
  constexpr int meshWidth=4;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  auto tMeshObject = *tMesh;
  auto tMarks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 12 /* class id */);
  auto tOrdinals = Omega_h::collect_marked(tMarks);

  auto tOrdinalVector = PlatoDevel::setVectorFromDeviceData(tOrdinals);
  auto tCoordVectorY = PlatoDevel::getCoordinateVector(tMeshObject,tOrdinals,1);
  auto tCoordVectorZ = PlatoDevel::getCoordinateVector(tMeshObject,tOrdinals,2);

  auto tEdgeX0Z1Ordinals = PlatoDevel::getLowerDimensionalOrdinals("max",tOrdinalVector,tCoordVectorZ);
  std::vector<Plato::OrdinalType> tGoldOrdinals = {10,18,22,24,25};
  TEST_EQUALITY(tGoldOrdinals.size(), tEdgeX0Z1Ordinals.size());
  for (size_t iOrdinal = 0; iOrdinal < tGoldOrdinals.size(); iOrdinal++)
      TEST_EQUALITY(tGoldOrdinals[iOrdinal], tEdgeX0Z1Ordinals[iOrdinal]);

  auto tEdgeX0Z0Ordinals = PlatoDevel::getLowerDimensionalOrdinals("min",tOrdinalVector,tCoordVectorZ);
  tGoldOrdinals = {0,1,46,47,48};
  TEST_EQUALITY(tGoldOrdinals.size(), tEdgeX0Z0Ordinals.size());
  for (size_t iOrdinal = 0; iOrdinal < tGoldOrdinals.size(); iOrdinal++)
      TEST_EQUALITY(tGoldOrdinals[iOrdinal], tEdgeX0Z0Ordinals[iOrdinal]);

  auto tX0Z0EdgeCoordVectorY = PlatoDevel::getCoordinateVector(tMeshObject,tEdgeX0Z0Ordinals,1);
  auto tX0Z1EdgeCoordVectorY = PlatoDevel::getCoordinateVector(tMeshObject,tEdgeX0Z1Ordinals,1);

  auto tCornerX0Z0Y0Ordinal = PlatoDevel::getLowerDimensionalOrdinals("min",tEdgeX0Z0Ordinals,tX0Z0EdgeCoordVectorY);
  Plato::OrdinalType tGoldOrdinal = 0;
  TEST_EQUALITY(1u, tCornerX0Z0Y0Ordinal.size());
  TEST_EQUALITY(tGoldOrdinal, tCornerX0Z0Y0Ordinal[0]);

  auto tCornerX0Z0Y1Ordinal = PlatoDevel::getLowerDimensionalOrdinals("max",tEdgeX0Z0Ordinals,tX0Z0EdgeCoordVectorY);
  tGoldOrdinal = 46;
  TEST_EQUALITY(1u, tCornerX0Z0Y1Ordinal.size());
  TEST_EQUALITY(tGoldOrdinal, tCornerX0Z0Y1Ordinal[0]);

  auto tCornerX0Z1Y0Ordinal = PlatoDevel::getLowerDimensionalOrdinals("min",tEdgeX0Z1Ordinals,tX0Z1EdgeCoordVectorY);
  tGoldOrdinal = 10;
  TEST_EQUALITY(1u, tCornerX0Z1Y0Ordinal.size());
  TEST_EQUALITY(tGoldOrdinal, tCornerX0Z1Y0Ordinal[0]);

  auto tCornerX0Z1Y1Ordinal = PlatoDevel::getLowerDimensionalOrdinals("max",tEdgeX0Z1Ordinals,tX0Z1EdgeCoordVectorY);
  tGoldOrdinal = 25;
  TEST_EQUALITY(1u, tCornerX0Z1Y1Ordinal.size());
  TEST_EQUALITY(tGoldOrdinal, tCornerX0Z1Y1Ordinal[0]);
}

/******************************************************************************/
/*!
  \brief storing faces for face nodes
*/
/******************************************************************************/

  /* std::sort(tVectorX0.begin(), tVectorX0.end()); // set_intersection assumes sorted */
  /* std::sort(tVectorY0.begin(), tVectorY0.end()); // set_intersection assumes sorted */
  /* std::set_intersection(tVectorX0.begin(),tVectorX0.end(),tVectorY0.begin(),tVectorY0.end(),std::back_inserter(tVectorEdge14)); */

/* TEUCHOS_UNIT_TEST( RVETests, Store faces ) */
/* { */
  /* auto tMeshObject = *tMesh; */

  /* std::cout << '\n' << "Face 12 (x0) Nodes:" << std::endl; */
  /* auto tX0Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 12 /1* class id *1/); */
  /* auto tX0Ordinals = Omega_h::collect_marked(tX0Marks); */
  /* PlatoUtestHelpers::print_ordinals(tX0Ordinals); */
  /* PlatoUtestHelpers::print_3d_coords(*tMesh,tX0Ordinals); */

  /* std::cout << '\n' << "Face 14 (x1) Nodes:" << std::endl; */
  /* auto tX1Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 14 /1* class id *1/); */
  /* auto tX1Ordinals = Omega_h::collect_marked(tX1Marks); */
  /* PlatoUtestHelpers::print_ordinals(tX1Ordinals); */
  /* PlatoUtestHelpers::print_3d_coords(*tMesh,tX1Ordinals); */

  /* std::cout << '\n' << "Face 10 (y0) Nodes:" << std::endl; */
  /* auto tY0Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 10 /1* class id *1/); */
  /* auto tY0Ordinals = Omega_h::collect_marked(tY0Marks); */
  /* PlatoUtestHelpers::print_ordinals(tY0Ordinals); */
  /* PlatoUtestHelpers::print_3d_coords(*tMesh,tY0Ordinals); */

  /* std::cout << '\n' << "Face 16 (y1) Nodes:" << std::endl; */
  /* auto tY1Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 16 /1* class id *1/); */
  /* auto tY1Ordinals = Omega_h::collect_marked(tY1Marks); */
  /* PlatoUtestHelpers::print_ordinals(tY1Ordinals); */
  /* PlatoUtestHelpers::print_3d_coords(*tMesh,tY1Ordinals); */

  /* std::cout << '\n' << "Face 4 (z0) Nodes:" << std::endl; */
  /* auto tZ0Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 4 /1* class id *1/); */
  /* auto tZ0Ordinals = Omega_h::collect_marked(tZ0Marks); */
  /* PlatoUtestHelpers::print_ordinals(tZ0Ordinals); */
  /* PlatoUtestHelpers::print_3d_coords(*tMesh,tZ0Ordinals); */

  /* std::cout << '\n' << "Face 22 (z1) Nodes:" << std::endl; */
  /* auto tZ1Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 22 /1* class id *1/); */
  /* auto tZ1Ordinals = Omega_h::collect_marked(tZ1Marks); */
  /* PlatoUtestHelpers::print_ordinals(tZ1Ordinals); */
  /* PlatoUtestHelpers::print_3d_coords(*tMesh,tZ1Ordinals); */
/* } */

