#ifndef FLUX_DIVERGENCE
#define FLUX_DIVERGENCE

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Flux divergence functor.
  
    Given a thermal flux, compute the flux divergence.
*/
/******************************************************************************/
template<int SpaceDim, int NumDofsPerNode=1, int DofOffset=0>
class FluxDivergence : public Plato::Simplex<SpaceDim>
{
  private:

    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;

  public:

    template<
      typename ForcingScalarType, 
      typename FluxScalarType,
      typename GradientScalarType,
      typename VolumeScalarType>
    DEVICE_TYPE inline void
    operator()( int cellOrdinal,
                Plato::ScalarMultiVectorT< ForcingScalarType > q,
                Plato::ScalarMultiVectorT< FluxScalarType    > tflux,
                Plato::ScalarArray3DT<     GradientScalarType > gradient,
                Plato::ScalarVectorT<VolumeScalarType> cellVolume,
                Plato::Scalar scale = 1.0 ) const {

      // compute flux divergence
      //
      for( int iNode=0; iNode<mNumNodesPerCell; iNode++){
        Plato::OrdinalType localOrdinal = iNode*NumDofsPerNode+DofOffset;
        for(int iDim=0; iDim<SpaceDim; iDim++){
          q(cellOrdinal, localOrdinal) += scale*tflux(cellOrdinal,iDim)*gradient(cellOrdinal,iNode,iDim)*cellVolume(cellOrdinal);
        }
      }
    }
};
// class FluxDivergence

} // namespace Plato
#endif
