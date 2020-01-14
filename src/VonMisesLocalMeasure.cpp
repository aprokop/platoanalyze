/*
 * VonMisesLocalMeasure.cpp
 *
 */

#include "plato/VonMisesLocalMeasure.hpp"


#ifdef PLATO_1D
PLATO_EXPL_DEF2(Plato::VonMisesLocalMeasure, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEF2(Plato::VonMisesLocalMeasure, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEF2(Plato::VonMisesLocalMeasure, Plato::SimplexMechanics, 3)
#endif