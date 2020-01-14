#ifndef ELLIPTIC_VMS_PROBLEM_HPP
#define ELLIPTIC_VMS_PROBLEM_HPP

#include <memory>
#include <sstream>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "NaturalBCs.hpp"
#include "EssentialBCs.hpp"
#include "ImplicitFunctors.hpp"
#include "ApplyConstraints.hpp"

#include "plato/VectorFunctionVMS.hpp"
#include "plato/ScalarFunctionBase.hpp"
#include "plato/ScalarFunctionIncBase.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/PlatoAbstractProblem.hpp"
#include "plato/ParseTools.hpp"
#include "plato/Plato_Solve.hpp"

#include "plato/ScalarFunctionBaseFactory.hpp"
#include "plato/ScalarFunctionIncBaseFactory.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Manage scalar and vector function evaluations
**********************************************************************************/
template<typename SimplexPhysics>
class EllipticVMSProblem: public Plato::AbstractProblem
{
private:

    static constexpr Plato::OrdinalType SpatialDim = SimplexPhysics::mNumSpatialDims; /*!< spatial dimensions */

    // required
    Plato::VectorFunctionVMS<SimplexPhysics> mEqualityConstraint; /*!< equality constraint interface */
    Plato::VectorFunctionVMS<typename SimplexPhysics::ProjectorT> mStateProjection; /*!< projection interface */

    // optional
    std::shared_ptr<const Plato::ScalarFunctionBase>    mConstraint; /*!< constraint constraint interface */
    std::shared_ptr<const Plato::ScalarFunctionIncBase> mObjective;  /*!< objective constraint interface */

    Plato::OrdinalType mNumSteps, mNumNewtonSteps;
    Plato::Scalar mTimeStep;

    Plato::ScalarVector      mResidual;
    Plato::ScalarMultiVector mStates; /*!< state variables */
    Plato::ScalarMultiVector mLambda;
    Teuchos::RCP<Plato::CrsMatrixType> mJacobian; /*!< Jacobian matrix */

    Plato::ScalarVector mProjResidual;
    Plato::ScalarVector mProjPGrad;
    Plato::ScalarVector mProjectState;
    Plato::ScalarVector mEta;
    Teuchos::RCP<Plato::CrsMatrixType> mProjJacobian; /*!< Jacobian matrix */



    Plato::LocalOrdinalVector mBcDofs; /*!< list of degrees of freedom associated with the Dirichlet boundary conditions */
    Plato::ScalarVector mBcValues; /*!< values associated with the Dirichlet boundary conditions */

public:
    /******************************************************************************//**
     * @brief PLATO problem constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    EllipticVMSProblem(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputParams) :
            mEqualityConstraint(aMesh, aMeshSets, mDataMap, aInputParams, aInputParams.get<std::string>("PDE Constraint")),
            mStateProjection(aMesh, aMeshSets, mDataMap, aInputParams, std::string("State Gradient Projection")),
            mNumSteps      (Plato::ParseTools::getSubParam<int>   (aInputParams, "Time Stepping", "Number Time Steps",    1  )),
            mTimeStep      (Plato::ParseTools::getSubParam<Plato::Scalar>(aInputParams, "Time Stepping", "Time Step",     1.0)),
            mNumNewtonSteps(Plato::ParseTools::getSubParam<int>   (aInputParams, "Newton Iteration", "Number Iterations", 2  )),
            mConstraint(nullptr),
            mObjective(nullptr),
            mResidual("MyResidual", mEqualityConstraint.size()),
            mStates("States", mNumSteps, mEqualityConstraint.size()),
            mJacobian(Teuchos::null),
            mProjResidual("MyProjResidual", mStateProjection.size()),
            mProjPGrad("Projected PGrad", mStateProjection.size()),
            mProjectState("Project State", aMesh.nverts()),
            mProjJacobian(Teuchos::null)
    {
        this->initialize(aMesh, aMeshSets, aInputParams);
    }

    /******************************************************************************//**
     * @brief Return number of degrees of freedom in solution.
     * @return Number of degrees of freedom
    **********************************************************************************/
    Plato::OrdinalType getNumSolutionDofs()
    {
        return SimplexPhysics::mNumDofsPerNode;
    }

    /******************************************************************************//**
     * @brief Set state variables
     * @param [in] aState 2D view of state variables
    **********************************************************************************/
    void setState(const Plato::ScalarMultiVector & aState)
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));
        Kokkos::deep_copy(mStates, aState);
    }

    /******************************************************************************//**
     * @brief Return 2D view of state variables
     * @return aState 2D view of state variables
    **********************************************************************************/
    Plato::ScalarMultiVector getState()
    {
        return mStates;
    }

    /******************************************************************************//**
     * @brief Return 2D view of adjoint variables
     * @return 2D view of adjoint variables
    **********************************************************************************/
    Plato::ScalarMultiVector getAdjoint()
    {
        return mLambda;
    }

    /******************************************************************************//**
     * @brief Apply Dirichlet constraints
     * @param [in] aMatrix Compressed Row Storage (CRS) matrix
     * @param [in] aVector 1D view of Right-Hand-Side forces
    **********************************************************************************/
    void applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector)
    {
        if(mJacobian->isBlockMatrix())
        {
            Plato::applyBlockConstraints<SimplexPhysics::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, mBcValues);
        }
        else
        {
            Plato::applyConstraints<SimplexPhysics::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, mBcValues);
        }
    }

    void applyBoundaryLoads(const Plato::ScalarVector & aForce){}

    /******************************************************************************//**
     * @brief Update physics-based parameters within optimization iterations
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 1D container of control variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    { return; }

    /******************************************************************************//**
     * @brief Solve system of equations
     * @param [in] aControl 1D view of control variables
     * @return 2D view of state variables
    **********************************************************************************/
    Plato::ScalarMultiVector solution(const Plato::ScalarVector & aControl)
    {

        Plato::ScalarVector tStateIncrement("State increment", mStates.extent(1));

        // outer loop for load/time steps
        for(Plato::OrdinalType tStepIndex = 1; tStepIndex < mNumSteps; tStepIndex++)
        {
            // compute the projected pressure gradient
            Plato::ScalarVector tState = Kokkos::subview(mStates, tStepIndex, Kokkos::ALL());
            Plato::fill(static_cast<Plato::Scalar>(0.0), tState);
            Plato::fill(static_cast<Plato::Scalar>(0.0), mProjPGrad);
            Plato::fill(static_cast<Plato::Scalar>(0.0), mProjectState);

            // inner loop for load/time steps
            for(Plato::OrdinalType tNewtonIndex = 0; tNewtonIndex < mNumNewtonSteps; tNewtonIndex++)
            {
                mProjResidual = mStateProjection.value      (mProjPGrad, mProjectState, aControl);
                mProjJacobian = mStateProjection.gradient_u (mProjPGrad, mProjectState, aControl);

                Plato::Solve::RowSummed<SimplexPhysics::mNumSpatialDims>(mProjJacobian, mProjPGrad, mProjResidual);

                // compute the state solution
                mResidual = mEqualityConstraint.value      (tState, mProjPGrad, aControl);
                mJacobian = mEqualityConstraint.gradient_u (tState, mProjPGrad, aControl);

                this->applyConstraints(mJacobian, mResidual);

                Plato::Solve::Consistent<SimplexPhysics::mNumDofsPerNode>(mJacobian, tStateIncrement, mResidual);

                // update the state with the new increment
                Plato::update(-1.0, tStateIncrement, 1.0, tState);

                // copy projection state
                Plato::extract<SimplexPhysics::mNumDofsPerNode,
                               SimplexPhysics::ProjectorT::SimplexT::mProjectionDof>(tState, mProjectState);
            }

            mResidual = mEqualityConstraint.value(tState, mProjPGrad, aControl);

        }
        return mStates;
    }

    /******************************************************************************//**
     * @brief Evaluate objective function
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return objective function value
    **********************************************************************************/
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE VALUE REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        return mObjective->value(aState, aControl, mTimeStep);
    }

    /******************************************************************************//**
     * @brief Evaluate constraint function
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return constraint function value
    **********************************************************************************/
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT VALUE REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(aState, tLastStepIndex, Kokkos::ALL());
        return mConstraint->value(tState, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate objective function
     * @param [in] aControl 1D view of control variables
     * @return objective function value
    **********************************************************************************/
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl)
    {
        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE VALUE REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        Plato::ScalarMultiVector tStates = solution(aControl);
        return mObjective->value(tStates, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate constraint function
     * @param [in] aControl 1D view of control variables
     * @return constraint function value
    **********************************************************************************/
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControl)
    {
        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT VALUE REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(mStates, tLastStepIndex, Kokkos::ALL());
        return mConstraint->value(tState, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate objective gradient wrt control variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return 1D view - objective gradient wrt control variables
    **********************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        // compute dfdz: partial of objective wrt z
        auto t_df_dz = mObjective->gradient_z(aState, aControl, mTimeStep);

        // outer loop for load/time steps
        auto tLastStepIndex = mNumSteps - 1;
        for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--)
        {
            // compute dfdu: partial of objective wrt u
            auto t_df_du = mObjective->gradient_u(aState, aControl, mTimeStep, tStepIndex);
            Plato::scale(static_cast<Plato::Scalar>(-1), t_df_du);

            // compute nodal projection of pressure gradient
            Plato::ScalarVector tStateAtStepK = Kokkos::subview(aState, tStepIndex, Kokkos::ALL());
            Plato::fill(static_cast<Plato::Scalar>(0.0), mProjPGrad);
            // extract projection state
            Plato::extract<SimplexPhysics::mNumDofsPerNode,
                           SimplexPhysics::ProjectorT::SimplexT::mProjectionDof>(tStateAtStepK, mProjectState);
            mProjResidual = mStateProjection.value      (mProjPGrad, mProjectState, aControl);
            mProjJacobian = mStateProjection.gradient_u (mProjPGrad, mProjectState, aControl);
            Plato::Solve::RowSummed<SimplexPhysics::mNumSpatialDims>(mProjJacobian, mProjPGrad, mProjResidual);

            // compute dgdu^T: Transpose of partial of PDE wrt state
            mJacobian = mEqualityConstraint.gradient_u_T(tStateAtStepK, mProjPGrad, aControl);

            // compute dPdn^T: Transpose of partial of projection residual wrt state
            auto t_dP_dn_T = mStateProjection.gradient_n_T(mProjPGrad, mProjectState, aControl);

            // compute dgdPI^T: Transpose of partial of PDE wrt projected pressure gradient
            auto t_dg_dPI_T = mEqualityConstraint.gradient_n_T(tStateAtStepK, mProjPGrad, aControl);

            // compute dgdu^T - dP_dn_T X (mProjJacobian)^-1 X t_dg_dPI_T
            auto tRow = SimplexPhysics::ProjectorT::SimplexT::mProjectionDof;
            Plato::Condense(mJacobian, t_dP_dn_T, mProjJacobian,  t_dg_dPI_T, tRow);

            this->applyConstraints(mJacobian, t_df_du);

            Plato::ScalarVector tLambda = Kokkos::subview(mLambda, tStepIndex, Kokkos::ALL());
            Plato::Solve::Consistent<SimplexPhysics::mNumDofsPerNode>(mJacobian, tLambda, t_df_du);

            // compute adjoint variable for projection equation
            Plato::fill(static_cast<Plato::Scalar>(0.0), mProjResidual);
            Plato::MatrixTimesVectorPlusVector(t_dg_dPI_T, tLambda, mProjResidual);
            Plato::scale(static_cast<Plato::Scalar>(-1), mProjResidual);
            Plato::Solve::RowSummed<SimplexPhysics::mNumSpatialDims>(mProjJacobian, mEta, mProjResidual);

            // compute dgdz: partial of PDE wrt state.
            // dgdz is returned transposed, nxm.  n=z.size() and m=u.size().
            auto t_dg_dz = mEqualityConstraint.gradient_z(tStateAtStepK, mProjPGrad, aControl);

            // compute dfdz += dgdz . lambda
            // dPdz is returned transposed, nxm.  n=z.size() and m=u.size().
            Plato::MatrixTimesVectorPlusVector(t_dg_dz, tLambda, t_df_dz);

            // compute dPdz: partial of projection wrt state.
            // dPdz is returned transposed, nxm.  n=z.size() and m=PI.size().
            auto t_dP_dz = mStateProjection.gradient_z(mProjPGrad, tStateAtStepK, aControl);

            // compute dfdz += dPdz . eta
            Plato::MatrixTimesVectorPlusVector(t_dP_dz, mEta, t_df_dz);
        }

        return t_df_dz;
    }

    /******************************************************************************//**
     * @brief Evaluate objective gradient wrt configuration variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return 1D view - objective gradient wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        // compute dfdx: partial of objective wrt x
        auto t_df_dx = mObjective->gradient_x(aState, aControl, mTimeStep);

        // outer loop for load/time steps
        auto tLastStepIndex = mNumSteps - 1;
        for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--)
        {
            // compute dfdu: partial of objective wrt u
            auto t_df_du = mObjective->gradient_u(aState, aControl, mTimeStep, tStepIndex);
            Plato::scale(static_cast<Plato::Scalar>(-1), t_df_du);

            // compute nodal projection of pressure gradient
            Plato::ScalarVector tStateAtStepK = Kokkos::subview(aState, tStepIndex, Kokkos::ALL());
            Plato::fill(static_cast<Plato::Scalar>(0.0), mProjPGrad);
            auto mProjResidual = mStateProjection.value      (mProjPGrad, tStateAtStepK, aControl);
            auto mProjJacobian = mStateProjection.gradient_u (mProjPGrad, tStateAtStepK, aControl);
            // extract projection state
            Plato::extract<SimplexPhysics::mNumDofsPerNode,
                           SimplexPhysics::ProjectorT::SimplexT::mProjectionDof>(tStateAtStepK, mProjectState);
            mProjResidual = mStateProjection.value      (mProjPGrad, mProjectState, aControl);
            mProjJacobian = mStateProjection.gradient_u (mProjPGrad, mProjectState, aControl);
            Plato::Solve::RowSummed<SimplexPhysics::mNumSpatialDims>(mProjJacobian, mProjPGrad, mProjResidual);

            // compute dgdu^T: Transpose of partial of PDE wrt state
            mJacobian = mEqualityConstraint.gradient_u_T(tStateAtStepK, mProjPGrad, aControl);

            // compute dPdn^T: Transpose of partial of projection residual wrt state
            auto t_dP_dn_T = mStateProjection.gradient_n_T(mProjPGrad, mProjectState, aControl);

            // compute dgdPI: Transpose of partial of PDE wrt projected pressure gradient
            auto t_dg_dPI_T = mEqualityConstraint.gradient_n_T(tStateAtStepK, mProjPGrad, aControl);

            // compute dgdu^T - dP_dn_T X (mProjJacobian)^-1 X t_dg_dPI_T
            auto tRow = SimplexPhysics::ProjectorT::SimplexT::mProjectionDof;
            Plato::Condense(mJacobian, t_dP_dn_T, mProjJacobian,  t_dg_dPI_T, tRow);

            this->applyConstraints(mJacobian, t_df_du);

            Plato::ScalarVector tLambda = Kokkos::subview(mLambda, tStepIndex, Kokkos::ALL());
            Plato::Solve::Consistent<SimplexPhysics::mNumDofsPerNode>(mJacobian, tLambda, t_df_du);

            // compute adjoint variable for projection equation
            Plato::fill(static_cast<Plato::Scalar>(0.0), mProjResidual);
            Plato::MatrixTimesVectorPlusVector(t_dg_dPI_T, tLambda, mProjResidual);
            Plato::scale(static_cast<Plato::Scalar>(-1), mProjResidual);
            Plato::Solve::RowSummed<SimplexPhysics::mNumSpatialDims>(mProjJacobian, mEta, mProjResidual);

            // compute dgdx: partial of PDE wrt configuration
            // dgdx is returned transposed, nxm.  n=z.size() and m=u.size().
            auto t_dg_dx = mEqualityConstraint.gradient_x(tStateAtStepK, mProjPGrad, aControl);

            // compute dfdx += dgdx . lambda
            // dPdx is returned transposed, nxm.  n=z.size() and m=u.size().
            Plato::MatrixTimesVectorPlusVector(t_dg_dx, tLambda, t_df_dx);

            // compute dPdx: partial of projection wrt configuration
            // dPdx is returned transposed, nxm.  n=z.size() and m=PI.size().
            auto t_dP_dx = mStateProjection.gradient_x(mProjPGrad, tStateAtStepK, aControl);

            // compute dfdx += dPdx . eta
            Plato::MatrixTimesVectorPlusVector(t_dP_dx, mEta, t_df_dx);
        }

        return t_df_dx;
    }

    /******************************************************************************//**
     * @brief Evaluate constraint partial derivative wrt control variables
     * @param [in] aControl 1D view of control variables
     * @return 1D view - constraint partial derivative wrt control variables
    **********************************************************************************/
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControl)
    {
        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT GRADIENT REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(mStates, tLastStepIndex, Kokkos::ALL());
        return mConstraint->gradient_z(tState, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate constraint partial derivative wrt control variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return 1D view - constraint partial derivative wrt control variables
    **********************************************************************************/
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT GRADIENT REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(aState, tLastStepIndex, Kokkos::ALL());
        return mConstraint->gradient_z(tState, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate objective partial derivative wrt control variables
     * @param [in] aControl 1D view of control variables
     * @return 1D view - objective partial derivative wrt control variables
    **********************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl)
    {
        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        return mObjective->gradient_z(mStates, aControl, mTimeStep);
    }

    /******************************************************************************//**
     * @brief Evaluate objective partial derivative wrt configuration variables
     * @param [in] aControl 1D view of control variables
     * @return 1D view - objective partial derivative wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControl)
    {
        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE CONFIGURATION GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        return mObjective->gradient_x(mStates, aControl, mTimeStep);
    }

    /******************************************************************************//**
     * @brief Evaluate constraint partial derivative wrt configuration variables
     * @param [in] aControl 1D view of control variables
     * @return 1D view - constraint partial derivative wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControl)
    {
        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT CONFIGURATION GRADIENT REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(mStates, tLastStepIndex, Kokkos::ALL());
        return mConstraint->gradient_x(tState, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate constraint partial derivative wrt configuration variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return 1D view - constraint partial derivative wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT CONFIGURATION GRADIENT REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(aState, tLastStepIndex, Kokkos::ALL());
        return mConstraint->gradient_x(tState, aControl);
    }

private:
    /******************************************************************************//**
     * @brief Initialize member data
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputParams)
    {
        if(aInputParams.isSublist("Time Stepping") == true)
        {
            mNumSteps = aInputParams.sublist("Time Stepping").get<int>("Number Time Steps");
            mTimeStep = aInputParams.sublist("Time Stepping").get<Plato::Scalar>("Time Step");
        } 
        else
        {
            mNumSteps = 1;
            mTimeStep = 1.0;
        }

        if(aInputParams.isSublist("Newton Iteration") == true)
        {
            mNumNewtonSteps = aInputParams.sublist("Newton Iteration").get<int>("Number Iterations");
        } 
        else
        {
            mNumNewtonSteps = 2;
        }

        Plato::ScalarFunctionBaseFactory<SimplexPhysics> tFunctionBaseFactory;
        Plato::ScalarFunctionIncBaseFactory<SimplexPhysics> tFunctionIncBaseFactory;
        if(aInputParams.isType<std::string>("Constraint"))
        {
            std::string tName = aInputParams.get<std::string>("Constraint");
            mConstraint = tFunctionBaseFactory.create(aMesh, aMeshSets, mDataMap, aInputParams, tName);
        }

        if(aInputParams.isType<std::string>("Objective"))
        {
            std::string tName = aInputParams.get<std::string>("Objective");
            mObjective = tFunctionIncBaseFactory.create(aMesh, aMeshSets, mDataMap, aInputParams, tName);

            auto tLength = mEqualityConstraint.size();
            mLambda = Plato::ScalarMultiVector("Lambda", mNumSteps, tLength);

            mEta = Plato::ScalarVector("Eta", tLength);
        }

        // parse constraints
        //
        Plato::EssentialBCs<SimplexPhysics>
            tEssentialBoundaryConditions(aInputParams.sublist("Essential Boundary Conditions",false));
        tEssentialBoundaryConditions.get(aMeshSets, mBcDofs, mBcValues);
    }
};
// class EllipticVMSProblem

} // namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATO_1D
extern template class Plato::EllipticVMSProblem<::Plato::StabilizedMechanics<1>>;
//extern template class Plato::EllipticVMSProblem<::Plato::Electromechanics<1>>;
extern template class Plato::EllipticVMSProblem<::Plato::StabilizedThermomechanics<1>>;
#endif
#ifdef PLATO_2D
extern template class Plato::EllipticVMSProblem<::Plato::StabilizedMechanics<2>>;
//extern template class Plato::EllipticVMSProblem<::Plato::Electromechanics<2>>;
extern template class Plato::EllipticVMSProblem<::Plato::StabilizedThermomechanics<2>>;
#endif
#ifdef PLATO_3D
extern template class Plato::EllipticVMSProblem<::Plato::StabilizedMechanics<3>>;
//extern template class Plato::EllipticVMSProblem<::Plato::Electromechanics<3>>;
extern template class Plato::EllipticVMSProblem<::Plato::StabilizedThermomechanics<3>>;
#endif

#endif // PLATO_PROBLEM_HPP
