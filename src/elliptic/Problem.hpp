#ifndef PLATO_PROBLEM_HPP
#define PLATO_PROBLEM_HPP

#include "PlatoUtilities.hpp"

#include <memory>
#include <sstream>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "BLAS1.hpp"
#include "NaturalBCs.hpp"
#include "EssentialBCs.hpp"
#include "ImplicitFunctors.hpp"
#include "ApplyConstraints.hpp"
#include "MultipointConstraints.hpp"

#include "PlatoMathHelpers.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoAbstractProblem.hpp"

#include "elliptic/VectorFunction.hpp"
#include "elliptic/ScalarFunctionBaseFactory.hpp"
#include "AnalyzeMacros.hpp"

#include "alg/ParallelComm.hpp"
#include "alg/PlatoSolverFactory.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Manage scalar and vector function evaluations
**********************************************************************************/
template<typename SimplexPhysics>
class Problem: public Plato::AbstractProblem
{
private:

    static constexpr Plato::OrdinalType SpatialDim = SimplexPhysics::mNumSpatialDims; /*!< spatial dimensions */

    // required
    std::shared_ptr<Plato::Elliptic::VectorFunction<SimplexPhysics>> mPDE; /*!< equality constraint interface */

    // optional
    std::shared_ptr<Plato::Elliptic::ScalarFunctionBase> mConstraint; /*!< constraint constraint interface */
    std::shared_ptr<Plato::Elliptic::ScalarFunctionBase> mObjective; /*!< objective constraint interface */

    Plato::ScalarMultiVector mAdjoint;
    Plato::ScalarVector mResidual;

    Plato::ScalarMultiVector mStates; /*!< state variables */

    bool mIsSelfAdjoint; /*!< indicates if problem is self-adjoint */

    Teuchos::RCP<Plato::CrsMatrixType> mJacobian; /*!< Jacobian matrix */

    Plato::LocalOrdinalVector mBcDofs; /*!< list of degrees of freedom associated with the Dirichlet boundary conditions */
    Plato::ScalarVector mBcValues; /*!< values associated with the Dirichlet boundary conditions */

    std::shared_ptr<Plato::MultipointConstraints<SimplexPhysics>> mMPCs; /*!< multipoint constraint interface */

    rcp<Plato::AbstractSolver> mSolver;

public:
    /******************************************************************************//**
     * \brief PLATO problem constructor
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    Problem(
      Omega_h::Mesh& aMesh,
      Omega_h::MeshSets& aMeshSets,
      Teuchos::ParameterList& aInputParams,
      Comm::Machine aMachine
    ) :
      mPDE(std::make_shared<Plato::Elliptic::VectorFunction<SimplexPhysics>>(aMesh, aMeshSets, mDataMap, aInputParams, aInputParams.get<std::string>("PDE Constraint"))),
      mConstraint(nullptr),
      mObjective(nullptr),
      mResidual("MyResidual", mPDE->size()),
      mStates("States", static_cast<Plato::OrdinalType>(1), mPDE->size()),
      mJacobian(Teuchos::null),
      mIsSelfAdjoint(aInputParams.get<bool>("Self-Adjoint", false)),
      mMPCs(nullptr)
    {
        this->initialize(aMesh, aMeshSets, aInputParams);

        Plato::SolverFactory tSolverFactory(aInputParams.sublist("Linear Solver"));
        mSolver = tSolverFactory.create(aMesh.nverts(), aMachine, SimplexPhysics::mNumDofsPerNode);
    }

    virtual ~Problem(){}

    Plato::OrdinalType numNodes() const
    {
        const auto tNumNodes = mPDE->numNodes();
        return (tNumNodes);
    }

    Plato::OrdinalType numCells() const
    {
        const auto tNumCells = mPDE->numCells();
        return (tNumCells);
    }
    
    Plato::OrdinalType numDofsPerCell() const
    {
        const auto tNumDofsPerCell = mPDE->numDofsPerCell();
        return (tNumDofsPerCell);
    }

    Plato::OrdinalType numNodesPerCell() const
    {
        const auto tNumNodesPerCell = mPDE->numNodesPerCell();
        return (tNumNodesPerCell);
    }

    Plato::OrdinalType numDofsPerNode() const
    {
        const auto tNumDofsPerNode = mPDE->numDofsPerNode();
        return (tNumDofsPerNode);
    }

    Plato::OrdinalType numControlsPerNode() const
    {
        const auto tNumControlsPerNode = mPDE->numControlsPerNode();
        return (tNumControlsPerNode);
    }

    void appendResidual(const std::shared_ptr<Plato::Elliptic::VectorFunction<SimplexPhysics>>& aPDE)
    {
        mPDE = aPDE;
    }

    void appendObjective(const std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>& aObjective)
    {
        mObjective = aObjective;
    }

    void appendConstraint(const std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>& aConstraint)
    {
        mConstraint = aConstraint;
    }

    /******************************************************************************//**
     * \brief Return number of degrees of freedom in solution.
     * \return Number of degrees of freedom
    **********************************************************************************/
    Plato::OrdinalType getNumSolutionDofs()
    {
        return SimplexPhysics::mNumDofsPerNode;
    }

    /******************************************************************************//**
     * \brief Set state variables
     * \param [in] aGlobalState 2D view of state variables
    **********************************************************************************/
    void setGlobalState(const Plato::ScalarMultiVector & aGlobalState)
    {
        assert(aGlobalState.extent(0) == mStates.extent(0));
        assert(aGlobalState.extent(1) == mStates.extent(1));
        Kokkos::deep_copy(mStates, aGlobalState);
    }

    /******************************************************************************//**
     * \brief Return 2D view of state variables
     * \return aGlobalState 2D view of state variables
    **********************************************************************************/
    Plato::ScalarMultiVector getGlobalState()
    {
        return mStates;
    }

    /******************************************************************************//**
     * \brief Return 2D view of adjoint variables
     * \return 2D view of adjoint variables
    **********************************************************************************/
    Plato::ScalarMultiVector getAdjoint()
    {
        return mAdjoint;
    }

    /******************************************************************************//**
     * \brief Apply Dirichlet constraints
     * \param [in] aMatrix Compressed Row Storage (CRS) matrix
     * \param [in] aVector 1D view of Right-Hand-Side forces
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
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aGlobalState 2D container of state variables
     * \param [in] aControl 1D container of control variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aGlobalState)
    {
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aGlobalState, tTIME_STEP_INDEX, Kokkos::ALL());
        mObjective->updateProblem(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * \brief Solve system of equations
     * \param [in] aControl 1D view of control variables
     * \return 2D view of state variables
    **********************************************************************************/
    Plato::ScalarMultiVector solution(const Plato::ScalarVector & aControl)
    {
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tStatesSubView);

        mResidual = mPDE->value(tStatesSubView, aControl);
        Plato::blas1::scale(-1.0, mResidual);

        mJacobian = mPDE->gradient_u(tStatesSubView, aControl);
        this->applyConstraints(mJacobian, mResidual);

        if(mMPCs)
        {
            const Plato::OrdinalType tNumChildNodes = mMPCs->getNumChildNodes();

            Teuchos::RCP<Plato::CrsMatrixType> tTransformMatrix = mMPCs->getTransformMatrix();
            Teuchos::RCP<Plato::CrsMatrixType> tTransformMatrixTranspose = mMPCs->getTransformMatrixTranspose();
            Plato::ScalarVector tMpcRhs = mMPCs->getRhsVector();

            auto tNumNodes           = mPDE->numNodes();
            auto tNumDofsPerNode     = mPDE->numDofsPerNode();
            auto tNumTotalDofs       = tNumNodes*tNumDofsPerNode;
            auto tNumCondensedDofs   = (tNumNodes - tNumChildNodes)*tNumDofsPerNode;

            auto tCondensedJacobianLeft = Teuchos::rcp( new Plato::CrsMatrixType(tNumTotalDofs, tNumCondensedDofs, tNumDofsPerNode, tNumDofsPerNode) );
            auto tCondensedJacobian     = Teuchos::rcp( new Plato::CrsMatrixType(tNumCondensedDofs, tNumCondensedDofs, tNumDofsPerNode, tNumDofsPerNode) );

            Plato::MatrixMatrixMultiply(mJacobian, tTransformMatrix, tCondensedJacobianLeft);
            Plato::MatrixMatrixMultiply(tTransformMatrixTranspose, tCondensedJacobianLeft, tCondensedJacobian);

            Plato::ScalarVector tInnerResidual = mResidual;
            Plato::blas1::scale(-1.0, tMpcRhs);
            Plato::MatrixTimesVectorPlusVector(mJacobian, tMpcRhs, tInnerResidual);

            Plato::ScalarVector tCondensedResidual("Condensed Residual", tNumCondensedDofs);
            Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tCondensedResidual);

            Plato::MatrixTimesVectorPlusVector(tTransformMatrixTranspose, tInnerResidual, tCondensedResidual);

            /* if(tCondensedJacobian->isBlockMatrix()) */
            /* { */
            /*     Plato::setBlockConstrainedDiagonals<SimplexPhysics::mNumDofsPerNode>(tCondensedJacobian, tCondensedResidual, tMpcChildDofs); */
            /* } */
            /* else */
            /* { */
            /*     Plato::setConstrainedDiagonals<SimplexPhysics::mNumDofsPerNode>(tCondensedJacobian, tCondensedResidual, tMpcChildDofs); */
            /* } */

            Plato::ScalarVector tCondensedState("Condensed State Solution", tNumCondensedDofs);
            Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tCondensedState);
            mSolver->solve(*tCondensedJacobian, tCondensedState, tCondensedResidual);

            Plato::ScalarVector tFullState("Full State Solution", tStatesSubView.extent(0));
            Plato::blas1::copy(tMpcRhs, tFullState);
            Plato::blas1::scale(-1.0, tFullState); // since tMpcRhs was scaled by -1 above, set back to original values

            Plato::MatrixTimesVectorPlusVector(tTransformMatrix, tCondensedState, tFullState);
            Plato::blas1::axpy<Plato::ScalarVector>(1.0, tFullState, tStatesSubView);

        }
        else
        {
            mSolver->solve(*mJacobian, tStatesSubView, mResidual);
        }

        mResidual = mPDE->value(tStatesSubView, aControl);
        Plato::blas1::scale(-1.0, mResidual);

        return mStates;
    }

    /******************************************************************************//**
     * \brief Evaluate objective function
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \return objective function value
    **********************************************************************************/
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aGlobalState)
    {
        assert(aGlobalState.extent(0) == mStates.extent(0));
        assert(aGlobalState.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            const std::string tErrorMessage = "OBJECTIVE POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aGlobalState, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tObjFuncValue = mObjective->value(tStatesSubView, aControl);
        return tObjFuncValue;
    }

    /******************************************************************************//**
     * \brief Evaluate constraint function
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \return constraint function value
    **********************************************************************************/
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aGlobalState)
    {
        assert(aGlobalState.extent(0) == mStates.extent(0));
        assert(aGlobalState.extent(1) == mStates.extent(1));

        if(mConstraint == nullptr)
        {
            const std::string tErrorMessage = "CONSTRAINT POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aGlobalState, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->value(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * \brief Evaluate objective function
     * \param [in] aControl 1D view of control variables
     * \return objective function value
    **********************************************************************************/
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl)
    {
        if(mObjective == nullptr)
        {
            const std::string tErrorMessage = "OBJECTIVE POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        Plato::ScalarMultiVector tStates = solution(aControl);
        auto tStatesSubView = Kokkos::subview(tStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mObjective->value(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * \brief Evaluate constraint function
     * \param [in] aControl 1D view of control variables
     * \return constraint function value
    **********************************************************************************/
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControl)
    {
        if(mConstraint == nullptr)
        {
            const std::string tErrorMessage = "CONSTRAINT POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->value(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * \brief Evaluate objective gradient wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \return 1D view - objective gradient wrt control variables
    **********************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aGlobalState)
    {
        assert(aGlobalState.extent(0) == mStates.extent(0));
        assert(aGlobalState.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            const std::string tErrorMessage = "OBJECTIVE POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        if(static_cast<Plato::OrdinalType>(mAdjoint.size()) <= static_cast<Plato::OrdinalType>(0))
        {
            const auto tLength = mPDE->size();
            mAdjoint = Plato::ScalarMultiVector("Adjoint Variables", 1, tLength);
        }

        // compute dfdz: partial of objective wrt z
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tPartialObjectiveWRT_Control = mObjective->gradient_z(tStatesSubView, aControl);
        if(mIsSelfAdjoint)
        {
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tPartialObjectiveWRT_Control);
        }
        else
        {
            // compute dfdu: partial of objective wrt u
            auto tPartialObjectiveWRT_State = mObjective->gradient_u(tStatesSubView, aControl);
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tPartialObjectiveWRT_State);

            // compute dgdu: partial of PDE wrt state
            mJacobian = mPDE->gradient_u_T(tStatesSubView, aControl);

            this->applyAdjointConstraints(mJacobian, tPartialObjectiveWRT_State);

            // adjoint problem uses transpose of global stiffness, but we're assuming the constrained
            // system is symmetric.

            Plato::ScalarVector tAdjointSubView = Kokkos::subview(mAdjoint, tTIME_STEP_INDEX, Kokkos::ALL());

            mSolver->solve(*mJacobian, tAdjointSubView, tPartialObjectiveWRT_State);

            // compute dgdz: partial of PDE wrt state.
            // dgdz is returned transposed, nxm.  n=z.size() and m=u.size().
            auto tPartialPDE_WRT_Control = mPDE->gradient_z(tStatesSubView, aControl);

            // compute dgdz . adjoint + dfdz
            Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Control, tAdjointSubView, tPartialObjectiveWRT_Control);
        }
        return tPartialObjectiveWRT_Control;
    }

    /******************************************************************************//**
     * \brief Evaluate objective gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \return 1D view - objective gradient wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aGlobalState)
    {
        assert(aGlobalState.extent(0) == mStates.extent(0));
        assert(aGlobalState.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            const std::string tErrorMessage = "OBJECTIVE POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        // compute partial derivative wrt x
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aGlobalState, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tPartialObjectiveWRT_Config  = mObjective->gradient_x(tStatesSubView, aControl);

        if(mIsSelfAdjoint)
        {
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tPartialObjectiveWRT_Config);
        }
        else
        {
            // compute dfdu: partial of objective wrt u
            auto tPartialObjectiveWRT_State = mObjective->gradient_u(tStatesSubView, aControl);
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tPartialObjectiveWRT_State);

            // compute dgdu: partial of PDE wrt state
            mJacobian = mPDE->gradient_u(tStatesSubView, aControl);
            this->applyConstraints(mJacobian, tPartialObjectiveWRT_State);

            // adjoint problem uses transpose of global stiffness, but we're assuming the constrained
            // system is symmetric.

            Plato::ScalarVector
              tAdjointSubView = Kokkos::subview(mAdjoint, tTIME_STEP_INDEX, Kokkos::ALL());

            mSolver->solve(*mJacobian, tAdjointSubView, tPartialObjectiveWRT_State);

            // compute dgdx: partial of PDE wrt config.
            // dgdx is returned transposed, nxm.  n=x.size() and m=u.size().
            auto tPartialPDE_WRT_Config = mPDE->gradient_x(tStatesSubView, aControl);

            // compute dgdx . adjoint + dfdx
            Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Config, tAdjointSubView, tPartialObjectiveWRT_Config);
        }
        return tPartialObjectiveWRT_Config;
    }

    /******************************************************************************//**
     * \brief Evaluate constraint partial derivative wrt control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view - constraint partial derivative wrt control variables
    **********************************************************************************/
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControl)
    {
        if(mConstraint == nullptr)
        {
            const std::string tErrorMessage = "CONSTRAINT POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->gradient_z(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * \brief Evaluate constraint partial derivative wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \return 1D view - constraint partial derivative wrt control variables
    **********************************************************************************/
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aGlobalState)
    {
        assert(aGlobalState.extent(0) == mStates.extent(0));
        assert(aGlobalState.extent(1) == mStates.extent(1));

        if(mConstraint == nullptr)
        {
            const std::string tErrorMessage = "CONSTRAINT POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aGlobalState, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->gradient_z(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * \brief Evaluate objective partial derivative wrt control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view - objective partial derivative wrt control variables
    **********************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl)
    {
        if(mObjective == nullptr)
        {
            const std::string tErrorMessage = "OBJECTIVE POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mObjective->gradient_z(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * \brief Evaluate objective partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view - objective partial derivative wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControl)
    {
        if(mObjective == nullptr)
        {
            const std::string tErrorMessage = "OBJECTIVE POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mObjective->gradient_x(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * \brief Evaluate constraint partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view - constraint partial derivative wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControl)
    {
        if(mConstraint == nullptr)
        {
            const std::string tErrorMessage = "CONSTRAINT POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->gradient_x(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * \brief Evaluate constraint partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \return 1D view - constraint partial derivative wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aGlobalState)
    {
        assert(aGlobalState.extent(0) == mStates.extent(0));
        assert(aGlobalState.extent(1) == mStates.extent(1));

        if(mConstraint == nullptr)
        {
            const std::string tErrorMessage = "CONSTRAINT POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aGlobalState, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->gradient_x(tStatesSubView, aControl);
    }

    /***************************************************************************//**
     * \brief Read essential (Dirichlet) boundary conditions from the Exodus file.
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
    *******************************************************************************/
    void readEssentialBoundaryConditions(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputParams)
    {
        if(aInputParams.isSublist("Essential Boundary Conditions") == false)
        {
            THROWERR("ESSENTIAL BOUNDARY CONDITIONS SUBLIST IS NOT DEFINED IN THE INPUT FILE.")
        }
        Plato::EssentialBCs<SimplexPhysics> tEssentialBoundaryConditions(aInputParams.sublist("Essential Boundary Conditions", false));
        tEssentialBoundaryConditions.get(aMeshSets, mBcDofs, mBcValues);
    }

    /***************************************************************************//**
     * \brief Set essential (Dirichlet) boundary conditions
     * \param [in] aDofs   degrees of freedom associated with Dirichlet boundary conditions
     * \param [in] aValues values associated with Dirichlet degrees of freedom
    *******************************************************************************/
    void setEssentialBoundaryConditions(const Plato::LocalOrdinalVector & aDofs, const Plato::ScalarVector & aValues)
    {
        if(aDofs.size() != aValues.size())
        {
            std::ostringstream tError;
            tError << "DIMENSION MISMATCH: THE NUMBER OF ELEMENTS IN INPUT DOFS AND VALUES ARRAY DO NOT MATCH."
                << "DOFS SIZE = " << aDofs.size() << " AND VALUES SIZE = " << aValues.size();
            THROWERR(tError.str())
        }
        mBcDofs = aDofs;
        mBcValues = aValues;
    }

private:
    /******************************************************************************//**
     * \brief Initialize member data
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputParams)
    {
        auto tName = aInputParams.get<std::string>("PDE Constraint");
        mPDE = std::make_shared<Plato::Elliptic::VectorFunction<SimplexPhysics>>(aMesh, aMeshSets, mDataMap, aInputParams, tName);

        Plato::Elliptic::ScalarFunctionBaseFactory<SimplexPhysics> tFunctionBaseFactory;
        if(aInputParams.isType<std::string>("Constraint"))
        {
            std::string tName = aInputParams.get<std::string>("Constraint");
            mConstraint = tFunctionBaseFactory.create(aMesh, aMeshSets, mDataMap, aInputParams, tName);
        }

        if(aInputParams.isType<std::string>("Objective"))
        {
            std::string tName = aInputParams.get<std::string>("Objective");
            mObjective = tFunctionBaseFactory.create(aMesh, aMeshSets, mDataMap, aInputParams, tName);

            auto tLength = mPDE->size();
            mAdjoint = Plato::ScalarMultiVector("Adjoint Variables", 1, tLength);
        }

        if(aInputParams.isType<std::string>("Multipoint Constraints"))
        {
            Plato::OrdinalType tNumNodes = mPDE->numNodes();
            auto & tMyParams = aInputParams.sublist("Multipoint Constraints", false);
            mMPCs = std::make_shared<Plato::MultipointConstraints<SimplexPhysics>>(tNumNodes, tMyParams);
            mMPCs->setupTransform(aMeshSets);
        }
    }

    void applyAdjointConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector)
    {
        Plato::ScalarVector tDirichletValues("Dirichlet Values For Adjoint Problem", mBcValues.size());
        Plato::blas1::scale(static_cast<Plato::Scalar>(0.0), tDirichletValues);
        if(mJacobian->isBlockMatrix())
        {
            Plato::applyBlockConstraints<SimplexPhysics::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, tDirichletValues);
        }
        else
        {
            Plato::applyConstraints<SimplexPhysics::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, tDirichletValues);
        }
    }

};
// class Problem

} // namespace Elliptic

} // namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Elliptic::Problem<::Plato::Thermal<1>>;
extern template class Plato::Elliptic::Problem<::Plato::Mechanics<1>>;
extern template class Plato::Elliptic::Problem<::Plato::Electromechanics<1>>;
extern template class Plato::Elliptic::Problem<::Plato::Thermomechanics<1>>;
#endif
#ifdef PLATOANALYZE_2D
extern template class Plato::Elliptic::Problem<::Plato::Thermal<2>>;
extern template class Plato::Elliptic::Problem<::Plato::Mechanics<2>>;
extern template class Plato::Elliptic::Problem<::Plato::Electromechanics<2>>;
extern template class Plato::Elliptic::Problem<::Plato::Thermomechanics<2>>;
#endif
#ifdef PLATOANALYZE_3D
extern template class Plato::Elliptic::Problem<::Plato::Thermal<3>>;
extern template class Plato::Elliptic::Problem<::Plato::Mechanics<3>>;
extern template class Plato::Elliptic::Problem<::Plato::Electromechanics<3>>;
extern template class Plato::Elliptic::Problem<::Plato::Thermomechanics<3>>;
#endif

#endif // PLATO_PROBLEM_HPP
