/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2022, JSK, The University of Tokyo.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the JSK, The University of Tokyo nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: Hirokazu Ishida */

#include <type_traits>
#include "ompl/util/Console.h"
#define BOOST_TEST_MODULE "ExperienceBasedPlanning"
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>

#include <memory>

#include "ompl/base/State.h"
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/tools/lightning/Lightning.h>
#include <ompl/tools/thunder/Thunder.h>

#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/kpiece/KPIECE1.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;
namespace ot = ompl::tools;

template <typename ExpPlannerT, typename PlannerT, typename ReplannerT>
struct PlannerSetter
{
    static void setPlanners(const std::unique_ptr<ExpPlannerT> &exp_planner)
    {
        const auto si = exp_planner->getSpaceInformation();
        const auto algo_planner = std::make_shared<PlannerT>(si);
        const auto algo_replanner = std::make_shared<ReplannerT>(si);
        exp_planner->setPlanner(algo_planner);
        exp_planner->setRepairPlanner(algo_replanner);
    }
};

template <typename ExpPlannerT>
struct PlannerSetter<ExpPlannerT, void, void>
{
    static void setPlanners(const std::unique_ptr<ExpPlannerT> &exp_planner)
    {
        // don't set
        (void)exp_planner;
    }
};

template <typename ExpPlannerT, typename PlannerT, typename ReplannerT>
struct ExperienceBasedPlannerTest
{
    static std::unique_ptr<ExpPlannerT> setupExperienceBasedPlanner()
    {
        // create planner for 2d square space
        const auto space(std::make_shared<ob::RealVectorStateSpace>());
        space->addDimension(0.0, 1.0);
        space->addDimension(0.0, 1.0);
        const auto si = std::make_shared<ob::SpaceInformation>(space);
        si->setStateValidityCheckingResolution(0.002);
        auto exp_planner = std::make_unique<ExpPlannerT>(si);
        PlannerSetter<ExpPlannerT, PlannerT, ReplannerT>::setPlanners(exp_planner);
        exp_planner->setStateValidityChecker([](const ob::State *state) {
            const auto rs = state->as<ob::RealVectorStateSpace::StateType>();
            double x = rs->values[0];
            double y = rs->values[1];
            if (x < 0.2 || x > 0.8)
            {
                return true;
            }
            return std::abs(y - 0.5) < 0.004;
        });

        return exp_planner;
    }

    static void runTest()
    {
        const auto temp_path = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();
        const size_t n_problem = 30;

        // solve problem
        {
            const auto exp_planner = setupExperienceBasedPlanner();
            exp_planner->setFilePath(temp_path.string());

            const auto valid_sampler = exp_planner->getSpaceInformation()->allocValidStateSampler();
            // note that the experience is not always stored even when solved
            // so we need while loop
            while (exp_planner->getExperiencesCount() < n_problem)
            {
                exp_planner->clear();
                ob::ScopedState<> start(exp_planner->getStateSpace());
                valid_sampler->sample(start.get());
                ob::ScopedState<> goal(exp_planner->getStateSpace());
                valid_sampler->sample(goal.get());
                exp_planner->setStartAndGoalStates(start, goal);
                exp_planner->solve(10);
                exp_planner->doPostProcessing();
            }
            BOOST_CHECK_EQUAL(exp_planner->getExperiencesCount(), n_problem);
            exp_planner->save();
        }

        // use experience to solve problem
        // if the planner has no experience, and database file path is already set, the database
        // is supposed to be loaded at the setup() time.
        {
            const auto exp_planner = setupExperienceBasedPlanner();

            const auto space = exp_planner->getSpaceInformation()->getStateSpace();
            const auto valid_sampler = exp_planner->getSpaceInformation()->allocValidStateSampler();
            ob::ScopedState<> start(exp_planner->getStateSpace());
            ob::ScopedState<> goal(exp_planner->getStateSpace());
            auto rs_start = start->as<ob::RealVectorStateSpace::StateType>();
            auto rs_goal = goal->as<ob::RealVectorStateSpace::StateType>();
            rs_start->values[0] = 0.1;
            rs_start->values[1] = 0.1;
            rs_goal->values[0] = 0.9;
            rs_goal->values[1] = 0.9;
            exp_planner->setStartAndGoalStates(start, goal);

            BOOST_CHECK_EQUAL(exp_planner->getExperiencesCount(), 0);
            exp_planner->setFilePath(temp_path.string());
            exp_planner->setup();  // load experience
            BOOST_CHECK_EQUAL(exp_planner->getExperiencesCount(), n_problem);

            OMPL_INFORM("solve without database");
            exp_planner->clear();
            exp_planner->enablePlanningFromScratch(true);
            exp_planner->enablePlanningFromRecall(false);
            const auto result_without_experience = exp_planner->solve(10.0);
            BOOST_CHECK(result_without_experience);
            const auto elapsed_without_db = exp_planner->getLastPlanComputationTime();

            OMPL_INFORM("solve with database");
            exp_planner->clear();
            exp_planner->enablePlanningFromScratch(false);
            exp_planner->enablePlanningFromRecall(true);
            const auto result_with_experience = exp_planner->solve(10.0);
            const auto elapsed_with_db = exp_planner->getLastPlanComputationTime();
            BOOST_CHECK(result_with_experience);

            OMPL_INFORM("with db: %f, without db: %f", elapsed_with_db, elapsed_without_db);
            if constexpr (std::is_same_v<PlannerT, ReplannerT>)
            {
                // solving with database must be faster than without database
                // at least if planner and replanner types are the same
                BOOST_CHECK_LT(elapsed_with_db, elapsed_without_db);
            }
        }
    }
};

BOOST_AUTO_TEST_CASE(LightningTest)
{
    ExperienceBasedPlannerTest<ot::Lightning, void, void>::runTest();
    ExperienceBasedPlannerTest<ot::Lightning, og::RRTConnect, og::RRTConnect>::runTest();
    ExperienceBasedPlannerTest<ot::Lightning, og::KPIECE1, og::KPIECE1>::runTest();
    ExperienceBasedPlannerTest<ot::Lightning, og::KPIECE1, og::RRTConnect>::runTest();
    ExperienceBasedPlannerTest<ot::Lightning, og::RRTConnect, og::KPIECE1>::runTest();
}

// BOOST_AUTO_TEST_CASE(ThunderTest)
// {
//   ExperienceBasedPlannerTest<ot::Thunder, void, void>::runTest();
//   ExperienceBasedPlannerTest<ot::Thunder, og::RRTConnect, og::RRTConnect>::runTest();
//   ExperienceBasedPlannerTest<ot::Thunder, og::KPIECE1, og::KPIECE1>::runTest();
//   ExperienceBasedPlannerTest<ot::Thunder, og::KPIECE1, og::RRTConnect>::runTest();
//   ExperienceBasedPlannerTest<ot::Thunder, og::RRTConnect, og::KPIECE1>::runTest();
// }
