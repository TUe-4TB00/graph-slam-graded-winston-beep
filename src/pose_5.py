import numpy as np
import copy
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    return result

def minimize_marginals(graph, initial_estimate, pose_options):

    best_pose = None    
    best_landmark = None 
    best_score = float('inf') 
    sum_of_marginals = 0
    
    # Try all combinations.
    for pose_label, pose_5 in pose_options.items():
        for landmark in [1, 2]:

            graph_trial = copy.deepcopy(graph)
            estimate_trial = copy.deepcopy(initial_estimate)


            graph_trial, estimate_trial = add_pose(graph_trial, estimate_trial, pose_5)

            result_trial = optimize(graph_trial, estimate_trial)

            graph_trial = add_landmark_measurement(graph_trial, result_trial, pose_5, landmark)
  
            result_trial = optimize(graph_trial, estimate_trial)


            marginals_trial = gtsam.Marginals(graph_trial, result_trial)
            cov_l1 = marginals_trial.marginalCovariance(L(1))
            cov_l2 = marginals_trial.marginalCovariance(L(2))
            score = np.trace(cov_l1) + np.trace(cov_l2)


            if score < best_score: 
                best_score = score
                best_pose = pose_label
                best_landmark = landmark

                sum_of_marginals = cov_l1.sum() + cov_l2.sum()
    
    print(f"\nBest pose: {best_pose}, Best landmark: L({best_landmark}), score: {best_score:.6f}")
    return best_pose, best_landmark, sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    
    best_pose = None       
    best_landmark = None   
    best_score = float('inf') 
    sum_of_errors = 0

    for pose_label, pose_5 in pose_options.items():
        for landmark in [1, 2]:

            # Independent copy for each trial
            graph_trial = copy.deepcopy(graph)
            estimate_trial = copy.deepcopy(initial_estimate)

            graph_trial, estimate_trial = add_pose(graph_trial, estimate_trial, pose_5)
            result_trial = optimize(graph_trial, estimate_trial)

            graph_trial = add_landmark_measurement(graph_trial, result_trial, pose_5, landmark)
            result_trial = optimize(graph_trial, estimate_trial)


            marginals_trial = gtsam.Marginals(graph_trial, result_trial)
            cov_l1 = marginals_trial.marginalCovariance(L(1))
            cov_l2 = marginals_trial.marginalCovariance(L(2))


            score = cov_l1.sum() + cov_l2.sum()

            print(f"Pose {pose_label}, Landmark {landmark}: score = {score:.6f}")

            if score < best_score: 
                best_score = score
                best_pose = pose_label
                best_landmark = landmark
                sum_of_errors = score

                x1 = result_trial.atPose2(X(1))
                x2 = result_trial.atPose2(X(2))
                x3 = result_trial.atPose2(X(3))
                sum_of_errors = (
                    abs(x1.x()) + abs(x1.y()) + abs(x1.theta()) + 
                    abs(x2.y()) + abs(x2.theta()) +
                    abs(x3.y()) + abs(x3.theta())
                )



    return best_pose, best_landmark, sum_of_errors