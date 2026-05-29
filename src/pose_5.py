import numpy as np
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
    # Use the optimized pose for X(5) (from the result) when computing the bearing/range
    try:
        pose_for_measurement = result.atPose2(X(5))
    except Exception:
        # Fall back to the provided global candidate pose
        pose_for_measurement = pose_5

    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_for_measurement,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # Initialize the optimizer with the graph and initial estimate
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    
    # Perform the optimization and return the result
    result = optimizer.optimize()
    
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    min_marginals_sum = float('inf')
    
    # Try all combinations of pose options and landmarks
    for pose_key in pose_options:
        pose_5 = pose_options[pose_key]
        
        for landmark_num in [1, 2]:
            # Clone the graph and initial estimate for this iteration
            test_graph = gtsam.NonlinearFactorGraph(graph)
            test_estimate = gtsam.Values(initial_estimate)
            
            # Add the pose and measurement
            test_graph, test_estimate = add_pose(test_graph, test_estimate, pose_5)
            result = optimize(test_graph, test_estimate)
            test_graph = add_landmark_measurement(test_graph, result, pose_5, landmark_num)
            result = optimize(test_graph, result)
            
            # Calculate marginal covariances
            marginals = gtsam.Marginals(test_graph, result)

            # Trace of marginal covariance for the measured landmark (sum of variances)
            sum_of_marginals = 0.0
            for i in [1, 2]:
                sum_of_marginals += float(np.trace(marginals.marginalCovariance(L(i))))
            
            # Keep track of the best option
            if sum_of_marginals < min_marginals_sum:
                min_marginals_sum = sum_of_marginals
                best_pose = pose_key
                best_landmark = landmark_num
    
    return best_pose, best_landmark, min_marginals_sum

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    min_error_sum = float('inf')
    
    # Try all combinations of pose options and landmarks
    for pose_key in pose_options:
        pose_5 = pose_options[pose_key]
        
        for landmark_num in [1, 2]:
            # Clone the graph and initial estimate for this iteration
            test_graph = gtsam.NonlinearFactorGraph(graph)
            test_estimate = gtsam.Values(initial_estimate)
            
            # Add the pose and measurement
            test_graph, test_estimate = add_pose(test_graph, test_estimate, pose_5)
            result = optimize(test_graph, test_estimate)
            test_graph = add_landmark_measurement(test_graph, result, pose_5, landmark_num)
            result = optimize(test_graph, result)
            
            # Calculate the sum of pose errors for X(1), X(2), X(3)
            # Compare optimized result to the initial estimates (test_estimate)
            error_sum = 0.0
            for i in [1, 2, 3]:
                optimized_pose = result.atPose2(X(i))
                prior_pose = test_estimate.atPose2(X(i))
                delta = optimized_pose.localCoordinates(prior_pose)
                # use squared norm to accumulate error
                error_sum += float(np.dot(delta, delta))

            # Keep track of the best option
            if error_sum < min_error_sum:
                min_error_sum = error_sum
                best_pose = pose_key
                best_landmark = landmark_num
    
    return best_pose, best_landmark, min_error_sum