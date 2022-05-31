import sys
import warnings
import random
import numpy as np
import scipy as sp
import scipy.optimize
import scipy.signal
import pandas as pd
from audiot.signal_processing.wiener_filter import WienerFilter
from audiot.signal_processing.noise_estimation import MedianFilterNoiseEstimator
from audiot.signal_processing.auto_segmenter import AutoSegmenter
from audiot.spectral_features import get_row_resizing_matrix


def reduce_noise(audio_signal):
    """
    Attempts to reduce background noise in the signal without causing noticeable perceptual 
    artifacts in how the signal sounds.
    """
    WienerFilter.get_perceptually_pleasant_noise_reduction_filter().filter_signal(audio_signal)


def reduce_noise_very_aggressively(audio_signal):
    """
    Aggressively reduces noise in the signal, zeroing out most of the signal so that only sounds
    that stick out remain.  Will likely cause very noticeable noise reduction artifacts in how the
    audio sounds.
    """
    return WienerFilter.get_very_aggressive_noise_reduction_filter().filter_signal(audio_signal)


def segment_signal(audio_signal):
    """
    Segment out sounds that are louder than their surroundings in the signal using default 
    segmentation settings that seem to work well across a wide variety of data.
    """
    return AutoSegmenter.get_default_segmenter().segment_signal(audio_signal)


def calc_signal_strength_features(audio_signal):
    """
    Compute signal strength features (estimate of how much of the energy is from a foreground versus
    a background sound).

    Returns:
        SpectralFeatures: A SpectralFeatures object containing the computed features.
    """
    wiener_filter = WienerFilter.get_very_aggressive_noise_reduction_filter()
    return wiener_filter.get_signal_strength_features(audio_signal)


def compute_pitch_upsweep_downsweep(pitch_track_frequencies):
    total_upsweep = np.maximum(0, np.diff(pitch_track_frequencies)).sum()
    total_downsweep = np.maximum(0, (-np.diff(pitch_track_frequencies))).sum()
    return (total_upsweep, total_downsweep)


def compute_chirp_features_for_segments(
    signal_strength_features, segment_index_list, n_time_divisions=3
):
    """
    Compute segment-level features Brandon picked to try to represent / detect chirps in a way that
    is invariant to the pitch of the chirps and return them in a DataFrame (which also includes the
    segment information with the features).

    These features are computed by taking various statistics vertically along the columns of the 
    signal strength features, then dividing them into n_time_divisions chunks and averaging them
    over each chunk of time within the segment.  The statistics include the mean, variance, 
    skewness, kurtosis, support (number of non-zero values), max, and the mean divided by the max of
    each column.  The goal was to be able to distinguish columns that have a lot of energy in just a
    few frequency bands (like would be expected for a chirp) from columns that have energy more 
    evenly distributed across many frequency bands.

    Args:
        signal_strength_features (SpectralFeatures): The signal strength features for the recording,
            as computed by the calc_signal_strength_features() function above.
        segment_index_list (tuple): The segment information for the recording, obtained by running
            AutoSegmenter.segment_signal_strength_features() on the signal_strength_features.  Note
            that the segments start / end values should be indexes, and not time values.
        n_time_divisions (int): The number of time chunks to divide each segment into and compute
            statistics over.
    """
    # Feature info
    feature_names = [
        "mean",
        "variance",
        "skewness",
        "kurtosis",
        "support",
        "max",
        "mean_over_max",
    ]
    # Assign unique index values to each feature name
    feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    n_feature_types = len(feature_name_to_idx)
    feature_column_names = [f"{fn}_{t}" for fn in feature_names for t in range(n_time_divisions)]
    column_names = feature_column_names + [
        "signal_strength",
        "segment_duration",
        "segment_start",
        "segment_end",
    ]

    # Compute features for each segment
    segment_time_list = AutoSegmenter.convert_segments_from_indexes_to_seconds(
        segment_index_list, signal_strength_features
    )
    segment_features_list = []
    for segment_index, segment_time in zip(segment_index_list, segment_time_list):
        segment_start_idx, segment_end_idx, _ = segment_index
        segment_start_seconds, segment_end_seconds, _ = segment_time
        segment_duration = segment_end_seconds - segment_start_seconds
        if segment_start_idx >= segment_end_idx:
            raise RuntimeError("Zero length segment found")

        # Vertical features
        seg_feat = signal_strength_features.features[:, segment_start_idx:segment_end_idx]
        column_features = np.zeros((n_feature_types, seg_feat.shape[1]))
        column_features[feature_name_to_idx["mean"], :] = np.mean(seg_feat, axis=0)
        column_features[feature_name_to_idx["variance"], :] = np.var(seg_feat, axis=0)
        column_features[feature_name_to_idx["skewness"], :] = sp.stats.skew(seg_feat, axis=0)
        column_features[feature_name_to_idx["kurtosis"], :] = sp.stats.kurtosis(seg_feat, axis=0)
        column_features[feature_name_to_idx["support"], :] = np.sum(seg_feat > 0, axis=0)
        column_features[feature_name_to_idx["max"], :] = np.max(seg_feat, axis=0)
        # Avoid NaNs by using np.maximum() to replace zeros in the denominator with epsilon
        column_features[feature_name_to_idx["mean_over_max"], :] = np.mean(
            seg_feat, axis=0
        ) / np.maximum(np.max(seg_feat, axis=0), sys.float_info.epsilon)

        # Average the features over each time division
        resized_vertical_features = column_features.dot(
            get_row_resizing_matrix(seg_feat.shape[1], n_time_divisions).transpose()
        )

        # Other features
        signal_strength = np.mean(np.max(seg_feat, axis=0))

        # Assemble features into one array
        segment_features = np.hstack(
            (
                resized_vertical_features.ravel(),
                [signal_strength, segment_duration, segment_start_seconds, segment_end_seconds],
            )
        )
        segment_features_list.append(segment_features)
    features_array = np.vstack(segment_features_list)
    features_df = pd.DataFrame(features_array, columns=column_names)
    return features_df


def gaussian(x, mean, standard_deviation, scalar):
    """
    Computes the values of a Guassian curve at the locations specified by x.
    """
    return scalar * np.exp(-np.square(x - mean) / (2 * np.square(standard_deviation)))


def fit_gaussian_to_histogram_using_ransac(
    bin_centers,
    bin_counts,
    n_ransac_iterations=100,
    overshoot_penalty_multiplier=0.3,
    undershoot_penalty_multiplier=0.075,
    outside_range_penalty_multiplier=0.15,
    percent_range_outside_histogram_bounds_to_penalize=0.25,
    bin_width=None,
):
    """
    Finds a Gaussian curve that tries to fit as many of the points in the histogram as possible,
    while ignoring outlier points that don't fit well.  Multiple curve candidates curve fits are 
    generated using the methodology of the RANSAC algorithm, and the one with the best fit score is 
    returned.

    The scoring for the candidate fit curves works as follows:
        - Inlier histogram points lying directly on the fit curve receive a positive score equal to 
            the bin_count value for that histogram bin (so bins containing more samples receive
            higher scores).  Inlier points not lying directly on the fit curve get this score 
            discounted linearly as they deviate further away from the fit curve, with the score 
            being discounted to zero at the boundary past which the point would no longer be 
            considered an inlier.
        - Each outlier histogram point penalizes the score proportional to the distance by which the
            fit curve overshot or undershot that point (with the proportionality constant set by
            the penatly multiplier parameters).  By default, overshooting a point is penalized much
            more than undershooting.  The reasoning behind this is that it makes sense to ignore 
            spurious peaks in the data that could be caused by false positives, but it doesn't make
            sense to introduce big peaks that aren't there in the histogram (e.g. when we fit the
            tails of the gaussian without regard to whether the peak actually matches any of the
            data).
        - To discourage fitting gaussians that peak outside the range of the histogram, we also 
            penalize non-zero values in the fit curve outside that range on both sides.

    Args:
        bin_centers (ndarray): The center (x-axis) values for each bin in the histogram.
        bin_counts (ndarray): Numbers representing how many things fell into each histogram bin 
            relative to all the other bins (this can be counts, a percentages, etc).
        n_ransac_iterations (int): The number of RANSAC iterations to apply when searching for the
            best fitting model.
        overshoot_penalty_multiplier (float): Multiplier for the score penalty a candidate fit curve
            receives for overshooting outlier points.  
        undershoot_penalty_multiplier (float): Multiplier for the score penalty a candidate fit 
            curve receives for undershooting outlier points.
        outside_range_penalty_multiplier (float): Multiplier for the score penalty a candidate fit
            curve receives for having energy outside the range of the original histogram.
        percent_range_outside_histogram_bounds_to_penalize (float): The distance to either side 
            outside the range of the original histogram over which non-zero energy in the candidate
            fit curves should be penalized, as a percentage of the total range of the histogram.
            For example, with a value of 0.25 and a histogram covering a frequency range of 0 to 
            8000 Hz, candidate fit curves would be penalized for having energy in the ranges 
            -2000 to 0 Hz and 8000 to 10000 Hz (25% of the total range on each side of the bounds of
            the original histogram).
        bin_width (float): The distance between bin centers.  If not specified, it is computed using
            the first two bin centers.  This value is used to construct bin locations outside the
            bounds of the original histogram (on both sides), which are in turn used to penalize 
            any non-zero energy that fit curve candidates have outside the original histogram range.
    """
    # Parameters
    bin_centers = np.asarray(bin_centers)
    bin_counts = np.asarray(bin_counts)
    n_bins = len(bin_centers)
    if bin_width is None:
        bin_width = bin_centers[1] - bin_centers[0]
    # The number of samples used to estimate each initial model for each RANSAC iteration.  Usually,
    # this is set to the minimum number of samples needed to fit that model type, which would be 3
    # in the case of a Guassian.  But I think I was getting better results (models that were more
    # likely to match one of the modes in the histogram) by using 5.  I think there's too much
    # variance / chaos in the curves that get fit with just 3, which makes it less likely to find
    # curves that actually have a decent number of other inlier points in the histogram.  Adding a
    # couple more points into the initial estimate helps constrain it a little bit more toward
    # something that is more likely to be in line with other samples in the histogram, while still
    # being small enough that hopefully it should still pick 5 non-outlier points often enough to
    # work.
    n_samples_to_estimate_model = 5

    # Set up bin center locations outside the range of the original histogram.  These will be used
    # to penalize area under the fit curve that falls outside the histogram range on either side.
    histogram_range = (bin_centers[-1] - bin_centers[0]) + bin_width
    range_outside_histogram_to_penalize = (
        histogram_range * percent_range_outside_histogram_bounds_to_penalize
    )
    bin_centers_outside_histogram_range = np.hstack(
        [
            np.arange(
                bin_centers[0], bin_centers[0] - range_outside_histogram_to_penalize, -bin_width,
            )
            - bin_width,
            np.arange(
                bin_centers[-1], bin_centers[0] + range_outside_histogram_to_penalize, bin_width,
            )
            + bin_width,
        ]
    )

    # Define internal helper functions
    def fit_gaussian(x_values, y_values):
        if np.all(y_values <= 0):
            # Degenerate case.  Any gaussian with a scalar of 0 will be all zeros, so we just set
            # mean=0 and std=1.
            return 0, 1, 0
        # Compute initial mean / standard_deviation / scalar values for the curve_fit function to
        # start from.  Note that curve_fit tries to fit a curve that passes as close to all the
        # points as possible, which is different from computing the sample mean an covarance from
        # the histogram.  But it generally should still provide a reasonable starting point for
        # doing the curve fitting.
        probabilities = y_values / y_values.sum()
        initial_mean = (x_values * probabilities).sum()
        # Avoid setting the initial std to zero to avoid divide by zero errors (e.g. in the case
        # where only one bin has a non-zero count).
        initial_standard_deviation = max(
            1e-10, np.sqrt(np.sum(probabilities * (x_values - initial_mean) ** 2))
        )
        # Now that we've selected an initial mean and std, compute the average scale factor to
        # make the gaussian pass through those points as closely as possible.
        unscaled = gaussian(x_values, initial_mean, initial_standard_deviation, 1)
        unscaled[unscaled < 1e-10] = 1e-10  # Avoid divisions by zero in the next step
        initial_scalar = (y_values / unscaled).mean()
        # Supress optimization warnings and run the optimization
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=scipy.optimize.OptimizeWarning)
            popt, pcov = scipy.optimize.curve_fit(
                f=gaussian,
                xdata=x_values,
                ydata=y_values,
                # TODO: Try removing the sigma parameter below. I'm not sure if it affects things much or not (I had forgotten I had set it and haven't really experimented with it much).
                # Set uncertainty higher for small y_values to weight them less in the curve fitting.
                # sigma=1.0 / ( y_values + 0.01),
                p0=[initial_mean, initial_standard_deviation, initial_scalar],
            )
        return popt

    def score_curve_fit(curve_mean, curve_std, curve_scalar):
        curve = gaussian(bin_centers, curve_mean, curve_std, curve_scalar)
        # Set an inlier distance threshold 30% above / below the fit curve, with some additional
        # wiggle room for small values.
        # Set wiggle room as a percentage of the average bin count
        wiggle_room = 0.05 * bin_counts.mean()
        inlier_distance_threshold = 0.3 * curve + wiggle_room
        inlier_distances = np.abs(bin_counts - curve)
        inlier_mask = inlier_distances < inlier_distance_threshold
        # Compute weights for the inliers, where points directly on the curve get a weight of 1.0,
        # and that weight linearly decays to zero as the point moves toward the inlier threshold.
        inlier_weight = 1 - inlier_distances / inlier_distance_threshold
        inlier_reward = (bin_counts[inlier_mask] * inlier_weight[inlier_mask]).sum()
        overshoot_penalty = (
            overshoot_penalty_multiplier
            * np.maximum(0, curve[~inlier_mask] - bin_counts[~inlier_mask]).sum()
        )
        undershoot_penalty = (
            undershoot_penalty_multiplier
            * np.maximum(0, bin_counts[~inlier_mask] - curve[~inlier_mask]).sum()
        )
        curve_energy_outside_histogram_range = np.abs(
            gaussian(bin_centers_outside_histogram_range, mean, std, scalar)
        ).sum()
        outside_range_penalty = (
            outside_range_penalty_multiplier * curve_energy_outside_histogram_range
        )
        curve_fit_score = (
            inlier_reward - overshoot_penalty - undershoot_penalty - outside_range_penalty
        )
        return curve_fit_score, inlier_mask

    # Set up variables to track the best curve fit and its score
    best_curve_fit_score = -np.inf
    best_mean = 0
    best_std = 0
    best_scalar = 0
    for iter in range(n_ransac_iterations):
        # Select a few random samples and fit a curve
        selected_indexes = random.sample(range(n_bins), n_samples_to_estimate_model)
        selected_x = bin_centers[selected_indexes]
        selected_y = bin_counts[selected_indexes]
        try:
            mean, std, scalar = fit_gaussian(selected_x, selected_y)
        except RuntimeError:
            # Skip iterations where the curve fitting failed to fit a curve (their score will
            # be zero).
            continue
        score, inlier_mask = score_curve_fit(mean, std, scalar)
        if score > best_curve_fit_score:
            best_curve_fit_score = score
            best_mean = mean
            best_std = std
            best_scalar = scalar

        # Skip the refinement step if there are no more inliers than what we randomly selected to
        # start with.
        if inlier_mask.sum() <= n_samples_to_estimate_model:
            continue

        # Fit a refined curve using all the inliers for the above curve
        try:
            refined_mean, refined_std, refined_scalar = fit_gaussian(
                bin_centers[inlier_mask], bin_counts[inlier_mask]
            )
        except RuntimeError:
            # Abort if the curve fitting failed (we'll just keep the inital fit curve in this case).
            continue
        refined_score, refined_inlier_mask = score_curve_fit(
            refined_mean, refined_std, refined_scalar
        )
        # Only update the fit if the refined curve scores better than the initial one.
        if refined_score > best_curve_fit_score:
            best_curve_fit_score = refined_score
            best_mean = refined_mean
            best_std = refined_std
            best_scalar = refined_scalar
    # Return the best curve fit parameters
    return best_mean, best_std, best_scalar


def estimate_pitch_mean_and_std_from_histograms(pitch_histogram_matrix, frequency_axis):
    """
    Fits gaussian curves to a matrix of pitch histograms (with each column representing a histogram)
    and returns a list of the mean pitches and a list of the pitch standard deviations from those
    curve fits.

    Args:
        pitch_histogram_matrix (ndarray): A 2d ndarray where each column of the matrix is a 
            histogram of pitch values.
        frequency_axis (ndarray): A 1d ndarray specifying the frequency associated with each bin 
            of the histograms.
    
    Returns:
        (pitch_means, pitch_stds):  Two 1d ndarrays.  The first lists the mean pitch of each fitted
            gaussian curve, while the second lists their standard deviation values.
    """
    gaussian_curve_params = [
        fit_gaussian_to_histogram_using_ransac(frequency_axis, pitch_histogram_matrix[:, idx])
        for idx in range(pitch_histogram_matrix.shape[1])
    ]
    pitch, pitch_std, _ = zip(*gaussian_curve_params)
    return pitch, pitch_std



