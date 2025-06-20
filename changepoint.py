import gpflow
import pandas as pd
import tensorflow as tf
import datetime as dt
from gpflow.kernels import ChangePoints, Matern32
from typing import Dict, List, Optional, Tuple, Union
from tensorflow_probability import bijectors as tfb
from sklearn.preprocessing import StandardScaler

Kernel = gpflow.kernels.base.Kernel

MAX_ITERATIONS = 50

## module service
class ChangePointDetection(ChangePoints):
    def __init__(
            self,
            kernels: Tuple[Kernel, Kernel],
            location: float,
            interval: Tuple[float, float],
            steepness: float = 1.0,
            name: Optional[str] = None
    ):
        if location < interval[0] or location > interval[1]:
            raise ValueError(
                "Location {loc} is not in range [{low},{high}]".format(
                    loc=location, low=interval[0], high=interval[1]
                )
            )
        locations = [location]
        super().__init__(
            kernels = kernels, locations = locations, steepness = steepness, name=name
        )

        affine = tfb.Shift(tf.cast(interval[0], tf.float64))(
            tfb.Scale(tf.cast(interval[1] - interval[0], tf.float64))
        )

        self.locations = gpflow.Parameter(
            locations, transform=tfb.Chain([affine, tfb.Sigmoid()]), dtype = tf.float64
        )

        def _sigmoids(self, X: tf.Tensor):
            locations = tf.reshape(self.locations, (1, 1, -1))
            steepness = tf.reshape(self.steepness, (1, 1, -1))
            return tf.sigmoid(steepness * (X[:, :, None] - locations))

## fit matern kernel - là cái baseline
def fit_matern_kernel(
        time_series_data: pd.DataFrame,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        likelihood_variance: float = 1.0
):

    model = gpflow.models.GPR(
        data = (
            time_series_data.loc[:, ["X"]].to_numpy(),
            time_series_data.loc[:, ["Y"]].to_numpy()
        ),
        kernel = Matern32(variance=variance, lengthscales=lengthscale),
        noise_variance=likelihood_variance
    )

    optimizer = gpflow.optimizers.Scipy()
    nlml = optimizer.minimize(
        model.training_loss, model.trainable_variables, options=dict(maxiter=MAX_ITERATIONS)
    ).fun
    parameters = {
        "kM_variance": model.kernel.variance.numpy(),
        "kM_lengthscales": model.kernel.lengthscales.numpy(),
        "kM_likelihood_variance": model.likelihood.variance.numpy()
    }

    return nlml, parameters

## fit cp kernel - là cái improved nhờ add cp vào?
def fit_changepoint_kernel(
        time_series_data: pd.DataFrame,
        k1_variance: float = 1.0,
        k1_lengthscale: float = 1.0,
        k2_variance: float = 1.0,
        k2_lengthscale: float = 1.0,
        kC_likelihood_variance = 1.0,
        kC_changepoint_location = None,
        kC_steepness = 1.0
):
    if not kC_changepoint_location:
        kC_changepoint_location = (
            time_series_data["X"].iloc[0] + time_series_data["X"].iloc[-1]
        ) / 2.0

    model = gpflow.models.GPR(
        data=(
            time_series_data.loc[:, ["X"]].to_numpy(),
            time_series_data.loc[:, ["Y"]].to_numpy()
        ),
        kernel = ChangePointDetection(
            [
                Matern32(variance=k1_variance, lengthscales=k1_lengthscale),
                Matern32(variance=k2_variance, lengthscales=k2_lengthscale)
            ],
            location=kC_changepoint_location,
            interval=(time_series_data["X"].iloc[0], time_series_data["X"].iloc[-1]),
            steepness=kC_steepness
        )
    )
    model.likelihood.variance.assign(kC_likelihood_variance)

    optimizer = gpflow.optimizers.Scipy()
    nlml = optimizer.minimize(
        model.training_loss, model.trainable_variables, options=dict(maxiter=MAX_ITERATIONS)
    ).fun
    changepoint_location = model.kernel.locations[0].numpy()
    parameters = {
        "k1_variance": model.kernel.kernels[0].variance.numpy().flatten()[0],
        "k1_lengthscale": model.kernel.kernels[0].lengthscales.numpy().flatten()[0],
        "k2_variance": model.kernel.kernels[1].variance.numpy().flatten()[0],
        "k2_lengthscale": model.kernel.kernels[1].lengthscales.numpy().flatten()[0],
        "kC_likelihood_variance": model.likelihood.variance.numpy().flatten()[0],
        "kC_changepoint_location": changepoint_location,
        "kC_steepness": model.kernel.steepness.numpy()
    }

    return changepoint_location, nlml, parameters

## tính severity , là cái v_t, và cái cp_location_normalized, là cái \gamma_t
def changepoint_severity(
     kC_nlml: Union[float, List[float]],
     kM_nlml: Union[float, List[float]]
):
    normalized_nlml = kC_nlml - kM_nlml
    return 1 - 1 / (np.mean(np.exp(-normalized_nlml)) + 1)
def changepoint_loc_and_score(
    time_series_data_window: pd.DataFrame,
    kM_variance: float = 1.0,
    kM_lengthscale: float = 1.0,
    kM_likelihood_variance: float = 1.0,
    k1_variance: float = None,
    k1_lengthscale: float = None,
    k2_variance: float = None,
    k2_lengthscale: float = None,
    kC_likelihood_variance: float = None,
    kC_changepoint_location: float = None,
    kC_steepness=1.0
):
    time_series_data = time_series_data_window.copy()
    Y_data = time_series_data[["Y"]].values
    time_series_data[["Y"]] = StandardScaler().fit(Y_data).transform(Y_data)


    if kM_variance == kM_lengthscale == kM_likelihood_variance == 1.0 :
        (kM_nlml, kM_params) = fit_matern_kernel(time_series_data)
    else:
        (kM_nlml, kM_params) = fit_matern_kernel(time_series_data, kM_variance, kM_lengthscale, kM_likelihood_variance)

    is_cp_location_default = (
        (not kC_changepoint_location)
        or kC_changepoint_location < time_series_data["X"].iloc[0]
        or kC_changepoint_location > time_series_data["X"].iloc[-1]
    )

    if is_cp_location_default:
        kC_changepoint_location = (
            time_series_data["X"].iloc[-1] + time_series_data["X"].iloc[0]
        ) / 2.0

    if not k1_variance:
        k1_variance = kM_params["kM_variance"]

    if not k1_lengthscale:
        k1_lengthscale = kM_params["kM_lengthscales"]

    if not k2_variance:
        k2_variance = kM_params["kM_variance"]

    if not k2_lengthscale:
        k2_lengthscale = kM_params["kM_lengthscales"]

    if not kC_likelihood_variance:
        kC_likelihood_variance = kM_params["kM_likelihood_variance"]


    if (k1_variance == k1_lengthscale == k2_variance == k2_lengthscale == kC_likelihood_variance == kC_steepness == 1.0) and is_cp_location_default:
        (changepoint_location, kC_nlml, kC_params) = fit_changepoint_kernel(time_series_data)
    else:
        (changepoint_location, kC_nlml, kC_params) = fit_changepoint_kernel(
            time_series_data,
            k1_variance=k1_variance,
            k1_lengthscale=k1_lengthscale,
            k2_variance=k2_variance,
            k2_lengthscale=k2_lengthscale,
            kC_likelihood_variance=kC_likelihood_variance,
            kC_changepoint_location=kC_changepoint_location,
            kC_steepness=kC_steepness,
        )

    cp_score = changepoint_severity(kC_nlml, kM_nlml)
    cp_loc_normalised = (time_series_data["X"].iloc[-1] - changepoint_location) / (
        time_series_data["X"].iloc[-1] - time_series_data["X"].iloc[0]
    )

    return cp_score, changepoint_location, cp_loc_normalised, kM_params, kC_params

## run thuật toán
def run_CPD(
    time_series_data: pd.DataFrame,
    lookback_window_length: int,
    start_date: dt.datetime = None,
    end_date: dt.datetime = None,
    use_kM_hyp_to_initialize_kC=True
):
    if start_date and end_date:
        print(time_series_data.loc[:start_date])
        first_window = time_series_data.loc[:start_date].iloc[
             -(lookback_window_length + 1) :, :
         ]

        # first_window = time_series_data.loc[:start_date].iloc[
        #    -(lookback_window_length + 1) :
        # ]
        remaining_data = time_series_data.loc[start_date:end_date, :]
        # remaining_data = time_series_data.loc[start_date:end_date]
        if remaining_data.index[0] == start_date:
            remaining_data = remaining_data.iloc[1:]
        else:
            first_window = first_window.iloc[1:]
        time_series_data = pd.concat([first_window, remaining_data]).copy()
    else:
        raise Exception("Pass start and end date.")

    time_series_data["Date"] = time_series_data.index
    time_series_data = time_series_data.reset_index(drop=True)

    results = []
    for window_end in range(lookback_window_length + 1, len(time_series_data)):
        ts_data_window = time_series_data.iloc[
            window_end - (lookback_window_length + 1) : window_end
        ][["Date", "daily_return"]].copy()
        ts_data_window["X"] = ts_data_window.index.astype(float)
        ts_data_window = ts_data_window.rename(columns={"daily_return": "Y"})
        time_index = window_end - 1
        window_date = ts_data_window["Date"].iloc[-1].strftime("%Y-%m-%d")

        if use_kM_hyp_to_initialize_kC:
            cp_score, cp_loc, cp_loc_normalised, _, _ = changepoint_loc_and_score(ts_data_window)
        else:
            cp_score, cp_loc, cp_loc_normalised, _, _ = changepoint_loc_and_score(
                    ts_data_window,
                    k1_lengthscale=1.0,
                    k1_variance=1.0,
                    k2_lengthscale=1.0,
                    k2_variance=1.0,
                    kC_likelihood_variance=1.0,
                )
        # results.append([window_date, time_index, cp_loc, cp_loc_normalised, cp_score])
        results.append([window_date, cp_loc_normalised, cp_score])
    #results_df = pd.DataFrame(results, columns=["date", "t", "cp_location", "cp_location_norm", "cp_score"])
    results_df = pd.DataFrame(results, columns = ['date', 'cp_location_norm', 'cp_score'])
    results_df.set_index('date')
    return results_df