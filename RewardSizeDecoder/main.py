from sklearn.model_selection import StratifiedKFold, KFold
from data_preprocessing.prepare_datasets import load_clean_align_data, get_t_slice_video
from data_preprocessing.resample_data import random_undersample, no_resample, predict_ensemble_proba, predict_ensemble, random_undersample_and_smote_oversample
from sklearn.metrics import average_precision_score
from models.LinearDiscriminantAnalysis import LDA
from models.LogisticRegression import LogisticRegressionModel
from models.SupportVectorMechine import SVM
from decoder_utils.decoder_utils import (compute_eval_metrics, compute_roc, save_parameters, _safe_folder_name,
                           confusion_indexes, separate_video_activity, binarize_target)
from decoder_utils.logging_tools import make_console_logger, attach_file_handler
import logging
from plot_results import plot_results
from sklearn.preprocessing import StandardScaler
import os
import pickle
from pathlib import Path
import numbers
import time
import numpy as np


class RewardSizeDecoder:
    def __init__(self, *,
                 subject_id,
                 session,
                 num_features,
                 frame_rate,
                 time_bin,
                 original_video_path,
                 model,
                 user_model_params,
                 resample_method,
                 dj_info,
                 save_folder_name,
                 save_video_folder,
                 handle_omission,
                 clean_ignore,
                 find_parameters=True,
                 verbose=False
                 ):

        self.subject_id = subject_id
        self.session = session
        self.num_features = num_features
        self.frame_rate = frame_rate
        self.time_bin = time_bin
        self.original_video_path = original_video_path
        self.model = model
        self.user_model_params = user_model_params
        self.resample_method = resample_method
        self.dj_info = dj_info
        self.save_folder_name = save_folder_name
        self.save_video_folder = save_video_folder
        self.handle_omission = handle_omission
        self.clean_ignore = clean_ignore
        self.find_parameters = find_parameters
        self.saveroot = None
        self.saveroot_video = None

        # base logger (handlers & formatting)
        base_name = "RewardSizeDecoder"
        self._logger = make_console_logger(base_name, verbose=verbose)

        # adapter with per-instance context
        self.log = logging.LoggerAdapter(
            self._logger,
            {"subject_id": self.subject_id, "session": self.session, "model": self.model},
        )
        self.log.debug("Initialized decoder")

    # ---------- validation of user parameters----------
    def validate_params(self, supported_models=None, supported_resampling=None):
        """
        Validate user-provided parameters. Optionally try a DataJoint connection
        with the provided dj_info to ensure credentials are correct.
        Raises ValueError on any problem.
        """
        self.log.debug("Validating parameters...")
        errors = []

        # subject/session
        if self.subject_id is None or (isinstance(self.subject_id, str) and not self.subject_id.strip()):
            errors.append("subject_id must be non-empty (int/str).")
        if self.session is None or (isinstance(self.session, str) and not self.session.strip()):
            errors.append("session must be non-empty (int/str).")

        # frame_rate
        if not isinstance(self.frame_rate, numbers.Real) or self.frame_rate <= 0:
            errors.append("frame_rate must be a positive number.")

        # time_bin must be a tuple/list of TWO numeric values indicating a range
        if not isinstance(self.time_bin, (list, tuple)) or len(self.time_bin) != 2:
            errors.append("time_bin must be a list/tuple of exactly two numeric values: [start, end].")
        else:
            start, end = self.time_bin
            if not (isinstance(start, numbers.Real) and isinstance(end, numbers.Real)):
                errors.append("time_bin values must be numeric.")
            elif not (start < end):
                errors.append(f"time_bin must satisfy start < end; got {self.time_bin}.")

        # model / params
        if not isinstance(self.model, str) or not self.model.strip():
            errors.append("model must be a non-empty string.")
        if supported_models is not None and self.model not in supported_models:
            errors.append(f"model '{self.model}' not in supported_models: {supported_models}")
        if not isinstance(self.user_model_params, dict):
            errors.append("user_model_params must be a dict of hyperparameters.")

        # resample method
        if self.resample_method not in supported_resampling:
            errors.append(f"resample_method '{self.resample_method}' must be one of {supported_resampling}.")

        # handle_omission
        if not isinstance(self.handle_omission, str) or not self.handle_omission.strip():
            errors.append("model must be a non-empty string.")
        if self.handle_omission not in ['keep', 'clean', 'convert']:
            errors.append('Invalid handle_omission, value should be "keep" or "clean" or "convert"')

        # clean ignore
        if not isinstance(self.clean_ignore, (bool, int)):
            errors.append("clean_ignore must be bool.")

        # sanitize folder name (fallback to 'results' if blank/invalid)
        if not isinstance(self.save_folder_name, str) or not self.save_folder_name.strip():
            self.save_folder_name = "my_run"
        self.save_folder_name = _safe_folder_name(self.save_folder_name)

        # sanitize video folder name (fallback to 'results' if blank/invalid)
        if not isinstance(self.save_video_folder, str) or not self.save_video_folder.strip():
            self.save_video_folder = "processed video"
        self.save_video_folder = _safe_folder_name(self.save_video_folder)

        # dj_info must be a dict with required string-like keys
        dj_required = ("host_path", "user_name", "password")
        if not isinstance(self.dj_info, dict):
            errors.append("dj_info must be a dict with keys: 'host_path', 'user_name', 'password'.")
        else:
            missing = [k for k in dj_required if k not in self.dj_info]
            if missing:
                errors.append(f"dj_info missing required keys: {missing}")
            else:
                bad_types = [k for k in dj_required if
                             not isinstance(self.dj_info.get(k), str) or not self.dj_info[k].strip()]
                if bad_types:
                    errors.append(f"dj_info values must be non-empty strings for keys: {bad_types}")

        if errors:
            raise ValueError("Parameter validation failed:\n- " + "\n- ".join(errors))

        self.log.info("Parameter validation passed.")
        return True

        # ---------- saveroot for results----------

    def define_saveroot(self, reference_path: str = None,
                        ensure_exists: bool = True,
                        results_folder_name: str = "results",
                        log_to_file: bool = False
                        ):
        """
        Define self.saveroot as <base>/<results_folder_name>/<save_folder_name>.

        - If reference_path is a FILE path, base = its parent directory.
        - If reference_path is a DIRECTORY, base = that directory.
        - If reference_path is None, try the directory of this module (__file__),
          if unavailable, fall back to the current working directory.
        """
        # Choose base directory
        if reference_path is not None:
            p = Path(reference_path).expanduser().resolve()
            base = p if p.is_dir() else p.parent.parent
        else:
            try:
                base = Path(__file__).resolve().parent.parent
            except NameError:
                base = Path.cwd().resolve().parent

        # Build saveroot path
        safe_leaf = _safe_folder_name(self.save_folder_name)
        saveroot = base / results_folder_name / safe_leaf

        # Build saveroot video path
        safe_leaf_video = _safe_folder_name(self.save_video_folder)
        saveroot_video = base / results_folder_name / safe_leaf_video

        # Create directories if requested
        if ensure_exists:
            saveroot.mkdir(parents=True, exist_ok=True)
            saveroot_video.mkdir(parents=True, exist_ok=True)

        self.saveroot = str(saveroot)
        self.saveroot_video = str(saveroot_video)

        if log_to_file:
            # attach file handler once we know where to write
            logfile = attach_file_handler(self._logger, Path(self.saveroot) / "logs")
            self.log.info(f"saveroot defined at: {self.saveroot}")
            self.log.info(f"file logging -> {logfile}")

    def save_user_parameters(self, fmt=("json", "excel"), filename_stem="decoder_params"):
        """Save parameters as JSON and/or Excel"""
        save_parameters(self, fmt, filename_stem)
        print("Saved model parameters")

    # ---------- Reward Size Decoder----------

    def decoder(self):

        resample_dict = {
            'No resample': no_resample,
            'simple undersample': random_undersample,
            'combine undersample(random) and oversample(SMOTE)': random_undersample_and_smote_oversample,
        }

        model_dict = {
            'LDA': LDA,
            'SVM': SVM,
            'LR': LogisticRegressionModel
        }
        t0 = time.perf_counter()
        self.log.debug('start load_clean_align_data')

        # preprocess data - load from dj, clean and align all data sets

        #start_trials, reward_labels, neural_indexes, video_features = (
            #load_clean_align_data(self.subject_id, self.session, self.num_features, self.frame_rate, self.time_bin, self.original_video_path, self.dj_info, self.saveroot, self.log, self.handle_omission, self.clean_ignore))

        start_trials, reward_labels, video_features = (
            load_clean_align_data(
                self.subject_id,
                self.session,
                self.num_features,
                self.frame_rate,
                self.time_bin,
                self.original_video_path,
                self.dj_info,
                self.saveroot,
                self.saveroot_video,
                self.log,
                self.handle_omission,
                self.clean_ignore
            )
        )

        import numpy as np
        self.log.info('finish load_clean_align_data')
        fr = int(np.floor(self.frame_rate))
        frames_bin = list(range(self.time_bin[0] * fr, self.time_bin[1] * fr + 1))

        # initialized empty outputs
        all_frames_scores = {}
        all_frames_roc = {}
        all_frames_pr_auc = {}
        all_frames_confusion = {}
        all_frames_pc_separated = {}
        all_frames_best_params = {}
        bin_frames_dic = {}
        frames_plot = []

        self.log.debug('start model training on all frames')
        self.log.info('start model training on all frames')
        missing_vid_dic = {}

        for frame_idx, frame_time in enumerate(frames_bin):
            trials_len = len(reward_labels)  # number of all trials within session
            #data_video, data_reward = get_t_slice_video(start_trials, frame_idx, video_features, neural_indexes, reward_labels)
            t_trials = (start_trials + frame_idx).astype(int)
            t_trials = np.array([t if 0 <= t < len(video_features) else float('nan') for t in t_trials])
            valid_mask =~np.isnan(t_trials)
            valid_indices = t_trials[valid_mask].astype(int)
            data_video = video_features[valid_indices]
            data_reward = reward_labels[valid_mask]
            folds_eval_scores = {}
            clf_trial_idx = {}
            folds_params = {}

            full_y_true = np.empty_like(data_reward, dtype=np.int64)
            full_y_pred = np.empty_like(data_reward, dtype=np.int64)
            full_y_probs = np.empty((len(data_reward)), dtype=np.float64)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            leftover_trials_len = len(data_reward)  # number of trials with video matching the neural frame
            missing_video_trials = round(1- leftover_trials_len/trials_len, 2)
            bin_frames_dic[frame_time] = missing_video_trials

            # encode target array
            data_reward = binarize_target(data_reward, 'large')
            large_per = round(np.average(data_reward),2) if data_reward.size > 0 else 0

            if missing_video_trials > 0.3 or large_per < 0.05:
                self.log.info(f'in frame idx {frame_idx}, time frame {frame_time}, there are more than {missing_video_trials} trials frames with missing video and {large_per} large trials')
                continue
            frames_plot.append(frame_time)

            # from imblearn.pipeline import make_pipeline
            # from imblearn.over_sampling import SMOTE
            # from imblearn.under_sampling import RandomUnderSampler
            # from sklearn.preprocessing import StandardScaler
            # from sklearn.linear_model import LogisticRegression
            # from sklearn.model_selection import LeaveOneOut, cross_validate
            # from sklearn.metrics import recall_score, confusion_matrix
            # import numpy as np
            #
            # scoring = {
            # 'accuracy': 'accuracy',
            # 'recall': 'recall',
            # 'auc': 'roc_auc',
            # 'f1': 'f1'
            # }
            #
            # pipe = make_pipeline(
            #     StandardScaler(),
            #     SMOTE(sampling_strategy=0.4, k_neighbors=1, random_state=42),
            #     RandomUnderSampler(sampling_strategy=0.6, random_state=42),
            #     LogisticRegression(class_weight={0: 35, 1: 65})
            # )
            # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            # scores = cross_validate(
            # pipe,
            # data_video,
            # data_reward,
            # cv=skf,
            # scoring=scoring,
            # return_train_score=False
            # )
            #
            # accuracy = scores['test_accuracy']
            # recall = scores['test_recall']
            # auc = scores['test_auc']
            # f1 = scores['test_f1']
            #
            # print('------------------------------------------------------------')
            # print(f'subject_id: {self.subject_id} | session: {self.session} | frame time: {frame_time}')
            # print(f'accuracy: {accuracy} | recall: {recall} | f1: {f1} | auc: {auc}')
            # print(f'missing video percentage: {missing_video_trials}')
            # print('------------------------------------------------------------')


            # splits the data while conserving data distribution in each fold
            for train_index, test_index in skf.split(data_video, data_reward):
                X_train, X_test = data_video[train_index], data_video[test_index]
                y_train, y_test = data_reward[train_index], data_reward[test_index]

                # Division of data to folds prior to resample methods (prevent data leakage)
                # Apply resample on train data
                # train and test model
                if self.resample_method == 'undersample and ensemble':
                    n_models = 5  # the number of different classifiers that will be trained
                    models = []
                    X_train_us, y_train_us = random_undersample(X_train, y_train)

                    for i in range(n_models):
                        clf = model_dict[self.model](**user_model_params[self.model])
                        if self.model in ['SVM', 'LR']:
                            clf.find_best_params(X_train_us, y_train_us)  # tune hyperparameters

                        # normalize X data before training on the full train data
                        X_scaler = StandardScaler()
                        X_train_us = X_scaler.fit_transform(X_train)
                        X_test = X_scaler.transform(X_test)

                        clf.train(X_train_us, y_train_us)
                        models.append(clf)
                    y_pred = predict_ensemble(models, X_test)
                    y_proba = predict_ensemble_proba(models, X_test)
                    eval_scores = compute_eval_metrics(y_pred, y_test)

                else:
                    X_train_us, y_train_us = resample_dict[self.resample_method](X_train, y_train)
                    clf = model_dict[self.model](**user_model_params[self.model])
                    if self.model in ['SVM', 'LR'] and self.find_parameters:
                        clf.find_best_params(X_train_us, y_train_us)
                        best_params = clf.get_best_params()
                        folds_params = {k: folds_params.get(k, []) + [v] for k, v in best_params.items()}

                    # normalize X data before training on the full train data
                    X_scaler = StandardScaler()
                    X_train_us = X_scaler.fit_transform(X_train_us)
                    X_test = X_scaler.transform(X_test)

                    self.log.info('train on full training dataset')
                    clf.train(X_train_us, y_train_us)
                    y_pred = clf.predict(X_test)
                    y_proba = clf.predict_proba(X_test)
                    eval_scores = clf.compute_metrics(y_test)

                full_y_true[test_index] = y_test
                full_y_pred[test_index] = y_pred
                full_y_probs[test_index] = y_proba
                folds_eval_scores = {k: folds_eval_scores.get(k, []) + [v] for k, v in eval_scores.items()}
                # compute tn, fp, fn, tp trials indexes
                clf_indexes = confusion_indexes(y_test, y_pred)
                clf_trial_idx = {k: clf_trial_idx.get(k, []) + [test_index[i] for i in v] for k, v in clf_indexes.items()}

            full_roc_auc = compute_roc(full_y_probs, full_y_true)
            full_pr_auc = average_precision_score(full_y_probs, full_y_true)


            # compute principal component k at t time point for tn, fp, fn, tp trials
            clf_indexes = confusion_indexes(full_y_true, full_y_pred)
            k_pc = 0
            separated_video = separate_video_activity(data_video[:, k_pc], clf_indexes)

            all_frames_scores[frame_time] = folds_eval_scores
            all_frames_roc[frame_time] = full_roc_auc
            all_frames_pr_auc[frame_time] = full_pr_auc
            all_frames_confusion[frame_time] = clf_trial_idx
            all_frames_pc_separated[frame_time] = separated_video
            all_frames_best_params[frame_time] = folds_params

        self.log.info('finish model training and testing on all frames')
        # save all nested dicts in results directory
        savedir = os.path.join(self.saveroot, f'Decoder {self.model} output', f'{self.subject_id}', f'session{self.session}')
        os.makedirs(savedir, exist_ok=True)
        to_dump = {
            "scores.pkl": all_frames_scores,
            "roc.pkl": all_frames_roc,
            "pr_auc.pkl": all_frames_pr_auc,
            "trial_idx_separated.pkl": all_frames_confusion,
            "hyperparameters.pklq": all_frames_best_params,
        }
        for fname, obj in to_dump.items():
            fpath = os.path.join(savedir, fname)
            with open(fpath, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        import numpy as np
        plot_results(self.saveroot, all_frames_scores, all_frames_roc, all_frames_pr_auc, self.subject_id, self.session, self.model, frames_plot, int(np.floor(self.frame_rate)))
        self.log.info("Decoder finished in %.3fs", time.perf_counter() - t0)
        return bin_frames_dic







if __name__ == '__main__':

    supported_resampling = ['No resample', 'combine undersample(random) and oversample(SMOTE)', 'simple undersample', 'undersample and ensemble']
    supported_models = ['LDA', 'SVM', 'LR']
    user_model_params = {'LDA': {}, 'SVM': {'probability': True}, 'LR': {}}
    host = "arseny-lab.cmte3q4ziyvy.il-central-1.rds.amazonaws.com"
    user = 'ShaniE'
    password = 'opala'
    dj_info = {'host_path': host, 'user_name': user, 'password': password}
    video_frame_rates = [10] #[2, 5, 10, 20, 50]
    subject_lst = [464724, 464725, 463189, 463190] #[464724, 464725, 463189, 463190]
    session_lists = [[1, 2, 3, 4, 5, 6], [1, 2, 6, 7, 8, 9], [1, 3, 4, 9], [2, 3, 5, 6, 10]] # [[1, 2, 3, 4, 5, 6], [1, 2, 6, 7, 8, 9], [1, 3, 4, 9], [2, 3, 5, 6, 10]]
    all_sessions = {}
    for vid_fr in video_frame_rates:
        for i, subject in enumerate(subject_lst):
            session_list = session_lists[i]
            for j, session in enumerate(session_list):
                decoder = RewardSizeDecoder(
                    subject_id=subject,  # subject id
                    session=session,  # session number
                    num_features=200,  # number of predictive features from video
                    frame_rate=vid_fr,  # neural frame rate(Hz)
                    time_bin=(-10, 50),  # trial bin duration(sec)
                    original_video_path='D:/Arseny_behavior_video',  # path to raw original video data - shared video folder located on Z drive
                    model="LR",  # type of classification model to apply on data - supported_models = ['LDA', 'SVM', 'LR']
                    user_model_params=user_model_params,
                    # model hyperparameters, if not specify then the default will be set/ apply parameters search
                    resample_method='simple undersample',
                    # choose resample method to handle unbalanced data
                    dj_info=dj_info,  # data joint user credentials
                    save_folder_name=f"hparams search- fps {vid_fr} Hz",    # choose new folder name for each time you run the model with different parameters
                    save_video_folder = f"processed video - fps {vid_fr} Hz",   # save processed video outputs (downsampled video, svd)
                    handle_omission='convert',
                    # ['keep'(no change), 'clean'(throw omission trials), 'convert'(convert to regular)]
                    clean_ignore=True,  # throw out ignore trials (trials in which the mouse was not responsive)
                    find_parameters=True   # enable hyperparameters search
                )

                decoder.validate_params(supported_models={"LR", "SVM", "LDA"}, supported_resampling=supported_resampling)
                decoder.define_saveroot(reference_path=None,
                                        # data file path/ directory to save results, if None results will be save in the parent folder
                                        log_to_file=False)  # no file logs
                decoder.save_user_parameters(fmt="excel")


                frames_dic = decoder.decoder()






