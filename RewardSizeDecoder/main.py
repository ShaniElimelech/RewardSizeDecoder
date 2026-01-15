from visualization.sessions_average import compute_averages
from extra_analyses.cumulative_pc_auc import cumulative_pc_auc_analysis
from extra_analyses.significance_frames import plot_significant_frames_perm, plot_significant_frames_baseline
import os
import datajoint as dj
import pandas as pd
from RewardSizeDecoder_class import RewardSizeDecoder


#######################################################################

# Run this code lines in your terminal(if there's access to conda path)/ anaconda prompt -
# 1. create conda environment with all dependencies:
# conda env create -f environment.yml
# 2. activate environment:
# conda activate RewardSizeDecoder_pipeline

#######################################################################


if __name__ == '__main__':

    supported_resampling = ['No resample', 'combine undersample(random) and oversample(SMOTE)', 'simple undersample', 'undersample and ensemble']
    supported_models = ['LDA', 'SVM', 'LR']

    #----------------------------------------------------------------------------
    # Define Model Parameters
    #----------------------------------------------------------------------------

    user_model_params = {'LDA': {}, 'SVM': {'probability': True}, 'LR': {'thresh':0.65}}
    host = "arseny-lab.cmte3q4ziyvy.il-central-1.rds.amazonaws.com"
    user = 'ShaniE'
    password = 'opala'
    dj_info = {'host_path': host, 'user_name': user, 'password': password}

    video_frame_rates = [5]        # choose the frame rates that the video will be downsampled to e.g.[2, 5, 10]. you may not be able to run higher fps (depends on compute power)
    pc_analysis = False              # if true the model will run several number of pc's and plot a cumulative auc as a function of number of pc's (default is num_features=200)
    compute_perm = True             # if true compute significant frames with labels permutation
    num_permutations = 2          # if compute_permutation define the number of permutations
    p_value = 0.05                 # define p value for significance

    user_pipeline_params = {
        'num_features' : 200,                                   # number of predictive features from video
        'time_bin' : (-10, 1),                                 # trial bin duration(sec)
        'original_video_path' : 'D:/Arseny_behavior_video',     # path to raw original video data
        'model' : "LR",                                         # type of classification model to apply on data - supported_models = ['LDA', 'SVM', 'LR']
        'user_model_params' : user_model_params,                # model hyperparameters, if not specify then the default will be set/ apply parameters search
        'resample_method' : 'simple undersample',               # choose resample method to handle unbalanced data
        'dj_info' : dj_info,                                    # data joint user credentials
        'save_folder_name' : f"test new version",         # choose new folder name for each time you run the model with different parameters
        'handle_omission' : 'convert',                          # ['keep'(no change), 'clean'(throw omission trials), 'convert'(convert to regular)]
        'clean_ignore' : True,                                  # throw out ignore trials (trials in which the mouse was not responsive)
        'find_parameters' : False,                              # enable hyperparameters search
        'save_results' : True                                   # plot evaluation metrics
    }

    # choose subjects and sessions that you want to test, in case non provided the subjects and sessions will be puled from the database (full list)
    subject_lst = [464724, 464725]              # [464724, 464725, 463189, 463190]
    session_lists = [[1], [13]]               # [[1, 2, 3, 4, 5, 6], [1, 2, 6, 7, 8, 9], [1, 3, 4, 9], [2, 3, 5, 6, 10]]
    if subject_lst is None or session_lists is None:
        # Connect to Datajoint
        try:
            # Try to connect; will raise on failure
            dj.config['database.host'] = dj_info['host_path']
            dj.config['database.user'] = dj_info['user_name']
            dj.config['database.password'] = dj_info['password']
            conn = dj.conn()
        except Exception as e:
            raise ValueError(f"DataJoint connection failed with provided dj_info: {e}, "
                             f"please check if credentials are correct.")

        img = dj.VirtualModule('IMG', 'arseny_learning_imaging')
        exp2 = dj.VirtualModule('EXP2', 'arseny_s1alm_experiment2')
        keySource = exp2.SessionEpoch & exp2.TrialLickPort & 'session_epoch_type="behav_only"' & img.Mesoscope
        df = pd.DataFrame(keySource.fetch('subject_id', 'session')).T.set_axis(['subject_id', 'session'], axis=1)
        df_grouped = df.groupby('subject_id')['session'].apply(list).reset_index()
        subject_lst = df_grouped['subject_id'].tolist()
        session_lists = df_grouped['session'].tolist()


    # ----------------------------------------------------------------------------
    # Run Model
    # ----------------------------------------------------------------------------

    all_sessions = {}

    for vid_fr in video_frame_rates:
        user_pipeline_params['frame_rate'] = vid_fr                                         # neural frame rate(Hz)
        user_pipeline_params['save_video_folder'] = f"processed video test - fps {vid_fr} Hz"   # save processed video outputs (downsampled video, svd)
        if pc_analysis:
            cumulative_pc_auc_analysis(user_pipeline_params.copy(), subject_lst, session_lists, supported_resampling)

        for i, subject in enumerate(subject_lst):
            session_list = session_lists[i]
            for j, session in enumerate(session_list):
                user_pipeline_params['subject_id'] = subject
                user_pipeline_params['session'] = session
                decoder = RewardSizeDecoder(**user_pipeline_params)
                decoder.validate_params(supported_models={"LR", "SVM", "LDA"}, supported_resampling=supported_resampling)
                decoder.define_saveroot(reference_path=None,            # data file path/ directory to save results, if None results will be save in the parent folder
                                        reference_path_video=None,
                                        log_to_file=False)              # dont save logs to file
                decoder.save_user_parameters(fmt="excel")

                all_frames_scores = decoder.decoder()

        # plot average results across subject sessions and all subjects
        compute_averages(subject_lst, session_lists, decoder.model, vid_fr, decoder.saveroot)

        # compute significant frames - compare results to baseline and to shuffle data if compute_perm = True
        sig_save_path = os.path.join(decoder.saveroot,'significant analysis')
        os.makedirs(sig_save_path, exist_ok=True)
        if compute_perm:
            # the function first checks if the permutation was already computed - if so it will not compute again and will load existing outputs
            # ot will look for the output in the significant analysis folder in your save folder - make sure its really there
            plot_significant_frames_perm(
                num_permutations,
                p_value,
                user_pipeline_params.copy(),
                subject_lst,
                session_lists,
                supported_resampling,
                decoder.saveroot,
                sig_save_path
            )

        plot_significant_frames_baseline(subject_lst,session_lists, p_value, user_pipeline_params.copy(), decoder.saveroot, sig_save_path)






