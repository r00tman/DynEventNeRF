import logging
import configargparse


logger = logging.getLogger(__package__)

def setup_logger():
    global logger
    # create logger
    logger = logging.getLogger(__package__)
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # grey = "\x1b[38;20m"
    # yellow = "\x1b[33;20m"
    # bold_yellow = "\x1b[33;1m"
    # red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # create formatter
    formatter = logging.Formatter(f'%(asctime)s [{bold_red}%(levelname)s{reset}] %(name)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


def config_parser(config_path=None):
    # note that we pass config_path as the default value
    def open_config_file(path):
        # override the supplied path
        if config_path:
            path = config_path
        return open(path)

    parser = configargparse.ArgumentParser(config_file_open_func=open_config_file)

    def str2bool(v):
        if isinstance(v, bool):
            return v
        elif v == 'True':
            return True
        elif v == 'False':
            return False
        else:
            raise configargparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument('--config', is_config_file=True, help='config file path', default=config_path)
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--slurmjob", type=str, default='', help='slurm job id')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')

    # ablation options
    parser.add_argument("--use_lr_scheduler", type=str2bool, default=True)
    parser.add_argument("--use_annealing", type=str2bool, default=True)
    parser.add_argument("--use_ray_jitter", type=str2bool, default=True)
    parser.add_argument("--use_pe", type=str2bool, default=True)

    parser.add_argument("--use_event_loss", type=str2bool, default=True)
    parser.add_argument("--use_rgb_loss", type=str2bool, default=True)
    parser.add_argument("--use_accumulation_loss", type=str2bool, default=True)

    parser.add_argument("--is_rgb_only", type=str2bool, default=False)

    parser.add_argument("--seed_offset", type=int, default=0, help='random seed offset')

    # dataset options
    parser.add_argument("--datadir", type=str, default=None, help='input data directory')
    parser.add_argument("--scene", type=str, default=None, help='scene name')
    parser.add_argument("--train_split", type=str, default='train', help='training split')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    parser.add_argument("--trainskip", type=int, default=1,
                        help='will load 1/N images from train sets, useful for large datasets like deepvoxels')

    parser.add_argument("--event_threshold", type=float, default=0.5, help='event threshold')
    parser.add_argument("--polarity_offset", type=float, default=0.0, help='polarity offset')
    parser.add_argument("--damping_strength", type=float, default=0.93, help='event damping strength')

    parser.add_argument("--tonemap_eps", type=float, default=1e-5, help='tonemapping eps')

    parser.add_argument("--bg_color", type=float, default=159., help='background color in srgb')

    parser.add_argument("--tstart", type=float, default=0., help='sequence start time')
    parser.add_argument("--tend", type=float, default=1000., help='sequence end time')

    # model size
    parser.add_argument("--backend", type=str, default='mlp', help='nerf backend (mlp, tcnn, tensorfvm, tensorfcp)')

    parser.add_argument("--crop_y_min", type=float, default=-1, help='zero density of everything below')
    parser.add_argument("--crop_y_max", type=float, default=1, help='zero density of everything above')
    parser.add_argument("--crop_r", type=float, default=1, help='zero density of everything outside of x2+z2<=r')

    # mlp backend parameters
    parser.add_argument("--netdepth", type=int, default=8, help='layers in coarse network')
    parser.add_argument("--netwidth", type=int, default=256, help='channels per layer in coarse network')
    parser.add_argument("--use_viewdirs", type=str2bool, default=False, help='use full 5D input instead of 3D')

    parser.add_argument("--activation", type=str, default=None, help='activation function (relu, elu, sine, garf, tanh)')
    parser.add_argument("--garf_sigma", type=float, default=1.0, help='garf activation function sigma')

    # tensorf backend parameters
    parser.add_argument("--tensorf_grid_dim", type=int, default=500, help='tensorf grid dimension')
    parser.add_argument("--tensorf_grid_dim_time", type=int, default=24, help='tensorf grid dimension across time')
    parser.add_argument("--tensorf_rank", type=int, default=1, help='tensorf rank')
    parser.add_argument("--tensorf_Hsteps", type=int, default=10, help='tensorf progressive growing steps')
    parser.add_argument("--tensorf_Hmin", type=int, default=16, help='tensorf progressive growing starting grid dimension')
    parser.add_argument("--tensorf_Hmin_time", type=int, default=16, help='tensorf progressive growing starting grid dimension')
    parser.add_argument("--tensorf_Hiters", type=int, default=2000, help='tensorf progressive growing total iteration count')

    # checkpoints
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--init_ckpt_path", type=str, default=None,
                        help='specific weights npy file to use as initialization (no camara mgr, no optim are going to be loaded). if there are other checkpoints they will be used instead')
    parser.add_argument("--force_ckpt_path", type=str, default=None,
                        help='force to use these specific weights npy file as initialization. all other ckpt options are ignored')

    # batch size
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 2,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk_size", type=int, default=1024 * 8,
                        help='number of rays processed in parallel, decrease if running out of memory')

    # iterations
    parser.add_argument("--N_iters", type=int, default=250001,
                        help='number of iterations')

    parser.add_argument("--N_iters_transform", type=int, default=2000,
                        help='for how many iterations we only optimize the transform of the initial checkpoint')

    # render only
    parser.add_argument("--render_splits", type=str, default='test',
                        help='splits to render')

    parser.add_argument("--render_view", type=str, default='',
                        help='which view to render (leave empty to render all views in the splits)')

    parser.add_argument("--render_out_dir", type=str, default='',
                        help='override render directory')

    parser.add_argument("--render_bullet_time_path", type=str, default='',
                        help='bullet time render config path')

    parser.add_argument("--write_video", type=str2bool, default=False, help='convert rendered images to videos')

    parser.add_argument("--render_bullet_time", type=str2bool, default=False, help='each view is a single frame')

    parser.add_argument("--render_timestamp_periods", type=float, default=1,
                        help='how many time periods to render (1=1 period)')

    parser.add_argument("--render_timestamp_frames", type=int, default=100,
                        help='how many frames to render per view in total')

    parser.add_argument("--render_tstart", type=float, default=-1, help='render sequence start time (-1 for tstart)')
    parser.add_argument("--render_tend", type=float, default=-1, help='render sequence end time (-1 for tend)')

    # cascade training
    parser.add_argument("--cascade_level", type=int, default=2,
                        help='number of cascade levels')
    parser.add_argument("--cascade_samples", type=str, default='64,64',
                        help='samples at each level')

    # multiprocess learning
    parser.add_argument("--world_size", type=int, default='-1',
                        help='number of processes')

    # regularizers
    parser.add_argument("--use_ldist_reg", type=str2bool, default=True)
    parser.add_argument("--ldist_reg", type=float, default=0.001, help='distortion regularizer (mipnerf 360)')

    parser.add_argument("--use_tv_reg", type=str2bool, default=False)
    parser.add_argument("--tv_reg", type=float, default=0.000, help='total variation  pixel-wise sparsity')

    parser.add_argument("--use_lambda_reg", type=str2bool, default=True)
    parser.add_argument("--lambda_reg", type=float, default=0.000, help='L1 pixel-wise sparsity')

    parser.add_argument("--use_tensorf_sparsity", type=str2bool, default=False)
    parser.add_argument("--tensorf_sparsity", type=float, default=0.000, help='tensorf term sparsity regularizer weight')
    parser.add_argument("--tensorf_sparsity_startit", type=int, default=2000, help='iteration from which tensorf term sparsity kicks in')

    parser.add_argument("--use_tensorf_smoothness", type=str2bool, default=False)
    parser.add_argument("--tensorf_smoothness", type=float, default=0.000, help='tensorf term smoothness reglarizer weight')

    parser.add_argument("--use_tensorf_tv", type=str2bool, default=False)
    parser.add_argument("--tensorf_tv", type=float, default=0.000, help='tensorf term total variation reglarizer weight')

    parser.add_argument("--optimize_transform", type=str2bool, default=True, help='do optimize transform along the model?')

    parser.add_argument("--neg_ratio", type=float, default=0,
                        help='ratio of samples at pixels without events')

    parser.add_argument("--neg_ratio_anneal", type=int, default=0,
                        help='number of negative ratio anneal iterations')

    parser.add_argument("--init_gain", type=float, default=5,
                        help='initialisation gain')


    # learning rate options
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay_factor", type=float, default=0.1,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--lrate_decay_steps", type=int, default=5000,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--weight_decay", type=float, default=0, help='weight decay')

    # rendering options
    parser.add_argument("--det", action='store_true', help='deterministic sampling for coarse and fine samples')
    parser.add_argument("--max_freq_log2_pos", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--max_freq_log2_time", type=int, default=10,
                        help='log2 of max freq for positional encoding (1D time)')
    parser.add_argument("--max_freq_log2_viewdirs", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--N_anneal", type=int, default=100000,
                        help='number of embedder anneal iterations')
    parser.add_argument("--N_anneal_min_freq", type=int, default=0,
                        help='number of embedder frequencies to start annealing from')
    parser.add_argument("--N_anneal_min_freq_viewdirs", type=int, default=0,
                        help='number of viewdir embedder frequencies to start annealing from')

    parser.add_argument("--N_anneal_lambda", type=float, default=40000,
                        help='lambda reg annealing factor (lower -- earlier onset, higher -- later onset)')

    parser.add_argument("--load_min_depth", action='store_true', help='whether to load min depth')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100, help='frequency of console printout and metric logging')
    parser.add_argument("--i_img", type=int, default=500, help='frequency of tensorboard image logging (if <0, then no images are logged)')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')

    return parser
