import numpy as np
def cache_init(method, model_kwargs=None, cfgcache_runtime=None):   
    '''
    Initialization for cache.
    '''
    model_kwargs = model_kwargs or {}

    num_layers = int(model_kwargs.get("num_layers", 30))   # CogVideoX 默认 30
    use_true_cfg = bool(model_kwargs.get("use_true_cfg", True))
    branch_names = ["cond", "uncond"] if use_true_cfg else ["main"]
    block_names = ["img_attn", "txt_attn", "img_mlp", "txt_mlp"]

    cache_dic = {}
    cache = {-1: {}}
    cache_index = {-1: {}, "layer_index": {}}

    cache_dic["cache_counter"] = 0
    cache_dic["mode"] = method
    cache_dic["taylor_cache"] = False
    for model in branch_names:
        cache[-1][model] = {}
        for j in range(num_layers):
            cache[-1][model][j] = {name: {} for name in block_names}
            if j not in cache_index[-1]:
                cache_index[-1][j] = {}

    if method == 'original':
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_threshold'] = 1
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 3
        cache_dic['use_grouped_taylor'] = False

    elif method == 'Taylor':
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_threshold"] = model_kwargs.get("interval", 6)
        cache_dic["taylor_cache"] = True
        cache_dic["max_order"] = model_kwargs.get("max_order", 2)
        cache_dic["first_enhance"] = model_kwargs.get("first_enhance", 3)
        cache_dic["use_grouped_taylor"] = False
        def _make_taylor_branch_state():
            return {
                "cache_counter": 0,
                "activated_steps": [0],
            }
        cache_dic["taylor_branch_state"] = {
            branch: _make_taylor_branch_state() for branch in branch_names
        }
    elif method == 'HiCache':
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_threshold"] = model_kwargs.get("interval", 6)
        cache_dic["taylor_cache"] = True
        cache_dic["max_order"] = model_kwargs.get("max_order", 1)
        cache_dic["first_enhance"] = model_kwargs.get("first_enhance", 3)
        cache_dic["use_grouped_taylor"] = False
        cache_dic["hicache_scale_factor"] = model_kwargs.get("hicache_scale", 0.5)
        cache_dic["prediction_mode"] = "hicache"
        def _make_taylor_branch_state():
            return {
                "cache_counter": 0,
                "activated_steps": [0],
            }
        cache_dic["taylor_branch_state"] = {
            branch: _make_taylor_branch_state() for branch in branch_names
        }
    elif method == 'GroupedTaylor':
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_threshold'] = model_kwargs['interval']
        cache_dic['taylor_cache'] = True
        cache_dic['use_grouped_taylor'] = True
        cache_dic['max_order'] = model_kwargs['max_order']
        cache_dic['first_enhance'] = model_kwargs['first_enhance']
        cache_dic['n_clusters'] = 2
        cache_dic['group_orders'] = [1, 2]
        cache_dic['history_window'] = 4
        cache_dic['save_clustering_visualization'] = False
        cache_dic['current_image_idx'] = 0
        cache_dic['one_time_clustering'] = False
        cache_dic['clustering_count'] = 0
        def _make_taylor_branch_state():
            return {
                "cache_counter": 0,
                "activated_steps": [0],
            }
        cache_dic["taylor_branch_state"] = {
            branch: _make_taylor_branch_state() for branch in branch_names
        }
        cache_dic['grouped_cache'] = {}
        for model in ['cond', 'uncond']:
            for layer in range(60):
                for block in ['img_attn', 'txt_attn', 'img_mlp', 'txt_mlp']:
                    layer_key = (model, layer, block)
                    cache_dic['grouped_cache'][layer_key] = {}

    elif method == 'TeaCache':
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_threshold'] = 1
        cache_dic['cal_threshold'] = 1
        cache_dic['taylor_cache'] = False
        cache_dic['use_grouped_taylor'] = False
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = model_kwargs.get('first_enhance', 3)

        cache_dic['teacache_enable'] = True

        teacache_coeffs = model_kwargs.get(
            'teacache_coefficients',
            [-3.10658903e+01, 2.54732368e+01, -5.92380459e+00, 1.75769064e+00, -3.61568434e-03]
        )

        def _make_teacache_branch_state():
            return {
                'cnt': 0,
                'num_steps': model_kwargs.get('num_steps', 50),
                'rel_l1_thresh': model_kwargs.get('rel_l1_thresh', 0.4),
                'accumulated_rel_l1_distance': 0.0,
                'previous_modulated_input': None,
                'previous_residual': None,
                'previous_residual_encoder': None,
                'coefficients': teacache_coeffs,
            }

        cache_dic['teacache'] = {
            'cond': _make_teacache_branch_state(),
            'uncond': _make_teacache_branch_state(),
        }

    elif method == 'DiCache':
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_threshold'] = 1
        cache_dic['cal_threshold'] = 1
        cache_dic['taylor_cache'] = False
        cache_dic['use_grouped_taylor'] = False
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 0

        cache_dic['dicache_enable'] = True

        def _make_dicache_branch_state():
            return {
                'cnt': 0,
                'num_steps': model_kwargs.get('num_steps', 50),
                'ret_ratio': model_kwargs.get('ret_ratio', 0.1),
                'probe_depth': model_kwargs.get('probe_depth', 2),
                'rel_l1_thresh': model_kwargs.get('rel_l1_thresh', 0.4),
                'error_choice': model_kwargs.get('error_choice', 'delta_y'),
                'accumulated_rel_l1_distance': 0.0,
                'resume_flag': False,
                'base_img': None,
                'base_hidden_states': None,
                'base_encoder_hidden_states': None,
                'previous_input': None,
                'previous_input_encoder': None,

                'previous_probe_states': None,
                'previous_probe_states_encoder': None,

                'previous_residual': None,
                'previous_residual_encoder': None,
                'previous_probe_residual': None,
                'previous_probe_residual_encoder': None,
                'residual_window': [],
                'residual_window_encoder': [],

                'probe_residual_window': [],
                'probe_residual_window_encoder': [],
                '_probe_img': None,
                '_probe_txt': None,

                '_probe_hidden_states': None,
                '_probe_encoder_hidden_states': None,
            }

        cache_dic['dicache'] = {
            'cond': _make_dicache_branch_state(),
            'uncond': _make_dicache_branch_state(),
        }
    elif method == 'CFGCache':
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_threshold'] = 1
        cache_dic['cal_threshold'] = 1
        cache_dic['taylor_cache'] = False
        cache_dic['use_grouped_taylor'] = False
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = model_kwargs.get('first_enhance', 1)

        cache_dic['cfgcache_enable'] = True

        # --------------------------------------------------
        # single-stream CFGCache branch-local state
        # cache object = post-norm hidden (after norm_final, before norm_out)
        # --------------------------------------------------
        def _make_cfgcache_branch_state():
            return {
                # -------- final reused feature --------
                'previous_postnorm_hidden': None,

                # -------- proxy runtime (single-stream only) --------
                'proxy_ref_mode': model_kwargs.get('proxy_ref_mode', 'anchor'),
                'proxy_prev_input': None,               # previous base hidden
                'proxy_prev_probe_states': None,        # previous shallow-probe hidden
                'proxy_anchor_input': None,
                'proxy_anchor_probe_states': None,
                'proxy_anchor_modulated_inp': None,

                # -------- current-step probe cache --------
                '_probe_hidden_states': None,
                '_probe_payload': None,
                '_probe_depth': 0,
                '_probe_summary': None,
                '_probe_emb': None,
                '_probe_modulated_inp': None,

                # -------- bookkeeping --------
                'last_action': 'full_both',
                'last_dhat_branch': None,
            }

        def _build_prop_weight_schedule(T: int, a: float, alpha: float, b: float):
            T = max(int(T), 1)
            if T == 1:
                return [1.0]

            raws = []
            for k in range(T):
                xk = (T - 1 - k) / float(T - 1)
                raws.append(a * (xk ** alpha) + b)

            raw_mean = max(sum(raws) / len(raws), 1e-12)
            return [float(r / raw_mean) for r in raws]

        num_steps = int(model_kwargs.get('num_steps', 50))

        tau = float(model_kwargs.get('cfgcache_thresh', 0.4))
        use_prop_weight = bool(model_kwargs.get('use_prop_weight', True))
        prop_a = float(model_kwargs.get('prop_a', 2.185))
        prop_alpha = float(model_kwargs.get('prop_alpha', 3.449))
        prop_b = float(model_kwargs.get('prop_b', 0.213))

        joint_state = {
            'anchor_step': 0,
            'accumulated_risk': 0.0,
            'consecutive_reuse': 0,
            'last_action': 'full_both',
            'true_cfg_scale': float(model_kwargs.get('true_cfg_scale', 1.0)),
            'tau': tau,
            'log': [],
            'last_metrics': None,
            'prop_weight_schedule': _build_prop_weight_schedule(
                T=num_steps,
                a=prop_a,
                alpha=prop_alpha,
                b=prop_b,
            ),
        }

        cache_dic['cfgcache'] = {
            # -------------------------
            # branch-local runtime
            # -------------------------
            'cond': _make_cfgcache_branch_state(),
            'uncond': _make_cfgcache_branch_state(),

            # -------------------------
            # shared joint controller state
            # -------------------------
            'joint_state': joint_state,

            # -------------------------
            # shared scheduler / proxy config
            # -------------------------
            'cfgcache_thresh': tau,
            'use_prop_weight': use_prop_weight,
            'prop_a': prop_a,
            'prop_alpha': prop_alpha,
            'prop_b': prop_b,
            'num_steps': num_steps,
            'warmup_steps': int(model_kwargs.get('warmup_steps', 5)),

            # proxy config
            'proxy_name': model_kwargs.get('proxy_name', 'teacache_proxy'),
            'probe_depth': int(model_kwargs.get('probe_depth', 2)),
            'proxy_error_choice': model_kwargs.get('proxy_error_choice', 'delta_y'),
            'proxy_eps': float(model_kwargs.get('proxy_eps', 1e-6)),
            'proxy_stream': model_kwargs.get('proxy_stream', 'hidden'),

            # compatibility flag: single-stream CFGCache no longer reuses encoder branch
            'reuse_encoder': False,

            # optional Tea-like polynomial probe support
            'teacache_poly_coeffs': model_kwargs.get(
                'teacache_poly_coeffs',
                [-3.10658903e+01, 2.54732368e+01, -5.92380459e+00, 1.75769064e+00, -3.61568434e-03]
            ),

            # offline runtime payloads
            'rho_table_at': None if cfgcache_runtime is None else cfgcache_runtime.get('rho_table_at', None),
            'runtime': {} if cfgcache_runtime is None else cfgcache_runtime,

            # bookkeeping
            'last_prop_weight': 1.0,
            'last_dhat': None,
            'last_rhat': None,
            'last_action': 'full_both',
        }

        from ..racfgcache_utils.proxies import get_proxy
        proxy = get_proxy(cache_dic['cfgcache']['proxy_name'])
        cache_dic['cfgcache']['_proxy_obj'] = proxy
        proxy.init_state(cfgcache_runtime, cache_dic['cfgcache'])

    elif method == 'MagCache':
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_threshold'] = 1
        cache_dic['cal_threshold'] = 1
        cache_dic['taylor_cache'] = False
        cache_dic['use_grouped_taylor'] = False
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = model_kwargs.get('first_enhance', 3)
        cache_dic['magcache_enable'] = True

        def _nearest_interp(src_array, target_length):
            src_array = np.asarray(src_array, dtype=np.float64)
            src_length = len(src_array)
            if src_length == target_length:
                return src_array
            if target_length == 1:
                return np.array([src_array[-1]], dtype=np.float64)
            scale = (src_length - 1) / (target_length - 1)
            mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
            return src_array[mapped_indices]

        base_mag_ratios = np.array([1.0, 0.8945779800415039, 1.0000561475753784, 1.0088841915130615, 1.0048277378082275, 0.9904030561447144, 0.9913018345832825, 0.9970088005065918, 0.9956251978874207, 0.9893767833709717, 0.9884808659553528, 0.9870059490203857, 0.9897987842559814, 0.9861218333244324, 0.9903091192245483, 0.9824209213256836, 0.9890511631965637, 0.9905682802200317, 0.986688494682312, 0.9865743517875671, 0.9890210032463074, 0.9870433211326599, 0.9867755174636841, 0.9882943630218506, 0.9810936450958252, 0.9895802736282349, 0.9884207248687744, 0.985073983669281, 0.9882029891014099, 0.98781818151474, 0.9865850210189819, 0.9862693548202515, 0.990170419216156, 0.9860052466392517, 0.985201358795166, 0.9882901906967163, 0.987604022026062, 0.984825074672699, 0.9849751591682434, 0.9847162961959839, 0.985359251499176, 0.9830584526062012, 0.9847055077552795, 0.9812142848968506, 0.9805524945259094, 0.9764180779457092, 0.9747219681739807, 0.9719496369361877, 0.970180332660675, 0.9615857601165771], dtype=np.float64)

        ratios_in = model_kwargs.get('mag_ratios', base_mag_ratios)
        ratios_aligned = _nearest_interp(ratios_in, model_kwargs.get('num_steps', 50))

        def _make_branch_state():
            return {
                'cnt': 0,
                'num_steps': model_kwargs.get('num_steps', 50),
                'K': int(model_kwargs.get('K', 5)),
                'magcache_thresh': float(model_kwargs.get('magcache_thresh', 0.08)),
                'retention_ratio': float(model_kwargs.get('retention_ratio', 0.1)),
                'accumulated_ratio': 1.0,
                'accumulated_err': 0.0,
                'accumulated_steps': 0,
                'mag_ratios': ratios_aligned,
                'previous_residual': None,
                'previous_residual_encoder': None,
                'base_img': None,
                'base_encoder_hidden_states': None,
            }

        cache_dic['magcache'] = {
            'cond': _make_branch_state(),
            'uncond': _make_branch_state(),
        }
    else:
        raise ValueError(f"Unsupported cache method: {method}")

    default_branch = branch_names[0]

    current = {}
    current["step"] = 0
    if method in ["Taylor", "HiCache", "GroupedTaylor"]:
        current["activated_steps"] = cache_dic["taylor_branch_state"][default_branch]["activated_steps"]
    else:
        current["activated_steps"] = [0]
    current["num_steps"] = model_kwargs.get("num_steps", 50)
    current["model"] = default_branch
    current["layer"] = 0
    current["block"] = "img_attn"
    current["cfgcache_joint_action"] = "full_both"
    current["cfgcache_probe_prepared"] = False
    current["cfgcache_dhat"] = None
    current["cfgcache_ghat"] = 1.0
    current["cfgcache_rhat"] = None

    
    return cache_dic, current