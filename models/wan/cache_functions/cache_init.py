import numpy as np


def cache_init(method, model_kwargs=None, cfgcache_runtime=None):

    model_kwargs = model_kwargs or {}

    num_layers = int(model_kwargs.get("num_layers", 30))
    use_true_cfg = bool(model_kwargs.get("use_true_cfg", True))
    branch_names = ["cond", "uncond"] if use_true_cfg else ["main"]
    block_names = ["img_attn", "img_cross_attn", "img_mlp"]

    cache_dic = {}
    cache = {-1: {}}
    cache_index = {-1: {}, "layer_index": {}}

    cache_dic["cache_counter"] = 0
    cache_dic["mode"] = method
    cache_dic["model_type"] = "wan"
    cache_dic["taylor_cache"] = False

    for model in branch_names:
        cache[-1][model] = {}
        for j in range(num_layers):
            cache[-1][model][j] = {name: {} for name in block_names}
            if j not in cache_index[-1]:
                cache_index[-1][j] = {}

    if method == "original":
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_threshold"] = 1
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 3
        cache_dic["use_grouped_taylor"] = False

    elif method == "Taylor":
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

    elif method == "HiCache":
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

    elif method == "GroupedTaylor":
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_threshold"] = model_kwargs.get("interval", 6)
        cache_dic["taylor_cache"] = True
        cache_dic["use_grouped_taylor"] = True
        cache_dic["max_order"] = model_kwargs.get("max_order", 2)
        cache_dic["first_enhance"] = model_kwargs.get("first_enhance", 3)
        cache_dic["n_clusters"] = model_kwargs.get("n_clusters", 2)
        cache_dic["group_orders"] = model_kwargs.get("group_orders", [1, 2])
        cache_dic["history_window"] = model_kwargs.get("history_window", 4)
        cache_dic["save_clustering_visualization"] = False
        cache_dic["current_image_idx"] = 0
        cache_dic["one_time_clustering"] = False
        cache_dic["clustering_count"] = 0

        def _make_taylor_branch_state():
            return {
                "cache_counter": 0,
                "activated_steps": [0],
            }

        cache_dic["taylor_branch_state"] = {
            branch: _make_taylor_branch_state() for branch in branch_names
        }

        cache_dic["grouped_cache"] = {}
        for model in branch_names:
            for layer in range(num_layers):
                for block in block_names:
                    layer_key = (model, layer, block)
                    cache_dic["grouped_cache"][layer_key] = {}

    elif method == "TeaCache":
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_threshold"] = 1
        cache_dic["cal_threshold"] = 1
        cache_dic["taylor_cache"] = False
        cache_dic["use_grouped_taylor"] = False
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = model_kwargs.get("first_enhance", 3)

        cache_dic["teacache_enable"] = True

        teacache_coeffs = model_kwargs.get(
            "teacache_coefficients",
            [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01],
        )

        def _make_teacache_branch_state():
            return {
                "cnt": 0,
                "num_steps": model_kwargs.get("num_steps", 50),
                "rel_l1_thresh": model_kwargs.get("rel_l1_thresh", 0.2),
                "accumulated_rel_l1_distance": 0.0,
                "previous_modulated_input": None,
                "previous_residual": None,
                "coefficients": teacache_coeffs,
            }

        cache_dic["teacache"] = {
            branch: _make_teacache_branch_state() for branch in branch_names
        }

    elif method == "DiCache":
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_threshold"] = 1
        cache_dic["cal_threshold"] = 1
        cache_dic["taylor_cache"] = False
        cache_dic["use_grouped_taylor"] = False
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 0

        cache_dic["dicache_enable"] = True

        def _make_dicache_branch_state():
            return {
                "cnt": 0,
                "num_steps": model_kwargs.get("num_steps", 50),
                "ret_ratio": model_kwargs.get("ret_ratio", 0.2),
                "probe_depth": model_kwargs.get("probe_depth", 2),
                "rel_l1_thresh": model_kwargs.get("rel_l1_thresh", 0.4),
                "error_choice": model_kwargs.get("error_choice", "delta_y"),
                "accumulated_rel_l1_distance": 0.0,
                "resume_flag": False,

                # current base hidden
                "base_img": None,
                "base_hidden_states": None,

                # previous references
                "previous_input": None,
                "previous_probe_states": None,

                # reusable residuals
                "previous_residual": None,
                "previous_probe_residual": None,
                "residual_window": [],
                "probe_residual_window": [],

                # temporary probe states
                "_probe_img": None,
                "_probe_hidden_states": None,
            }

        cache_dic["dicache"] = {
            branch: _make_dicache_branch_state() for branch in branch_names
        }

    elif method == "CFGCache":
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_threshold"] = 1
        cache_dic["cal_threshold"] = 1
        cache_dic["taylor_cache"] = False
        cache_dic["use_grouped_taylor"] = False
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = model_kwargs.get("first_enhance", 1)

        cache_dic["cfgcache_enable"] = True

        def _make_cfgcache_branch_state():
            return {
                # final reused hidden / residual
                "previous_postnorm_hidden": None,
                "previous_residual": None,

                # current base hidden
                "base_hidden_states": None,
                "base_img": None,

                # proxy runtime
                "proxy_ref_mode": model_kwargs.get("proxy_ref_mode", "anchor"),
                "proxy_prev_input": None,
                "proxy_prev_probe_states": None,
                "proxy_anchor_input": None,
                "proxy_anchor_probe_states": None,
                "proxy_anchor_modulated_inp": None,

                # current-step probe cache
                "_probe_hidden_states": None,
                "_probe_img": None,
                "_probe_payload": None,
                "_probe_depth": 0,
                "_probe_summary": None,
                "_probe_emb": None,
                "_probe_modulated_inp": None,

                # bookkeeping
                "last_action": "full_both",
                "last_dhat_branch": None,
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

        num_steps = int(model_kwargs.get("num_steps", 50))

        tau = float(model_kwargs.get("cfgcache_thresh", 0.4))
        use_prop_weight = bool(model_kwargs.get("use_prop_weight", True))
        prop_a = float(model_kwargs.get("prop_a", 1.214))
        prop_alpha = float(model_kwargs.get("prop_alpha", 3.832))
        prop_b = float(model_kwargs.get("prop_b", 0.2463))

        joint_state = {
            "anchor_step": 0,
            "accumulated_risk": 0.0,
            "consecutive_reuse": 0,
            "last_action": "full_both",
            "true_cfg_scale": float(model_kwargs.get("true_cfg_scale", 1.0)),
            "tau": tau,
            "log": [],
            "last_metrics": None,
            "prop_weight_schedule": _build_prop_weight_schedule(
                T=num_steps,
                a=prop_a,
                alpha=prop_alpha,
                b=prop_b,
            ),
        }

        cache_dic["cfgcache"] = {
            # branch-local runtime
            "cond": _make_cfgcache_branch_state(),
            "uncond": _make_cfgcache_branch_state(),

            # shared joint controller state
            "joint_state": joint_state,

            # shared scheduler / proxy config
            "cfgcache_thresh": tau,
            "use_prop_weight": use_prop_weight,
            "prop_a": prop_a,
            "prop_alpha": prop_alpha,
            "prop_b": prop_b,
            "num_steps": num_steps,
            "warmup_steps": int(model_kwargs.get("warmup_steps", 10)),

            # proxy config
            "proxy_name": model_kwargs.get("proxy_name", "teacache_proxy"),
            "probe_depth": int(model_kwargs.get("probe_depth", 2)),
            "proxy_error_choice": model_kwargs.get("proxy_error_choice", "delta_y"),
            "proxy_eps": float(model_kwargs.get("proxy_eps", 1e-6)),
            "proxy_stream": model_kwargs.get("proxy_stream", "hidden"),

            # Wan is single-stream; no encoder stream reuse.
            "reuse_encoder": False,

            # optional Tea-like polynomial proxy
            "teacache_poly_coeffs": model_kwargs.get(
                "teacache_poly_coeffs",
                [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01],
            ),

            # offline runtime payloads
            "rho_table_at": None if cfgcache_runtime is None else cfgcache_runtime.get("rho_table_at", None),
            "runtime": {} if cfgcache_runtime is None else cfgcache_runtime,

            # bookkeeping
            "last_prop_weight": 1.0,
            "last_dhat": None,
            "last_rhat": None,
            "last_action": "full_both",
        }

        from ..racfgcache_utils.proxies import get_proxy

        proxy = get_proxy(cache_dic["cfgcache"]["proxy_name"])
        cache_dic["cfgcache"]["_proxy_obj"] = proxy
        proxy.init_state(cfgcache_runtime, cache_dic["cfgcache"])

    elif method == "MagCache":
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_threshold"] = 1
        cache_dic["cal_threshold"] = 1
        cache_dic["taylor_cache"] = False
        cache_dic["use_grouped_taylor"] = False
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = model_kwargs.get("first_enhance", 3)
        cache_dic["magcache_enable"] = True

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

        T = int(model_kwargs.get("num_steps", 50))

        official_13b_mag_ratios = np.array(
            [1.0] * 2 + [
                1.0124, 1.02213,
                1.00166, 1.0041,
                0.99791, 1.00061,
                0.99682, 0.99762,
                0.99634, 0.99685,
                0.99567, 0.99586,
                0.99416, 0.99422,
                0.99578, 0.99575,
                0.9957, 0.99563,
                0.99511, 0.99506,
                0.99535, 0.99531,
                0.99552, 0.99549,
                0.99541, 0.99539,
                0.9954, 0.99536,
                0.99489, 0.99485,
                0.99518, 0.99514,
                0.99484, 0.99478,
                0.99481, 0.99479,
                0.99415, 0.99413,
                0.99419, 0.99416,
                0.99396, 0.99393,
                0.99388, 0.99386,
                0.99349, 0.99349,
                0.99309, 0.99304,
                0.9927, 0.9927,
                0.99228, 0.99226,
                0.99171, 0.9917,
                0.99137, 0.99135,
                0.99068, 0.99063,
                0.99005, 0.99003,
                0.98944, 0.98942,
                0.98849, 0.98849,
                0.98758, 0.98757,
                0.98644, 0.98643,
                0.98504, 0.98503,
                0.9836, 0.98359,
                0.98202, 0.98201,
                0.97977, 0.97978,
                0.97717, 0.97718,
                0.9741, 0.97411,
                0.97003, 0.97002,
                0.96538, 0.96541,
                0.9593, 0.95933,
                0.95086, 0.95089,
                0.94013, 0.94019,
                0.92402, 0.92414,
                0.90241, 0.9026,
                0.86821, 0.86868,
                0.81838, 0.81939,
            ],
            dtype=np.float64,
        )

        ratios_in = np.asarray(
            model_kwargs.get("mag_ratios", official_13b_mag_ratios),
            dtype=np.float64,
        )

        if ratios_in.ndim == 1 and len(ratios_in) == 2 * T:
            ratios_cond = ratios_in[0::2]
            ratios_uncond = ratios_in[1::2]
        elif ratios_in.ndim == 1 and len(ratios_in) == T:
            ratios_cond = ratios_in
            ratios_uncond = ratios_in
        else:
            ratios_cond = _nearest_interp(ratios_in[0::2], T)
            ratios_uncond = _nearest_interp(ratios_in[1::2], T)

        ratios_cond = _nearest_interp(ratios_cond, T)
        ratios_uncond = _nearest_interp(ratios_uncond, T)

        def _make_branch_state(branch):
            if branch == "uncond":
                branch_ratios = ratios_uncond
            else:
                branch_ratios = ratios_cond

            return {
                "cnt": 0,
                "num_steps": T,
                "K": int(model_kwargs.get("K", 4)),
                "magcache_thresh": float(model_kwargs.get("magcache_thresh", 0.1)),
                "retention_ratio": float(model_kwargs.get("retention_ratio", 0.1)),
                "accumulated_ratio": 1.0,
                "accumulated_err": 0.0,
                "accumulated_steps": 0,
                "mag_ratios": branch_ratios,
                "previous_residual": None,
                "base_img": None,
                "base_hidden_states": None,
            }

        cache_dic["magcache"] = {
            branch: _make_branch_state(branch) for branch in branch_names
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

    # CFGCache runtime fields
    current["cfgcache_joint_action"] = "full_both"
    current["cfgcache_probe_prepared"] = False
    current["cfgcache_dhat"] = None
    current["cfgcache_ghat"] = 1.0
    current["cfgcache_rhat"] = None

    return cache_dic, current