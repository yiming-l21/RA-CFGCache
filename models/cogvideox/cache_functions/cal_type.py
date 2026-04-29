import numpy as np
def cal_type(cache_dic, current):
    '''
    Determine calculation type for this step
    '''
    if 'full_counter' not in cache_dic:
        cache_dic['full_counter'] = 0

    # -------------------------
    # TeaCache branch
    # -------------------------
    if cache_dic.get('mode') == 'TeaCache':
        if not cache_dic.get('teacache_enable', True):
            current['type'] = 'full'
            return

        branch = current.get('model', 'cond')
        tc = cache_dic['teacache'][branch]

        step = int(current.get('step', 0))
        T = int(tc.get('num_steps', current.get('num_steps', 0) or 0))
        cnt = int(tc.get('cnt', 0))
        mod_inp = current.get('teacache_modulated_inp', None)   # 这里应该是 emb.detach()

        current['type'] = 'full'

        # 官方：首步和末步强制 full
        if cnt == 0 or (T > 0 and cnt == T - 1):
            print(f"TeaCache step {step}: warmup/last step full (cnt={cnt}, T={T})")
            tc['accumulated_rel_l1_distance'] = 0.0
            tc['previous_modulated_input'] = mod_inp
            current['type'] = 'full'
            current.setdefault('activated_steps', []).append(step)
            cache_dic['full_counter'] += 1

            tc['cnt'] = cnt + 1
            if T > 0 and tc['cnt'] == T:
                tc['cnt'] = 0
            return

        prev = tc.get('previous_modulated_input', None)
        if mod_inp is None or prev is None:
            tc['previous_modulated_input'] = mod_inp
            current['type'] = 'full'
            current.setdefault('activated_steps', []).append(step)
            cache_dic['full_counter'] += 1

            tc['cnt'] = cnt + 1
            if T > 0 and tc['cnt'] == T:
                tc['cnt'] = 0
            return

        eps = 1e-12
        rel = (mod_inp - prev).abs().mean() / (prev.abs().mean() + eps)
        rel = float(rel.detach().cpu())

        coeff = tc.get('coefficients', None)
        if coeff is not None:
            rel_scaled = (((coeff[0] * rel + coeff[1]) * rel + coeff[2]) * rel + coeff[3]) * rel + coeff[4]
        else:
            rel_scaled = rel

        tc['accumulated_rel_l1_distance'] = float(tc.get('accumulated_rel_l1_distance', 0.0)) + float(rel_scaled)

        if tc['accumulated_rel_l1_distance'] < float(tc.get('rel_l1_thresh', 0.6)):
            current['type'] = 'TeaCacheSkip'
            print(f"TeaCache step {step}: skipping (cnt={cnt}, T={T}, rel={rel:.4f}, rel_scaled={rel_scaled:.4f}, accumulated_rel_l1_distance={tc['accumulated_rel_l1_distance']:.4f})")
        else:
            print(f"TeaCache step {step}: full (cnt={cnt}, T={T}, rel={rel:.4f}, rel_scaled={rel_scaled:.4f}, accumulated_rel_l1_distance={tc['accumulated_rel_l1_distance']:.4f})")
            current['type'] = 'full'
            current.setdefault('activated_steps', []).append(step)
            cache_dic['full_counter'] += 1
            tc['accumulated_rel_l1_distance'] = 0.0

        tc['previous_modulated_input'] = mod_inp
        tc['cnt'] = cnt + 1
        if T > 0 and tc['cnt'] == T:
            tc['cnt'] = 0
        return

    # -------------------------
    # DiCache branch
    # -------------------------
    if cache_dic.get('mode') == 'DiCache':
        if not cache_dic.get('dicache_enable', True):
            current['type'] = 'full'
            current['dicache_resume'] = False
            return

        branch = current.get('model', 'cond')
        dc = cache_dic['dicache'][branch]

        step = int(current.get('step', 0))
        T = int(dc.get('num_steps', current.get('num_steps', 0) or 0))
        cnt = int(dc.get('cnt', 0))
        ret_ratio = float(dc.get('ret_ratio', 0.2))

        current['type'] = 'full'
        current['dicache_resume'] = False

        # warmup + last step always full
        if (cnt <= int(ret_ratio * T)) or (T > 0 and cnt == T - 1):
            print(f"DiCache step {step}: warmup/last step full (cnt={cnt}, T={T})")
            dc['accumulated_rel_l1_distance'] = 0.0
            dc['resume_flag'] = False
            current['type'] = 'full'
            current.setdefault('activated_steps', []).append(step)
            cache_dic['full_counter'] += 1
            dc['cnt'] = cnt + 1
            if T > 0 and dc['cnt'] == T:
                dc['cnt'] = 0
            return

        dx = current.get('dicache_delta_x', None)
        dy = current.get('dicache_delta_y', None)

        if dx is None or dy is None:
            dc['accumulated_rel_l1_distance'] = 0.0
            dc['resume_flag'] = False
            current['type'] = 'full'
            current.setdefault('activated_steps', []).append(step)
            cache_dic['full_counter'] += 1

            dc['cnt'] = cnt + 1
            if T > 0 and dc['cnt'] == T:
                dc['cnt'] = 0
            return

        err_choice = dc.get('error_choice', 'delta_y')
        if err_choice == 'delta_minus':
            err = abs(float(dy) - float(dx))
        else:
            err = float(dy)

        dc['accumulated_rel_l1_distance'] = float(dc.get('accumulated_rel_l1_distance', 0.0)) + err
        if dc['accumulated_rel_l1_distance'] < float(dc.get('rel_l1_thresh', 0.4)):
            print(f"DiCache step {step}: skipping (cnt={cnt}, T={T}, err={err:.4f}, accumulated_err={dc['accumulated_rel_l1_distance']:.4f})")
            current['type'] = 'DiCacheSkip'
            current['dicache_resume'] = False
            dc['resume_flag'] = False
        else:
            print(f"DiCache step {step}: full (cnt={cnt}, T={T}, err={err:.4f}, accumulated_err={dc['accumulated_rel_l1_distance']:.4f})")
            current['type'] = 'full'
            current['dicache_resume'] = True
            dc['resume_flag'] = True
            dc['accumulated_rel_l1_distance'] = 0.0
            current.setdefault('activated_steps', []).append(step)
            cache_dic['full_counter'] += 1

        dc['cnt'] = cnt + 1
        if T > 0 and dc['cnt'] == T:
            dc['cnt'] = 0
        return
    
    # -------------------------
    # CFGCache branch
    # -------------------------
    if cache_dic.get('mode') == 'CFGCache':
        if not cache_dic.get('cfgcache_enable', True):
            current['type'] = 'full'
            current['cfgcache_resume'] = False
            return

        cc = cache_dic.get('cfgcache', {})
        joint_state = cc.get('joint_state', None)

        step = int(current.get('step', 0))
        branch = current.get('model', 'cond')
        bc = cc.get(branch, {})

        action = current.get('cfgcache_joint_action', 'full_both')

        current['type'] = 'full'
        current['cfgcache_resume'] = False

        # single-stream readiness:
        # CogVideoX CFGCache now reuses post-norm hidden only
        can_skip_local = (bc.get('previous_postnorm_hidden', None) is not None)
        if branch == 'uncond':
            print(
                f"CFGCache step {step} [{branch}]: "
                f"action={action}, can_skip_local={can_skip_local}"
            )

        # translate joint action -> local execution type
        # but guard against branch-local feature not ready
        if action == 'reuse_both' and can_skip_local:
            current['type'] = 'CFGCacheSkip'
            current['cfgcache_resume'] = False
        else:
            if action == 'reuse_both' and not can_skip_local:
                print(
                    f"CFGCache step {step} [{branch}]: "
                    f"reuse requested but postnorm feature not ready, fallback to full"
                )

            current['type'] = 'full'
            current['cfgcache_resume'] = True
            current.setdefault('activated_steps', []).append(step)
            cache_dic['full_counter'] += 1

        # bookkeeping
        current['cfgcache_can_skip_local'] = can_skip_local
        bc['last_action'] = action
        cc['last_action'] = action

        if joint_state is not None:
            joint_state['last_action'] = action

        return
    # -------------------------
    # MagCache branch
    # -------------------------
    if cache_dic.get('mode') == 'MagCache':
        if not cache_dic.get('magcache_enable', True):
            current['type'] = 'full'
            return

        branch = current.get('model', 'cond')
        mc = cache_dic['magcache'][branch]

        step = int(current.get('step', 0))
        T = int(mc.get('num_steps', current.get('num_steps', 0) or 0))
        cnt = int(mc.get('cnt', 0))

        current['type'] = 'full'

        has_residual = (
            mc.get('previous_residual', None) is not None
            and mc.get('previous_residual_encoder', None) is not None
        )

        if not has_residual:
            current.setdefault('activated_steps', []).append(step)
            cache_dic['full_counter'] = cache_dic.get('full_counter', 0) + 1
            mc['accumulated_ratio'] = 1.0
            mc['accumulated_steps'] = 0
            mc['accumulated_err'] = 0.0
            mc['cnt'] = (cnt + 1) % max(1, T)
            return

        skip_forward = False

        if cnt >= int(mc['retention_ratio'] * mc['num_steps'] + 0.5):
            cur_scale = float(mc['mag_ratios'][cnt])
            mc['accumulated_ratio'] *= cur_scale
            mc['accumulated_steps'] += 1

            local_err = float(abs(1.0 - mc['accumulated_ratio']))
            mc['accumulated_err'] += local_err

            if (
                mc['accumulated_err'] <= mc['magcache_thresh']
                and mc['accumulated_steps'] <= mc['K']
            ):
                skip_forward = True

        if skip_forward:
            current['type'] = 'MagCacheSkip'
            print(f"MagCache step {step}: skipping (cnt={cnt}, T={T}, cur_scale={cur_scale:.4f}, accumulated_ratio={mc['accumulated_ratio']:.4f}, accumulated_err={mc['accumulated_err']:.4f}, accumulated_steps={mc['accumulated_steps']})")
        else:
            current['type'] = 'full'
            print(f"MagCache step {step}: full (cnt={cnt}, T={T}, cur_scale={cur_scale if cnt >= int(mc['retention_ratio'] * mc['num_steps'] + 0.5) else 'N/A'}, accumulated_ratio={mc['accumulated_ratio']:.4f}, accumulated_err={mc['accumulated_err']:.4f}, accumulated_steps={mc['accumulated_steps']})")
            current.setdefault('activated_steps', []).append(step)
            cache_dic['full_counter'] = cache_dic.get('full_counter', 0) + 1

            # full refresh 后统一 reset
            mc['accumulated_ratio'] = 1.0
            mc['accumulated_steps'] = 0
            mc['accumulated_err'] = 0.0

        mc['cnt'] = (cnt + 1) % max(1, T)
        return
    # -------------------------
    # Taylor / HiCache / GroupedTaylor
    # -------------------------
    branch = current.get("model", "cond")
    branch_state = cache_dic["taylor_branch_state"][branch]
    current["activated_steps"] = branch_state["activated_steps"]

    first_step = (current["step"] < cache_dic["first_enhance"])

    if first_step or (branch_state["cache_counter"] == cache_dic["fresh_threshold"] - 1):
        current["type"] = "full"
        branch_state["cache_counter"] = 0
        branch_state["activated_steps"].append(current["step"])
        cache_dic["full_counter"] += 1

    elif cache_dic.get("use_grouped_taylor", False):
        current["type"] = "grouped_taylor_cache"
        branch_state["cache_counter"] += 1

    elif cache_dic["taylor_cache"]:
        current["type"] = "taylor_cache"
        branch_state["cache_counter"] += 1