from utils import *
from src import *
from functools import partial
from src.drv_feature_maps import build_drv_feature_maps, load_ap_lookup
from src.drv_predictor import DRVPredictor

def get_trunc_node_pos_fn(mov_node_size, data):
    node_pos_lb = mov_node_size / 2 + data.die_ll + 1e-4
    node_pos_ub = data.die_ur - mov_node_size / 2 + data.die_ll - 1e-4
    def trunc_node_pos_fn(x):
        x.data.clamp_(min=node_pos_lb, max=node_pos_ub)
        return x
    return trunc_node_pos_fn

def global_placement_main(gpdb, rawdb, ps: ParamScheduler, data: PlaceData, args, logger, params, gputimer=None, drv_predictor=None):
    init_density_map = data.init_density_map
    if not args.global_placement:
        logger.info("Global placement is switched off. Please make sure the input "
                    "placement solution is already placed globally.")
        node_pos, iteration = data.node_pos, 0
        hpwl, overflow = evaluate_placement(node_pos, init_density_map, ps, data, args)
        hpwl, overflow = hpwl.item(), overflow.item()
        info = ("%d_gp" % (iteration + 1), hpwl, data.design_name)
        if args.draw_placement:
            draw_fig_with_cairo_cpp(node_pos, data.node_size, data, info, args)
        logger.info("Input solution, exact HPWL: %.6E exact Overflow: %.4f" % (hpwl, overflow))
        gp_hpwl, overflow, gp_time, gp_per_iter = hpwl, overflow, 0, -1
        return node_pos, iteration, gp_hpwl, overflow, gp_time, gp_per_iter

    device = data.device
    mov_lhs, mov_rhs = data.movable_index
    mov_node_pos, mov_node_size, expand_ratio = data.get_mov_node_info()
    mov_node_pos = mov_node_pos.requires_grad_(True)

    trunc_node_pos_fn = get_trunc_node_pos_fn(mov_node_size, data)

    conn_fix_node_pos = data.node_pos.new_empty(0, 2)
    if data.fixed_connected_index[0] < data.fixed_connected_index[1]:
        lhs, rhs = data.fixed_connected_index
        conn_fix_node_pos = data.node_pos[lhs:rhs, ...]
    conn_fix_node_pos = conn_fix_node_pos.detach()

    def overflow_fn(mov_density_map):
        overflow_sum = ((mov_density_map - args.target_density) * data.bin_area).clamp_(min=0.0).sum()
        return overflow_sum / data.total_mov_area_without_filler
    overflow_helper = (mov_lhs, mov_rhs, overflow_fn)

    density_map_layer = ElectronicDensityLayer(
        unit_len=data.unit_len,
        num_bin_x=data.num_bin_x,
        num_bin_y=data.num_bin_y,
        device=device,
        overflow_helper=overflow_helper,
        sorted_maps=data.sorted_maps,
        expand_ratio=expand_ratio,
        deterministic=args.deterministic,
    ).to(device)

    # fix_lhs, fix_rhs = data.fixed_index
    # info = (0, 0, data.design_name + "_fix")
    # fix_node_pos = data.node_pos[fix_lhs:fix_rhs, ...]
    # fix_node_size = data.node_size[fix_lhs:fix_rhs, ...]
    # draw_fig_with_cairo(
    #     None, None, fix_node_pos, fix_node_size, None, None, data, info, args
    # )
    def calc_route_force(mov_node_pos, mov_node_size, expand_ratio, constraint_fn):
        return get_route_force(
            args, logger, data, rawdb, gpdb, ps, mov_node_pos, mov_node_size, expand_ratio,
            constraint_fn=constraint_fn
        )

    obj_and_grad_fn = partial(
        calc_obj_and_grad,
        constraint_fn=trunc_node_pos_fn,
        route_fn=calc_route_force,
        mov_node_size=mov_node_size,
        expand_ratio=expand_ratio,
        init_density_map=init_density_map,
        density_map_layer=density_map_layer,
        conn_fix_node_pos=conn_fix_node_pos,
        ps=ps,
        data=data,
        args=args,
    )
    evaluator_fn = partial(
        fast_evaluator,
        constraint_fn=trunc_node_pos_fn,
        mov_node_size=mov_node_size,
        init_density_map=init_density_map,
        density_map_layer=density_map_layer,
        conn_fix_node_pos=conn_fix_node_pos,
        ps=ps,
        data=data,
        args=args,
    )
    optimizer = NesterovOptimizer([mov_node_pos], lr=0)

    # initialization
    init_params(
        mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos,
        density_map_layer, mov_node_size, expand_ratio, init_density_map, optimizer,
        ps, data, args, route_fn=calc_route_force
    )
    # init learnig rate
    init_lr = estimate_initial_learning_rate(obj_and_grad_fn, trunc_node_pos_fn, mov_node_pos, args.lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = init_lr.item()

    torch.cuda.synchronize(device)
    gp_start_time = time.time()
    logger.info("start gp")

    # def trace_handler(prof):
    #     print(prof.key_averages().table(
    #         sort_by="self_cuda_time_total", row_limit=-1))
    #     prof.export_chrome_trace("test_trace_" + str(prof.step_num) + ".json")
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ], schedule=torch.profiler.schedule(
    #         wait=2,
    #         warmup=2,
    #         active=2),
    #     on_trace_ready=trace_handler
    #     ) as p:
    #         for iter in range(6):
    #             # optimizer.zero_grad()
    #             obj = optimizer.step(obj_and_grad_fn)
    #             hpwl, overflow = evaluator_fn(mov_node_pos)
    #             # update parameters
    #             ps.step(hpwl, overflow, mov_node_pos, data)
    #             if ps.need_to_early_stop():
    #                 break
    #             p.step()
    # exit(0)
    terminate_signal = False
    route_early_terminate_signal = False
    log_info = False
    timing_cali_thrs_overflow = 0.5
    timing_calibration = False
    for iteration in range(args.inner_iter):
        # optimizer.zero_grad() # zero grad inside obj_and_grad_fn
        obj = optimizer.step(obj_and_grad_fn)
        hpwl, overflow = evaluator_fn(mov_node_pos)
        # update parameters
        ps.step(hpwl, overflow, mov_node_pos, data)

        # Perform timing-opt.
        if args.timing_opt and iteration > args.timing_start_iter and iteration % 1 == 0:
            ps.enable_timing = True
            node_pos = torch.cat([mov_node_pos[mov_lhs:mov_rhs].clone(), data.node_pos[mov_rhs:]], dim=0)

            if args.calibration and ps.recorder.overflow[-1] < timing_cali_thrs_overflow:
                gputimer.update_timing_calibrated(node_pos, record=True)
                timing_cali_thrs_overflow -= args.calibration_step
                timing_calibration = True
            elif timing_calibration:
                gputimer.update_timing_calibrated(node_pos)
            else:
                gputimer.update_timing(node_pos)

            timing_metrics = gputimer.report_timing_slack()
            wns_early, tns_early, wns_late, tns_late = timing_metrics
            ps.push_timing_sol(timing_metrics, hpwl, overflow, mov_node_pos)

            if iteration % args.timing_freq == 0:
                gputimer.step(ps, node_pos, data)

        if ps.need_to_early_stop():
            terminate_signal = True
            log_info = True

        if ps.enable_mixed_size and not ps.zero_macro_grad and terminate_signal:
            ps.zero_macro_grad = True
            # Find best gp node_pos (including macros and std cells)
            best_res = ps.get_best_solution()
            if best_res[0] is not None:
                best_sol, hpwl, overflow = best_res
                # fillers are unused from now, we don't copy there data
                mov_node_pos[mov_lhs:mov_rhs].data.copy_(best_sol[mov_lhs:mov_rhs])
            node_pos = mov_node_pos[mov_lhs:mov_rhs]
            node_pos = torch.cat([node_pos, data.node_pos[mov_rhs:]], dim=0)
            # Evaluate the mixed placement solution
            hpwl, overflow = evaluate_placement(node_pos, init_density_map, ps, data, args)
            hpwl, overflow = hpwl.item(), overflow.item()
            if args.draw_placement:
                info = ("%d_mixed_gp" % (iteration + 1), hpwl, data.design_name)
                draw_fig_with_cairo_cpp(node_pos, data.node_size, data, info, args)
            logger.info("After Mixed-GP, best solution eval, exact HPWL: %.6E exact Overflow: %.4f" % (hpwl, overflow))
            # Run macro legalization to change node_pos inplace
            macro_legalization_main(node_pos, data, args, logger)
            if args.draw_placement:
                info = ("%d_mixed_gp_ml" % (iteration + 1), hpwl, data.design_name)
                draw_fig_with_cairo_cpp(node_pos, data.node_size, data, info, args)
            # Write node_pos into database to provide an initial solution for std cell placement
            data.node_pos[mov_lhs:mov_rhs].data.copy_(node_pos[mov_lhs:mov_rhs])
            # Prepare for std cell placement
            init_density_map = get_init_density_map(rawdb, gpdb, data, args, logger, ps=ps)
            data.__total_mov_area_without_filler__ = torch.sum(data.node_area[mov_lhs:mov_rhs][torch.logical_not(data.is_mov_macro[mov_lhs:mov_rhs])]).item()
            mov_node_pos, mov_node_size, expand_ratio = data.get_mov_node_info(init_method="randn_center")
            mov_macros_idx = data.is_mov_macro[mov_lhs:mov_rhs]
            mov_node_pos[mov_lhs:mov_rhs][mov_macros_idx] = data.node_pos[mov_lhs:mov_rhs][mov_macros_idx]
            mov_node_pos = mov_node_pos.requires_grad_(True)
            trunc_node_pos_fn = get_trunc_node_pos_fn(mov_node_size, data)
            density_map_layer.expand_ratio = expand_ratio
            density_map_layer.sorted_maps = data.sorted_maps
            # ignore the density and grad computation of macros by node_weight
            density_map_layer.cache_node_weight[mov_lhs:mov_rhs][mov_macros_idx] = -1.0
            # update partial function correspondingly
            obj_and_grad_fn.keywords["constraint_fn"] = trunc_node_pos_fn
            obj_and_grad_fn.keywords["mov_node_size"] = mov_node_size
            obj_and_grad_fn.keywords["expand_ratio"] = expand_ratio
            obj_and_grad_fn.keywords["init_density_map"] = init_density_map
            evaluator_fn.keywords["constraint_fn"] = trunc_node_pos_fn
            evaluator_fn.keywords["mov_node_size"] = mov_node_size
            evaluator_fn.keywords["init_density_map"] = init_density_map
            # reset nesterov optimizer
            logger.info("Reset optimizer...")
            optimizer = NesterovOptimizer([mov_node_pos], lr=0)
            # initialization
            init_params(
                mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos,
                density_map_layer, mov_node_size, expand_ratio, init_density_map, optimizer,
                ps, data, args, route_fn=calc_route_force
            )
            # init learnig rate
            cur_lr = estimate_initial_learning_rate(obj_and_grad_fn, trunc_node_pos_fn, mov_node_pos, args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = cur_lr.item()
            ps.reset_best_sol()
            ps.drv_grad = None  # stale gradient is invalid after position reset
            terminate_signal = False  # reset signal
            logger.info("Re-run std cell placement with fixed macros.")

        if ps.use_cell_inflate and ps.curr_optimizer_cnt < ps.max_route_opt and terminate_signal:
            terminate_signal = False  # reset signal
            ps.start_route_opt = True
            ps.curr_optimizer_cnt += 1
            best_res = ps.get_best_solution()
            if best_res[0] is not None:
                best_sol, hpwl, overflow = best_res
                mov_node_pos.data.copy_(best_sol)

        if ps.use_route_force:
            if ps.iter > 100 and ps.enable_route:
                if ps.recorder.overflow[-1] < 0.2 and ps.recorder.overflow[-2] >= 0.2 and not ps.start_route_opt:
                    ps.start_route_opt = True
                    ps.curr_optimizer_cnt += 1
                # if ps.recorder.overflow[-2] < 0.2 and ps.recorder.overflow[-1] >= 0.2 and ps.start_route_opt:
                #     ps.start_route_opt = False
                #     ps.curr_optimizer_cnt += 1

        # if ps.use_cell_inflate and ps.curr_optimizer_cnt < ps.max_route_opt:
        #     if ps.iter > 100 and ps.enable_route:
        #         if ps.recorder.overflow[-1] < 0.15 and ps.recorder.overflow[-2] >= 0.15 and not ps.start_route_opt:
        #             ps.start_route_opt = True
        #             ps.curr_optimizer_cnt += 1
        #         if ps.recorder.overflow[-2] < 0.15 and ps.recorder.overflow[-1] >= 0.15 and ps.start_route_opt:
        #             ps.start_route_opt = False

        if ps.start_route_opt and ps.enable_route:
            if (ps.iter % args.route_freq == 0 and ps.use_route_force) or \
                  (ps.curr_optimizer_cnt != ps.prev_optimizer_cnt and ps.curr_optimizer_cnt <= ps.max_route_opt):
                ps.rerun_route = True
            else:
                ps.rerun_route = False
        else:
            ps.rerun_route = False

        if ps.rerun_route:
            log_info = True
            # Release cached GPU memory before GR to avoid fragmentation from DRV backward passes
            if ps.drv_grad is not None:
                torch.cuda.empty_cache()
            new_mov_node_size, new_expand_ratio = None, None
            if ps.use_cell_inflate:
                output = route_inflation(
                    args, logger, data, rawdb, gpdb, ps, mov_node_pos, mov_node_size, expand_ratio,
                    constraint_fn=trunc_node_pos_fn, visualize=args.visualize_cgmap
                )  # ps.use_cell_inflate is updated in route_inflation
                if not ps.use_cell_inflate:
                    route_early_terminate_signal = True
                    terminate_signal = True
                    logger.info("Early stop cell inflation...")
                if output is not None:
                    gr_metrics, new_mov_node_size, new_expand_ratio = output
                    ps.push_gr_sol(gr_metrics, hpwl, overflow, mov_node_pos)
            route_fn=None
            if ps.use_route_force:
                route_fn=calc_route_force
            ps.prev_optimizer_cnt = ps.curr_optimizer_cnt
            if ps.use_cell_inflate or ps.use_route_force:
                logger.info("Reset optimizer...")
                if new_mov_node_size is not None:
                    # remove some fillers, we should update the size the pos
                    mov_node_size = new_mov_node_size
                    mov_node_pos = mov_node_pos[:new_mov_node_size.shape[0]].detach().clone()
                    mov_node_pos = mov_node_pos.requires_grad_(True)
                    # update expand ratio and precondition relevant data
                    expand_ratio = new_expand_ratio  # already update in route_inflation()
                    data.mov_node_to_num_pins = data.mov_node_to_num_pins[:new_mov_node_size.shape[0]]
                    data.mov_node_area = data.mov_node_area  # already update in route_inflation()
                    # update partial function correspondingly
                    trunc_node_pos_fn = get_trunc_node_pos_fn(mov_node_size, data)
                    obj_and_grad_fn.keywords["constraint_fn"] = trunc_node_pos_fn
                    obj_and_grad_fn.keywords["mov_node_size"] = mov_node_size
                    obj_and_grad_fn.keywords["expand_ratio"] = expand_ratio
                    evaluator_fn.keywords["constraint_fn"] = trunc_node_pos_fn
                    evaluator_fn.keywords["mov_node_size"] = mov_node_size
                    density_map_layer.expand_ratio = expand_ratio
                # reset nesterov optimizer
                optimizer = NesterovOptimizer([mov_node_pos], lr=0)
                init_params(
                    mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos,
                    density_map_layer, mov_node_size, expand_ratio, init_density_map, optimizer,
                    ps, data, args, route_fn=route_fn
                )
                cur_lr = estimate_initial_learning_rate(obj_and_grad_fn, trunc_node_pos_fn, mov_node_pos, args.lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = cur_lr.item()
                logger.info(
                    "Route Iter: %d | lr: %.2E density_weight: %.2E route_weight: %.2E "
                    "congest_weight: %.2E pseudo_weight: %.2E "
                    % (
                        ps.curr_optimizer_cnt - 1,
                        cur_lr.item(),
                        ps.density_weight,
                        ps.route_weight,
                        ps.congest_weight,
                        ps.pseudo_weight,
                    )
                )
                ps.reset_best_sol()
                ps.drv_grad = None  # stale gradient is invalid after position/size reset

        # ── DRV prediction (Phase 1: visualisation only, no force) ───────────
        if drv_predictor is not None and \
                DRVPredictor.should_predict(overflow, iteration,
                                            args.drv_pred_overflow,
                                            args.drv_pred_freq):
            feat = build_drv_feature_maps(
                mov_node_pos, data, gpdb,
                ap_lookup=drv_predictor.ap_lookup,
                grid_size=args.drv_pred_resolution,
            )
            drv_predictor.step(feat, iteration, overflow)

        if iteration % args.log_freq == 0 or iteration == args.inner_iter - 1 or log_info:
            log_info = False
            log_str = (
                "iter: %d | masked_hpwl: %.2E overflow: %.4f obj: %.4E "
                "density_weight: %.4E wa_coeff: %.4E"
                % (
                    iteration,
                    hpwl,
                    overflow,
                    obj,
                    ps.density_weight,
                    ps.wa_coeff,
                )
            )
            if ps.enable_timing:
                log_str += " | early WNS/TNS: %.4f %.4f (ns) | late WNS/TNS: %.4f %.4f (ns)" % (
                    wns_early, tns_early, wns_late, tns_late
                )
            logger.info(log_str)
            if args.draw_placement:
                info = (iteration, hpwl, data.design_name)
                node_pos_to_draw = mov_node_pos[mov_lhs:mov_rhs, ...].clone()
                node_size_to_draw = data.node_size[mov_lhs:mov_rhs, ...].clone()
                node_pos_to_draw = torch.cat(
                    [node_pos_to_draw, data.node_pos[mov_rhs:, ...].clone()], dim=0
                )
                node_size_to_draw = torch.cat(
                    [node_size_to_draw, data.node_size[mov_rhs:, ...].clone()], dim=0
                )
                if args.use_filler:
                    node_pos_to_draw = torch.cat(
                        [node_pos_to_draw, mov_node_pos[mov_rhs:, ...].clone()], dim=0
                    )
                    node_size_filler_to_draw = data.filler_size[:(mov_node_pos.shape[0] - mov_rhs), ...]
                    node_size_to_draw = torch.cat(
                        [node_size_to_draw, node_size_filler_to_draw], dim=0
                    )
                draw_fig_with_cairo_cpp(
                    node_pos_to_draw, node_size_to_draw, data, info, args
                )

        if terminate_signal:
            break

    # Save best solution
    best_res = ps.get_best_solution()
    if best_res[0] is not None:
        best_sol, hpwl, overflow = best_res
        # fillers are unused from now, we don't copy there data
        mov_node_pos[mov_lhs:mov_rhs].data.copy_(best_sol[mov_lhs:mov_rhs])
    if ps.enable_mixed_size and ps.zero_macro_grad:
        # rollback macro_pos to previous legalized results since trunc_node_pos_fn may change them
        mov_macros_idx = data.is_mov_macro[mov_lhs:mov_rhs]
        mov_node_pos.data[mov_lhs:mov_rhs][mov_macros_idx] = data.node_pos[mov_lhs:mov_rhs][mov_macros_idx]

    # ── Phase 2: DRV fine-tuning (pure SGD on DRV L2 loss) ────────────────────
    drv_finetune_iters = getattr(args, 'drv_finetune_iters', 0)
    if drv_predictor is not None and ps.drv_weight > 0 and drv_finetune_iters > 0:
        drv_predictor.out_dir = drv_predictor.phase2_out_dir  # switch to phase2 dir
        logger.info(f"[DRV] Starting fine-tuning phase ({drv_finetune_iters} iters, "
                    f"drv_weight={ps.drv_weight})")
        best_ft_loss = float('inf')
        best_ft_pos = mov_node_pos[mov_lhs:mov_rhs].detach().clone()

        # Calibrate step size from first gradient norm
        torch.cuda.empty_cache()
        drv_grad_raw, drv_l2_loss = drv_predictor.compute_force(
            mov_node_pos, data, gpdb, grid_size=args.drv_pred_resolution)
        torch.cuda.synchronize()
        grad_norm = drv_grad_raw[mov_lhs:mov_rhs].norm(p=2).clamp(min=1e-12)
        # target: move ~1% of die width per step
        die_width = (data.die_ur - data.die_ll)[0].item()
        ft_lr = ps.drv_weight * die_width * 0.01 / grad_norm.item()
        logger.info(f"[DRV] SGD lr={ft_lr:.4E}  grad_norm={grad_norm:.4E}  die_width={die_width:.2f}  DRV_L2={drv_l2_loss:.4E}")

        for ft_i in range(drv_finetune_iters):
            # Recompute gradient every step (no caching — we're debugging gradient descent)
            torch.cuda.empty_cache()
            drv_grad_raw, drv_l2_loss = drv_predictor.compute_force(
                mov_node_pos, data, gpdb, grid_size=args.drv_pred_resolution)
            torch.cuda.synchronize()

            # Pure gradient descent update on movable nodes only
            with torch.no_grad():
                mov_node_pos[mov_lhs:mov_rhs].data -= ft_lr * drv_grad_raw[mov_lhs:mov_rhs]
            trunc_node_pos_fn(mov_node_pos)

            if drv_l2_loss < best_ft_loss:
                best_ft_loss = drv_l2_loss
                best_ft_pos = mov_node_pos[mov_lhs:mov_rhs].detach().clone()

            if ft_i % 10 == 0:
                hpwl, overflow = evaluator_fn(mov_node_pos)
                logger.info(
                    f"[DRV finetune] {ft_i}/{drv_finetune_iters} | "
                    f"hpwl={hpwl:.4E} overflow={overflow:.4f} DRV_L2={drv_l2_loss:.4E} lr={ft_lr:.4E}"
                )
                feat_vis = build_drv_feature_maps(
                    mov_node_pos, data, gpdb,
                    ap_lookup=drv_predictor.ap_lookup,
                    grid_size=args.drv_pred_resolution,
                )
                drv_pred_vis = drv_predictor.step(feat_vis, ft_i, overflow)
                if drv_predictor._has_mpl:
                    stem = f"{drv_predictor.design_name}_iter{ft_i:05d}_ovfl{overflow:.3f}"
                    drv_predictor.save_drv_png(
                        drv_pred_vis, ft_i, overflow, drv_l2_loss,
                        os.path.join(drv_predictor.out_dir, stem + '_drv.png'),
                    )

        mov_node_pos[mov_lhs:mov_rhs].data.copy_(best_ft_pos)
        ps.drv_grad = None
        logger.info(f"[DRV] Fine-tuning done. Best DRV L2: {best_ft_loss:.4E}")

    # Free DRV model from GPU before GR to avoid VRAM contention
    if drv_predictor is not None:
        drv_predictor.model.cpu()
        drv_predictor = None
        torch.cuda.empty_cache()

    if ps.enable_route:
        route_inflation_roll_back(args, logger, data, mov_node_size)
        if not route_early_terminate_signal:
            ps.rerun_route = True
            gr_metrics = run_gr_and_fft_main(
                args, logger, data, rawdb, gpdb, ps, mov_node_pos, constraint_fn=trunc_node_pos_fn,
                skip_m1_route=True, report_gr_metrics_only=True, visualize=args.visualize_cgmap
            )
            ps.rerun_route = False
            ps.push_gr_sol(gr_metrics, hpwl, overflow, mov_node_pos)
        best_sol_gr = ps.get_best_gr_sol()
        mov_node_pos[mov_lhs:mov_rhs].data.copy_(best_sol_gr[mov_lhs:mov_rhs])
    if ps.enable_timing and not ps.enable_route:
        best_sol_timing = ps.get_best_timing_sol()
        if best_sol_timing is not None:
            mov_node_pos[mov_lhs:mov_rhs].data.copy_(best_sol_timing[mov_lhs:mov_rhs])

    node_pos = mov_node_pos[mov_lhs:mov_rhs]
    node_pos = torch.cat([node_pos, data.node_pos[mov_rhs:]], dim=0)
    torch.cuda.synchronize(device)
    gp_end_time = time.time()
    gp_time = gp_end_time - gp_start_time
    gp_per_iter = gp_time / (iteration + 1)
    logger.info("GP Stop! #Iters %d masked_hpwl: %.6E overflow: %.4f GP Time: %.4fs perIterTime: %.6fs" %
        (iteration, hpwl, overflow, gp_time, gp_time / (iteration + 1))
    )

    # Eval
    hpwl, overflow = evaluate_placement(node_pos, init_density_map, ps, data, args)
    hpwl, overflow = hpwl.item(), overflow.item()
    info = ("%d_gp" % (iteration + 1), hpwl, data.design_name)
    if args.draw_placement:
        draw_fig_with_cairo_cpp(node_pos, data.node_size, data, info, args)
        draw_placement_with_pdn(node_pos, data.node_size, gpdb, data, info, args)
        draw_bin_density_map(node_pos, data.node_size, data, info, args)
    logger.info("After GP, best solution eval, exact HPWL: %.6E exact Overflow: %.4f" % (hpwl, overflow))
    ps.visualize(args, logger)
    gp_hpwl = hpwl
    gp_time = gp_end_time - gp_start_time
    iteration += 1 # increase 1 For DP drawing

    return node_pos, iteration, gp_hpwl, overflow, gp_time, gp_per_iter


def run_placement_main_nesterov(args, logger):
    total_start = time.time()
    params = find_design_params(args, logger)
    data, rawdb, gpdb = load_dataset(args, logger, params)
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    assert args.use_eplace_nesterov
    logger.info("Start place %s/%s" % (args.dataset , args.design_name))
    logger.info("Use Nesterov optimizer!")
    data = data.to(device)
    data = data.preprocess()
    logger.info(data)
    logger.info(data.node_type_indices)
    # args.num_bin_x = args.num_bin_y = 2 ** math.ceil(math.log2(max(data.die_info).item() // 25))
    get_init_density_map(rawdb, gpdb, data, args, logger)
    data.init_filler()

    ps = ParamScheduler(data, args, logger)

    gputimer = None
    if args.timing_opt:
        gputimer = GPUTimer(data, rawdb, gpdb, params, args)
        data.gputimer = gputimer
        def timing_eval_func(node_pos):
            gputimer.update_timing_eval(node_pos)
            wns_early, tns_early, wns_late, tns_late = gputimer.report_timing_slack()
            logger.info("early WNS/TNS: %.4f/%.4f (ns) | late WNS/TNS: %.4f/%.4f (ns)" % (wns_early, tns_early, wns_late, tns_late))
            return wns_early, tns_early, wns_late, tns_late

    # ── DRV predictor (optional) ───────────────────────────────────────────────
    drv_predictor = None
    if getattr(args, 'drv_checkpoint', '') and os.path.isfile(args.drv_checkpoint):
        drv_out = os.path.join(
            args.result_dir, args.exp_id, args.output_dir, 'drv_pred', 'phase1')
        drv_predictor = DRVPredictor(
            args.drv_checkpoint, device, drv_out,
            design_name=args.design_name)
        drv_predictor.phase2_out_dir = os.path.join(
            args.result_dir, args.exp_id, args.output_dir, 'drv_pred', 'phase2')
        os.makedirs(drv_predictor.phase2_out_dir, exist_ok=True)
        ap_paths = [p.strip() for p in args.drv_ap_json.split(',') if p.strip()]
        drv_predictor.ap_lookup = load_ap_lookup(ap_paths)
        logger.info(
            f"DRVPredictor enabled | start_overflow={args.drv_pred_overflow} "
            f"freq={args.drv_pred_freq} resolution={args.drv_pred_resolution} "
            f"ap_channels={'yes' if drv_predictor.ap_lookup else 'no (pin-count fallback)'}")
    elif getattr(args, 'drv_checkpoint', ''):
        logger.warning(f"DRV checkpoint not found: {args.drv_checkpoint} — prediction disabled")

    # global placement
    node_pos, iteration, gp_hpwl, overflow, gp_time, gp_per_iter = global_placement_main(
        gpdb, rawdb, ps, data, args, logger, params, gputimer,
        drv_predictor=drv_predictor,
    )
    if args.timing_opt:
        wns_early_gp, tns_early_gp, wns_late_gp, tns_late_gp = timing_eval_func(node_pos)

    # detail placement
    node_pos, dp_hpwl, top5overflow, lg_time, dp_time = detail_placement_main(
        node_pos, gpdb, rawdb, ps, data, args, logger
    )
    if args.timing_opt:
        wns_early_dp, tns_early_dp, wns_late_dp, tns_late_dp = timing_eval_func(node_pos)
    iteration += 1

    route_metrics = None
    if ps.enable_route and args.final_route_eval:
        logger.info("Final routing evalution by GGR...")
        route_metrics = run_gr_and_fft(
            args, logger, data, rawdb, gpdb, ps,
            report_gr_metrics_only=True,
            skip_m1_route=True, given_gr_params={
                "rrrIters": 1,
                "route_guide": os.path.join(args.result_dir, args.exp_id, args.output_dir, "%s_%s.guide" %(args.output_prefix, args.design_name)),
            }
        )

    if args.load_from_raw:
        del gpdb, rawdb
        del gputimer

    place_time = time.time() - total_start
    logger.info("GP Time: %.4f LG Time: %.4f DP Time: %.4f Total Place Time: %.4f" % (
        gp_time, lg_time, dp_time, place_time))
    place_metrics = (dp_hpwl, gp_hpwl, top5overflow, overflow, gp_time, dp_time + lg_time, gp_per_iter, place_time)
    if args.timing_opt:
        place_metrics += (wns_early_dp, tns_early_dp, wns_late_dp, tns_late_dp)

    return place_metrics, route_metrics
