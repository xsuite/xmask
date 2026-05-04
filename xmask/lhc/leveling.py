from scipy.constants import c as clight

import xtrack as xt

def luminosity_leveling(collider, config_lumi_leveling, config_beambeam):

    opts = {}
    for ip_name in config_lumi_leveling.keys():

        print(f'\n --- Leveling in {ip_name} ---')

        config_this_ip = config_lumi_leveling[ip_name]
        bump_range = config_this_ip['bump_range']

        assert config_this_ip['preserve_angles_at_ip'], (
            'Only preserve_angles_at_ip=True is supported for now')
        assert config_this_ip['preserve_bump_closure'], (
            'Only preserve_bump_closure=True is supported for now')

        if 'b1' in collider.lines:
            line_names = ['b1', 'b2']
        else:
            line_names = ['lhcb1', 'lhcb2']
        line_b1 = collider[line_names[0]]

        beta0_b1 = line_b1.particle_ref.beta0[0]
        f_rev=1/(line_b1.get_length() /(beta0_b1 * clight))

        targets=[]
        vary=[]

        if 'luminosity' in config_this_ip.keys():
            targets.append(
                xt.TargetLuminosity(
                    ip_name=ip_name, luminosity=config_this_ip['luminosity'], crab=False,
                    tol=0.001 * config_this_ip['luminosity'],
                    f_rev=f_rev, num_colliding_bunches=config_this_ip['num_colliding_bunches'],
                    num_particles_per_bunch=config_beambeam['num_particles_per_bunch'],
                    sigma_z=config_beambeam['sigma_z'],
                    nemitt_x=config_beambeam['nemitt_x'],
                    nemitt_y=config_beambeam['nemitt_y'],
                    log=True)
            )
        elif 'separation_in_sigmas' in config_this_ip.keys():
            targets.append(
                xt.TargetSeparation(
                    ip_name=ip_name,
                    separation_norm=config_this_ip['separation_in_sigmas'],
                    tol=1e-4, # in sigmas
                    plane=config_this_ip['plane'],
                    nemitt_x=config_beambeam['nemitt_x'],
                    nemitt_y=config_beambeam['nemitt_y'])
            )
        else:
            raise ValueError('Either `luminosity` or `separation_in_sigmas` must be specified')

        tw0 = collider.twiss(lines=line_names,
                             chrom=False)
        if config_this_ip['impose_separation_orthogonal_to_crossing']:
            targets.append(
                xt.TargetSeparationOrthogonalToCrossing(ip_name='ip8'))
        for knob_name in config_this_ip['knobs']:
            vv = collider.vars[knob_name]._value
            # Preserve knob sign
            if vv >= 0:
                ll = (0, 1e200)
            else:
                ll = (-1e200, 0)
            vary.append(xt.Vary(name=knob_name, step=1e-7, limits=ll))

        # Target and knobs to rematch the crossing angles and close the bumps
        for line_name in line_names:
            targets += [
                # Preserve crossing angle
                xt.TargetList(['px', 'py'], at=ip_name, line=line_name, value=tw0, tol=1e-7, scale=1e3),
                # Close the bumps
                xt.TargetList(['x', 'y'], at=bump_range[line_name][-1], line=line_name, value=tw0, tol=1e-5, scale=1),
                xt.TargetList(['px', 'py'], at=bump_range[line_name][-1], line=line_name, value=tw0, tol=1e-5, scale=1e3),
            ]

        vary.append(xt.VaryList(config_this_ip['corrector_knob_names'], step=1e-7))

        # Match
        opt = collider.match(
            lines=line_names,
            start=[bump_range[line_names[0]][0], bump_range[line_names[1]][0]],
            end=[bump_range[line_names[0]][-1], bump_range[line_names[1]][-1]],
            init=tw0, init_at=xt.START,
            targets=targets,
            vary=vary,
            solve=False,
        )
        opt.solve()
        opts[ip_name] = opt
    return opts
