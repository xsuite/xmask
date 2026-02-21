
import numpy as np
import xtrack as xt
from integral_correction import IntegralCorrection

beam_name = 'b1'
env = xt.load('collider_00_from_mad_with_errors.json')
line = env[f'lhc{beam_name}']

# Reference twiss
env_no_err = xt.load('collider_00_from_mad_no_errors.json')
tw = env_no_err[f'lhc{beam_name}'].twiss4d() # Reference twiss


tt = line.get_table()
scale_multipole = np.zeros_like(tt.s)
scale_multipole[tt.rows.mask[r'mb.*']] = 1.0 # only bends as sources
scale_multipole[tt.rows.mask[r'mc.*']] = 1.0 # all magnets called mcXXX used as correctors
scale_multipole[tt.rows.mask[r'mss.*']] = 1.0 # all magnets called mssXXX used as correctors


# Let's have a look at the a3 (skew sextupole)

def chorm_coupling_integrand(tw, tt):
    return tt['k2sl'] * tw.dx * np.sqrt(tw.betx * tw.bety) * np.exp(1j*2*np.pi*(tw.mux - tw.muy))

arcs = ['12', '23', '34', '45', '56', '67', '78', '81']

# Global correction setup (we run it after logal)
start = tw.name[0]
end = tw.name[-1]
correction_knobs = [f'kss.a{arc_name}{beam_name}' for arc_name in arcs]
generated_knob_name = f'on_corr_k2sl_global'
multipole = 'k2sl'
target_quantities={
    'chrom_coupling_real': lambda tw, tt: chorm_coupling_integrand(tw, tt).real,
    'chrom_coupling_imag': lambda tw, tt: chorm_coupling_integrand(tw, tt).imag
}

rdt_contrib_glob = IntegralCorrection(
                        line=env['lhcb1'],
                        tw=tw,
                        feed_down=True,
                        start=start,
                        end=end,
                        correction_knobs=correction_knobs,
                        multipole=multipole,
                        ip=None,
                        target_quantities=target_quantities,
                        generated_knob_name=generated_knob_name,
                        scale_multipole=scale_multipole)
print("Original correction:")
rdt_contrib_glob.print_corrections()
knobs_original = rdt_contrib_glob.get_corrections()

opt_dct = {}
integ_dct = {}
for arc_name in arcs:
    start = f's.ds.r{arc_name[0]}.{beam_name}'
    end = f'e.ds.l{arc_name[1]}.{beam_name}'
    correction_knobs = [f'kss.a{arc_name}{beam_name}']
    generated_knob_name = f'on_corr_k2sl_a{arc_name}_local'
    multipole = 'k2sl'
    target_quantities={
        'chrom_coupling_real': lambda tw, tt: chorm_coupling_integrand(tw, tt).real,
        'chrom_coupling_imag': lambda tw, tt: chorm_coupling_integrand(tw, tt).imag
    }

    # Usage:
    arc_integ = IntegralCorrection(
                            line=env['lhcb1'],
                            tw=tw,
                            feed_down=True,
                            start=start,
                            end=end,
                            correction_knobs=correction_knobs,
                            multipole=multipole,
                            ip=None,
                            target_quantities=target_quantities,
                            generated_knob_name=generated_knob_name,
                            scale_multipole=scale_multipole)
    print("Original correction:")
    arc_integ.print_corrections()

    arc_integ.clear_corrections()
    opt = arc_integ.correct()

    print("Before setting the knob:")
    arc_integ.print_corrections()

    env[opt.knob_name] = 1.0
    print("After setting the knob:")
    arc_integ.print_corrections()

    opt_dct[arc_name] = opt
    integ_dct[arc_name] = arc_integ

# Global correction

# rdt_contrib.clear_corrections() # BAD!!!
opt = rdt_contrib_glob.correct()

print("Before setting the knob:")
rdt_contrib_glob.print_corrections()

env[opt.knob_name] = 1.0
print("After setting the knob:")
knobs_final = rdt_contrib_glob.get_corrections()
rdt_contrib_glob.print_corrections()

local_dct_opt = {}
for arc_name in arcs:
    oo = integ_dct[arc_name].run()
    local_dct_opt[arc_name] = oo['chrom_coupling_real'] + 1j * oo['chrom_coupling_imag']

tw_opt = env['lhcb1'].twiss4d(coupling_edw_teng=True, delta0=1e-3)
nlchr_opt = line.get_non_linear_chromaticity(num_delta=20)


env.vars.update(knobs_original)
local_dct_orig = {}
for arc_name in arcs:
    oo = integ_dct[arc_name].run()
    local_dct_orig[arc_name] = oo['chrom_coupling_real'] + 1j * oo['chrom_coupling_imag']

tw_orig = env['lhcb1'].twiss4d(coupling_edw_teng=True, delta0=1e-3)
nlchr_orig = line.get_non_linear_chromaticity(num_delta=20)


rdt_contrib_glob.clear_corrections()
local_dct_bare = {}
for arc_name in arcs:
    oo = integ_dct[arc_name].run()
    local_dct_bare[arc_name] = oo['chrom_coupling_real'] + 1j * oo['chrom_coupling_imag']

tw_bare = env['lhcb1'].twiss4d(coupling_edw_teng=True, delta0=1e-3)
nlchr_bare = line.get_non_linear_chromaticity(num_delta=20)

# Compare in a bar prot the abs of the local integrals before and after the global correction
import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
arc_names = list(opt_dct.keys())
local_opt_vals = [np.abs(local_dct_opt[arc_name]) for arc_name in arc_names]
local_orig_vals = [np.abs(local_dct_orig[arc_name]) for arc_name in arc_names]
local_bare_vals = [np.abs(local_dct_bare[arc_name]) for arc_name in arc_names]
x = np.arange(len(arc_names))
plt.bar(x-0.2, local_orig_vals, width=0.2, label='Original')
plt.bar(x, local_opt_vals, width=0.2, label='After global correction')
# plt.bar(x+0.2, local_bare_vals, width=0.2, label='Bare')
plt.legend()

plt.xticks(x, arc_names)
plt.xlabel('Arc name')
plt.ylabel('Abs of chromatic coupling integral')

plt.figure(2)
plt.plot(nlchr_orig.delta0, [ttt.c_minus for ttt in nlchr_orig.twiss], label='Original')
plt.plot(nlchr_opt.delta0, [ttt.c_minus for ttt in nlchr_opt.twiss], label='After global correction')
plt.plot(nlchr_bare.delta0, [ttt.c_minus for ttt in nlchr_bare.twiss], label='Bare')
plt.xlabel('delta0')
plt.ylabel('C-')
plt.legend()