
import numpy as np
import xtrack as xt
from integral_correction import IntegralCorrection

beam_name = 'b1'
env = xt.load('collider_00_from_mad_with_errors.json')
line = env[f'lhc{beam_name}']

# Reference twiss
env_no_err = xt.load('collider_00_from_mad_no_errors.json')
tw = env_no_err[f'lhc{beam_name}'].twiss4d() # Reference twiss

# Let's have a look at the a3 (skew sextupole)

def chorm_coupling_integrand(tw, tt):
    return tt['k2sl'] * tw.dx * np.sqrt(tw.betx * tw.bety)  * np.exp(1j*(tw.mux - tw.muy))

arc_name = '45'
start = f's.ds.r{arc_name[0]}.{beam_name}'
end = f'e.ds.l{arc_name[1]}.{beam_name}'
correction_knobs = [f'kss.a45{beam_name}']
multipole = 'k2sl'
target_quantities={
    'chrom_coupling_real': lambda tw, tt: chorm_coupling_integrand(tw, tt).real,
    'chrom_coupling_imag': lambda tw, tt: chorm_coupling_integrand(tw, tt).imag
}
generated_knob_name = 'on_corr_k3sl_a45'



tt = line.get_table()
scale_multipole = np.zeros_like(tt.s)
scale_multipole[tt.rows.mask[r'mb.*']] = 1.0 # only bends as sources
scale_multipole[tt.rows.mask[r'mc.*']] = 1.0 # all magnets called mcXXX used as correctors
scale_multipole[tt.rows.mask[r'mss.*']] = 1.0 # all magnets called mssXXX used as correctors

# Usage:
rdt_contrib = IntegralCorrection(
                         line=env['lhcb1'],
                         tw=tw,
                         feed_down=True,
                         start=start,
                         end=end,
                         correction_knobs=correction_knobs,
                         multipole=multipole,
                         ip=None,
                         target_quantities=target_quantities,
                         generated_knob_name='on_corr_k3_ip5',
                         scale_multipole=scale_multipole)
print("Original correction:")
rdt_contrib.print_corrections()

rdt_contrib.clear_corrections()
opt = rdt_contrib.correct()

print("Before setting the knob:")
rdt_contrib.print_corrections()

env[opt.knob_name] = 1.0
print("After setting the knob:")
rdt_contrib.print_corrections()