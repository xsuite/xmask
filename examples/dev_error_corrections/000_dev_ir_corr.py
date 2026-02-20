import xtrack as xt
import xdeps as xd
import numpy as np

# TODO:
# - remember to handle left/right sign for different rdt

env = xt.load('collider_00_from_mad_with_errors.json')

# Reference twiss
env_no_err = xt.load('collider_00_from_mad_no_errors.json')
tw = env_no_err.lhcb1.twiss4d() # Reference twiss

class RDTContrib:
    def __init__(self, env, line_name, tw, start, end, correction_knobs,
                 multipole, ip, rdt_indices, generated_knob_name):
        self.env = env
        self.tw = tw
        self.line = env[line_name]
        self.start = start
        self.end = end
        self.correction_knobs = correction_knobs
        self.multipole = multipole
        self.ip = ip
        self.rdt_indices = rdt_indices
        self.rdt_terms = {}
        self.generated_knob_name = generated_knob_name

    def clear_corrections(self):
        for kk in self.correction_knobs:
            self.env[kk] = 0.0

    def print_corrections(self):
        for kk in self.correction_knobs:
            print(f"{kk} = {self.env[kk]}")

    def run(self):
        tt = self.line.get_table(attr=True)
        tt_range = tt.rows[self.start:self.end]

        # Identify elements controlled by correction knobs
        elements_in_range = set(tt_range.env_name)
        correction_elements = []
        for kk in self.correction_knobs:
            for tt in self.line.ref[kk]._find_dependant_targets():
                if isinstance(tt, xd.refs.ItemRef) and tt._key in elements_in_range:
                    correction_elements.append(tt._key)

        mask_corr = tt_range.rows.mask[list(correction_elements)]
        tt_integral = tt_range.rows[(tt_range[self.multipole] != 0) | (mask_corr)]

        tw_integral = self.tw.rows[tt_integral.env_name]

        rdts = xt.rdt_first_order_perturbation(
            rdt=[f"f{ii}0{jj}0" for ii, jj in self.rdt_indices],
            twiss=tw_integral,
            strengths=tt_integral,
            feed_down=False
        )

        mysign = tw_integral.s * 0 + 1
        mysign[tw_integral.rows.mask['.*r5.*']] = -1

        for rdt_i in self.rdt_indices:
            assert len(rdt_i) == 2
            ii = rdt_i[0]
            jj = rdt_i[1]

            # Computation done in old correction routines
            r_ii_jj = (tw_integral.betx ** (ii / 2) * tw_integral.bety ** (jj / 2)
                       * tt_integral[self.multipole]) * mysign
            # self.rdt_terms[str(rdt_i)+'_sum'] = r_ii_jj.sum()
            # self.rdt_terms[str(rdt_i)+'_integrand_loc'] = r_ii_jj

            # integrand = rdts[f"f{ii}0{jj}0_integrand"]
            integrand = r_ii_jj

            self.rdt_terms[rdt_i] = np.abs(integrand.sum())
            self.rdt_terms[str(rdt_i)+'_integrand_rdt'] = rdts[f"f{rdt_i[0]}0{rdt_i[1]}0_integrand"]
        self.rdt_terms['s'] = tt_integral.s

        return self.rdt_terms

    def correct(self, n_steps=1):
        action_rdt_contrib = xt.Action(self.run)

        opt = env.match_knob(
            knob_name=self.generated_knob_name,
            run=False,
            vary=xt.VaryList(rdt_contrib.correction_knobs, step=1e-5),
            targets=[
                action_rdt_contrib.target(rdtind, 0.0) for rdtind in rdt_contrib.rdt_indices
            ])
        opt.step(n_steps)
        opt.generate_knob()

        return opt

# Normal sextupole correction
correction_knobs=['kcsx3.l5', 'kcsx3.r5']
multipole='k2l'
rdt_indices=[(1, 2), (2, 1)]

# # Normal octupole correction
# correction_knobs=['kcox3.l5', 'kcox3.r5']
# multipole='k3l'
# rdt_indices=[(0, 4), (4, 0)]

# Usage:
rdt_contrib = RDTContrib(env=env,
                         line_name='lhcb1',
                         tw=tw,
                         start='dfxj.4l5',
                         end='dfxj.4r5',
                         correction_knobs=correction_knobs,
                         multipole=multipole,
                         ip='ip5',
                         rdt_indices=rdt_indices,
                         generated_knob_name='on_corr_k3_ip5')
print("Original correction:")
rdt_contrib.print_corrections()

rdt_contrib.clear_corrections()
opt = rdt_contrib.correct()

print("Before setting the knob:")
rdt_contrib.print_corrections()

env[opt.knob_name] = 1.0
print("After setting the knob:")
rdt_contrib.print_corrections()

# oo = rdt_contrib.run()
# integrand_rdt = oo['(0, 4)_integrand_rdt']
# integrand_loc = oo['(0, 4)_integrand_loc']
# s = oo['s']
# integrand_rdt *= np.exp(-1j * np.angle(integrand_rdt[0]))

# import matplotlib.pyplot as plt
# plt.close('all')
# plt.figure(1)
# plt.plot(s, integrand_loc/np.max(np.abs(integrand_loc)), label='loc real')
# plt.plot(s, -integrand_rdt.real/np.max(np.abs(integrand_rdt)), label='rdt real')
# plt.plot(s, -integrand_rdt.imag/np.max(np.abs(integrand_rdt)), label='rdt imag')
# plt.plot(s, np.abs(integrand_rdt)/np.max(np.abs(integrand_rdt)), label='rdt abs')
# plt.legend()
# plt.title('Integrand for f0040')

# plt.show()
