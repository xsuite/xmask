import xtrack as xt
import xdeps as xd

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

        for rdt_i in self.rdt_indices:
            assert len(rdt_i) == 2
            ii = rdt_i[0]
            jj = rdt_i[1]

            r_ii_jj = (tw_integral.betx ** (ii / 2) * tw_integral.bety ** (jj / 2)
                       * tt_integral[self.multipole])
            self.rdt_terms[rdt_i] = r_ii_jj.sum()

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


# Usage:
rdt_contrib = RDTContrib(env=env,
                         line_name='lhcb1',
                         tw=tw,
                         start='dfxj.4l5',
                         end='dfxj.4r5',
                         correction_knobs=['kcox3.l5', 'kcox3.r5'],
                         multipole='k3l',
                         ip='ip5',
                         rdt_indices=[(0, 4), (4, 0)],
                         generated_knob_name='on_corr_k3_ip5')

rdt_contrib.clear_corrections()
opt = rdt_contrib.correct()

