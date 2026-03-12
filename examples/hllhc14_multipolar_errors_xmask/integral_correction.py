import numpy as np
import xtrack as xt
import xdeps as xd

NORMAL_STRENGTHS_FROM_ATTR=['k0l', 'k1l', 'k2l', 'k3l', 'k4l', 'k5l']
SKEW_STRENGTHS_FROM_ATTR=['k0sl', 'k1sl', 'k2sl', 'k3sl', 'k4sl', 'k5sl']

class IntegralCorrection:
    def __init__(self, line, tw, start, end, correction_knobs,
                 multipole, target_quantities, generated_knob_name,
                 scale_multipole=None, feed_down=True, orbit=None):
        self.env = line.env
        self.tw = tw
        self.line = line
        self.start = start
        self.end = end
        self.correction_knobs = correction_knobs
        self.multipole = multipole
        self.target_quantities = target_quantities
        self.rdt_terms = {}
        self.generated_knob_name = generated_knob_name
        self.scale_multipole = scale_multipole
        self.feed_down = feed_down
        self.orbit = orbit

        self.knob_opt = None

    def clear_corrections(self):
        for kk in self.correction_knobs:
            self.env[kk] = 0.0

    def print_corrections(self):
        for kk in self.correction_knobs:
            print(f"{kk} = {self.env[kk]}")

    def get_corrections(self):
        return {kk: self.env[kk] for kk in self.correction_knobs}

    def run(self):
        if self.line.tracker is None:
            self.line.build_tracker()
        # I do this instead of get_table to be faster
        tt0 = self.line.tracker._tracker_data_base._line_table
        tt = xt.Table(data={'name': tt0['name'], 'env_name': tt0['env_name'],
                            'parent_name': tt0['parent_name'], 's': tt0['s']})
        for kk in NORMAL_STRENGTHS_FROM_ATTR + SKEW_STRENGTHS_FROM_ATTR:
            tt[kk] = np.concatenate([self.line.attr[kk], [0]])

        if self.scale_multipole is not None:
            assert len(self.scale_multipole) == len(tt)
            tt[self.multipole] *= self.scale_multipole

        tt_range = tt.rows[self.start:self.end]

        # Identify elements controlled by correction knobs
        elements_in_range = set(list(tt_range.env_name) + list(tt_range.parent_name))
        correction_elements = []
        for kk in self.correction_knobs:
            for tt in self.line.ref[kk]._find_dependant_targets():
                if isinstance(tt, xd.refs.ItemRef) and tt._key in elements_in_range:
                    correction_elements.append(tt._key)

        mask_corr = tt_range.rows.mask[list(correction_elements)]
        tt_integral = tt_range.rows[(tt_range[self.multipole] != 0) | (mask_corr)]

        tw_integral = self.tw.rows[tt_integral.env_name]
        orbit_integral = None
        if self.orbit is not None:
            assert len(self.orbit) == len(self.tw)
            orbit_integral = self.orbit.rows[tt_integral.env_name]

        for nntq, ttqq in self.target_quantities.items():

            if isinstance(ttqq, str):
                rdts = xt.rdt_first_order_perturbation(
                    rdt=[ttqq],
                    twiss=tw_integral,
                    strengths=tt_integral,
                    feed_down=self.feed_down,
                    orbit=orbit_integral
                )
                integrand = rdts[f"{ttqq}_integrand"]
            else:
                # I assume it's a callable
                integrand = ttqq(tw_integral, tt_integral)

            self.rdt_terms[nntq] = np.abs(integrand.sum())
            self.rdt_terms[nntq+'_integrand'] = integrand
        self.rdt_terms['s'] = tt_integral.s

        return self.rdt_terms

    def _make_optimizer(self):
        action_rdt_contrib = xt.Action(self.run)

        knob_opt = self.env.match_knob(
            knob_name=self.generated_knob_name,
            run=False,
            vary=xt.VaryList(self.correction_knobs, step=1e-5),
            targets=[
                action_rdt_contrib.target(nttqq, 0.0)
                    for nttqq in self.target_quantities.keys()
            ])
        self.knob_opt = knob_opt

    def get_optimizer(self):
        if self.knob_opt is None:
            self._make_optimizer()
        return self.knob_opt

    def correct(self, n_steps=1):

        opt = self.get_optimizer()
        opt.step(n_steps)
        opt.generate_knob()

        return opt