import xtrack as xt
import xdeps as xd

env = xt.load('collider_00_from_mad.json')

tw = env.lhcb1_co_ref.twiss4d() # Reference twiss

class RDTCorrector:
    def __init__(self, env, tw):
        self.env = env
        self.tw = tw
        self.line = env.lhcb1
        self.start = 'dfxj.4l5'
        self.end = 'dfxj.4r5'
        self.correction_knobs = ['kcox3.l5', 'kcox3.r5']
        self.multipole = 'k3l'
        self.rdt_indices = [(0, 4), (4, 0)]
        self.rdt_terms = {}

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

        ref_names = [nn.replace('/lhcb1', '') for nn in tt_integral.env_name]
        tw_integral = self.tw.rows[ref_names]

        for rdt_i in self.rdt_indices:
            assert len(rdt_i) == 2
            ii = rdt_i[0]
            jj = rdt_i[1]

            r_ii_jj = (tw_integral.betx ** (ii / 2) * tw_integral.bety ** (jj / 2)
                       * tt_integral[self.multipole])
            self.rdt_terms[rdt_i] = r_ii_jj.sum()

        return self.rdt_terms


# Usage:
corrector = RDTCorrector(env, tw)
rdt_terms = corrector.run()
